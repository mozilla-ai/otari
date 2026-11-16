"""Creating, listing, canceling and resending organization invitations, and noting an unproven address.

With a set clock and an in-memory rate limit store, on SQLite.
"""

import dataclasses
import logging
import uuid
from collections.abc import AsyncIterator, Collection, Iterator, Sequence
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, col

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.adapters.rate_limit_store_adapter import InMemoryRateLimitStore
from gateway.core.config import GatewayConfig
from gateway.core.tokens import hash_token, is_shared_token
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import InvalidEmailError
from gateway.exceptions.organizations_exceptions import (
    AddressAlreadyInvitedError,
    DuplicateInvitationAddressError,
    InvitationLimitCeiling,
    InvitationLimitError,
    InvitationNotFoundError,
    InvitationRequestTooLargeError,
    InvitationResendLimitError,
    InvitationResendTooSoonError,
    InvitationsNeedSsoError,
    MembershipUpdateError,
    NotAuthorizedError,
    OrganizationMemberAlreadyExistsError,
    WorkspaceNotFoundError,
)
from gateway.log_config import logger as gateway_logger
from gateway.models.organizations import OrganizationInvitation
from gateway.models.tenancy import (
    DOMAIN_PROOF_TTL,
    MAX_BULK_INVITATIONS,
    MAX_WORKSPACE_ASSIGNMENTS,
    Organization,
    OrganizationDomain,
    OrganizationMember,
)
from gateway.ports.rate_limit_store_port import RateLimitStorePort, RateLimitWindow
from gateway.repositories.identity import BlockedEmailRepository
from gateway.repositories.organizations import NewInvitation, OrganizationRepositories
from gateway.repositories.organizations.organization_invitation_repository import OrganizationInvitationRepository
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceRepository,
)
from gateway.services.mail import MailDelivery, Mailer, MailMessage
from gateway.services.organizations import DeliveryRule, InvitationDelivery, OrganizationClaimedDomains
from gateway.services.organizations._delivery import MAX_ATTEMPTS
from gateway.services.organizations._invitation_facts import InvitationFactsReader
from gateway.services.organizations._invitation_limits import Admission, InvitationLimits
from gateway.services.organizations._invitations import (
    ActingMember,
    AddressNote,
    InvitationEntry,
    InvitationOffer,
    InvitationPolicy,
    Invitations,
    InvitationState,
    IssuedInvitation,
    OfferedWorkspace,
)

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)
_DELIVERED = MailDelivery(delivered=True, transport="test")
_NOT_DELIVERED = MailDelivery(delivered=False, transport="test", reason="refused")


class _Clock:
    def __init__(self) -> None:
        self.now = _NOW

    def __call__(self) -> datetime:
        return self.now


def _policy(**overrides: object) -> InvitationPolicy:
    values: dict[str, object] = {
        "expires_after": timedelta(days=7),
        "per_address_per_day": 10,
        "per_day": 10,
        "resend_limit": 3,
        "resend_wait": timedelta(hours=1),
    }
    values.update(overrides)
    return InvitationPolicy(**values)  # type: ignore[arg-type]


def _offer(role: str = "member", workspaces: tuple[OfferedWorkspace, ...] = ()) -> InvitationOffer:
    return InvitationOffer(role=role, workspaces=workspaces)  # type: ignore[arg-type]


def _normalized(entry: InvitationEntry) -> InvitationEntry:
    """``entry`` with its address and id replaced, for comparing two invitations apart from those."""
    return dataclasses.replace(entry, email="normalized@example.com", id=uuid.UUID(int=0))


def _config(*, mail_transport: str = "smtp") -> GatewayConfig:
    """A deployment that can deliver email, unless a transport is given that cannot."""
    return GatewayConfig(
        mail_from_email="otari@example.com",
        mail_transport=mail_transport,
        public_base_url="https://otari.example.com",
        smtp_host="smtp.example.com",
    )


class _Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.uow = UnitOfWork(session)
        self.clock = _Clock()
        self.rate_limit_store = InMemoryRateLimitStore()
        self.repositories = OrganizationRepositories.on(self.uow)

    def invitations(self, *, policy: InvitationPolicy | None = None, mailer: Mailer | None = None) -> Invitations:
        domains = OrganizationClaimedDomains(OrganizationDomainRepository(self.uow))
        return Invitations(
            self.uow,
            OrganizationRepositories.on(self.uow),
            clock=self.clock,
            domains=domains,
            mailer=mailer or Mailer(_config()),
            policy=policy or _policy(),
            store=self.rate_limit_store,
        )

    async def organization(self, name: str = "Acme", *, sso_required: bool = False) -> uuid.UUID:
        async with self.uow:
            row = Organization(name=name, slug=uuid.uuid4().hex, sso_required=sso_required)
            self._session.add(row)
            await self._session.flush()
            return row.id

    async def actor(
        self, organization_id: uuid.UUID, *, role: str = "owner", email: str | None = None, full_name: str = "Actor"
    ) -> ActingMember:
        async with self.uow:
            user = await UserRepository(self.uow).create_local_identity(
                full_name=full_name, active_organization_id=organization_id, email=email
            )
            await self.repositories.members.create_membership(
                organization_id=organization_id, user_id=user.id, role=role
            )
            return ActingMember(organization_id=organization_id, user_id=user.id)

    async def workspace(self, organization_id: uuid.UUID, owner_user_id: uuid.UUID, *, name: str = "Main") -> uuid.UUID:
        async with self.uow:
            row = await WorkspaceRepository(self.uow).create_workspace(
                name=name, organization_id=organization_id, created_by_user_id=owner_user_id
            )
            return row.id

    async def invitation(self, invitation_id: uuid.UUID) -> OrganizationInvitation | None:
        self._session.expire_all()
        async with self.uow:
            return await self.repositories.invitations.get(invitation_id)

    async def decline(self, invitation_id: uuid.UUID) -> None:
        async with self.uow:
            row = await self.repositories.invitations.get(invitation_id)
            assert row is not None
            row.declined_at = self.clock.now
            await self._session.flush()

    async def set_send_history(self, invitation_id: uuid.UUID, *, send_count: int) -> None:
        async with self.uow:
            row = await self.repositories.invitations.get(invitation_id)
            assert row is not None
            row.send_count = send_count
            await self._session.flush()

    async def set_delivery_state(
        self, invitation_id: uuid.UUID, *, next_send_at: datetime | None, attempts: int
    ) -> None:
        async with self.uow:
            row = await self.repositories.invitations.get(invitation_id)
            assert row is not None
            row.next_send_at = next_send_at
            row.delivery_attempts = attempts
            await self._session.flush()

    async def set_sso_required(self, organization_id: uuid.UUID, *, value: bool) -> None:
        async with self.uow:
            row = await self.repositories.organizations.get(organization_id)
            assert row is not None
            row.sso_required = value
            await self._session.flush()

    def delivery_with(self, mailer: Mailer) -> InvitationDelivery:
        domains = OrganizationClaimedDomains(OrganizationDomainRepository(self.uow))
        return InvitationDelivery(
            self.uow,
            OrganizationRepositories.on(self.uow),
            rule=DeliveryRule(domains, _Blocklist()),
            facts=InvitationFactsReader(
                domains=domains,
                organizations=OrganizationRepository(self.uow),
                users=UserRepository(self.uow),
                workspaces=WorkspaceRepository(self.uow),
            ),
            mailer=mailer,
            clock=self.clock,
        )


class _Blocklist:
    """Blocks nothing, so the delivery rule's own say is the only thing a resend test has to account for."""

    async def blocked_among(self, emails: Collection[str]) -> set[str]:
        return set()


class _RecordingMailer(Mailer):
    """Records each address sent to and answers with ``result``, with no real socket I/O."""

    def __init__(self, **settings: str) -> None:
        super().__init__(_config(**settings))
        self.sent: list[str] = []
        self.result = _DELIVERED

    async def send(
        self, *, to: str, message: MailMessage, reply_to: str | None = None, sender_name: str | None = None
    ) -> MailDelivery:
        self.sent.append(to)
        return self.result


class _SpyRateLimitStore:
    """Wraps a real store and counts calls to ``hit``, to prove a refused caller's addresses were never admitted."""

    def __init__(self, inner: RateLimitStorePort) -> None:
        self._inner = inner
        self.hit_calls = 0

    async def hit(self, key: str, limit: int, window_sec: float, cost: int = 1) -> RateLimitWindow:
        self.hit_calls += 1
        return await self._inner.hit(key, limit, window_sec, cost)

    async def settle(self, key: str, handle: str, cost: int) -> None:
        await self._inner.settle(key, handle, cost)

    async def acquire(self, key: str, limit: int, lease_sec: float) -> str | None:
        return await self._inner.acquire(key, limit, lease_sec)

    async def release(self, key: str, lease: str) -> None:
        await self._inner.release(key, lease)

    async def aclose(self) -> None:
        await self._inner.aclose()


class _RepositoryThatFailsAfterWriting(OrganizationInvitationRepository):
    """Succeeds its batched write and then raises, to simulate a late failure before commit."""

    async def add_many(self, news: Sequence[NewInvitation]) -> list[OrganizationInvitation]:
        await super().add_many(news)
        msg = "simulated failure after the write"
        raise RuntimeError(msg)


@pytest_asyncio.fixture
async def store() -> AsyncIterator[_Store]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield _Store(session)
    finally:
        await engine.dispose()


@pytest.fixture(autouse=True)
def _no_email_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OTARI_MAIL_FROM_EMAIL", "OTARI_MAIL_TRANSPORT", "OTARI_PUBLIC_BASE_URL", "OTARI_SMTP_HOST"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture the ``gateway`` logger, which does not propagate."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


async def test_invite_writes_one_due_row_and_returns_no_link_where_email_is_delivered(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(mailer=Mailer(_config(mail_transport="smtp")))

    issued = await service.invite(by, "ada@example.com", _offer())

    assert issued.accept_link is None
    row = await store.invitation(issued.invitation.id)
    assert row is not None
    assert row.next_send_at == store.clock.now


async def test_invite_returns_a_hand_carried_link_where_email_is_not_delivered(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(mailer=Mailer(_config(mail_transport="none")))

    issued = await service.invite(by, "ada@example.com", _offer())

    assert issued.accept_link is not None
    row = await store.invitation(issued.invitation.id)
    assert row is not None
    assert row.next_send_at is None
    token = issued.accept_link.rsplit("token=", 1)[-1]
    assert hash_token(token) == row.token_hash
    assert is_shared_token(token)


async def test_a_returned_link_is_never_on_a_due_row(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(mailer=Mailer(_config(mail_transport="none")))
    await service.invite(by, "ada@example.com", _offer())

    async with store.uow:
        due = await store.repositories.invitations.claim_due_send(
            now=store.clock.now,
            held_until=store.clock.now + timedelta(minutes=10),
            max_attempts=3,
            token_hash="probe",
        )
    assert due is None


async def test_invite_does_the_same_work_whatever_the_delivery_rule_says(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations()

    ordinary = await service.invite(by, "ordinary@example.com", _offer())

    other_organization_id = await store.organization("Other")
    async with store.uow:
        store._session.add(
            OrganizationDomain(
                organization_id=other_organization_id,
                domain="proven.test",
                verification_token=uuid.uuid4().hex,
                verified_at=store.clock.now,
            )
        )
        await store._session.flush()
    proven_domain = await service.invite(by, "someone@proven.test", _offer())

    async with store.uow:
        await BlockedEmailRepository(store.uow).add(
            pattern="blocked@example.com", reason=None, created_by_user_id=by.user_id
        )
    blocked = await service.invite(by, "blocked@example.com", _offer())

    results = (ordinary, proven_domain, blocked)
    normalized = {_normalized(result.invitation) for result in results}
    assert len(normalized) == 1
    assert [result.accept_link is None for result in results] == [True, True, True]


async def test_resend_does_the_same_work_whatever_the_delivery_rule_says(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations()

    ordinary = await service.invite(by, "ordinary@example.com", _offer())

    other_organization_id = await store.organization("Other")
    async with store.uow:
        store._session.add(
            OrganizationDomain(
                organization_id=other_organization_id,
                domain="proven.test",
                verification_token=uuid.uuid4().hex,
                verified_at=store.clock.now,
            )
        )
        await store._session.flush()
    proven_domain = await service.invite(by, "someone@proven.test", _offer())

    async with store.uow:
        await BlockedEmailRepository(store.uow).add(
            pattern="blocked@example.com", reason=None, created_by_user_id=by.user_id
        )
    blocked = await service.invite(by, "blocked@example.com", _offer())

    store.clock.now += timedelta(hours=2)
    resent = [await service.resend(by, issued.invitation.id) for issued in (ordinary, proven_domain, blocked)]

    normalized = {_normalized(result.invitation) for result in resent}
    assert len(normalized) == 1
    assert [result.accept_link is None for result in resent] == [True, True, True]


async def test_entries_hide_how_delivery_went(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations()

    delivered = await service.invite(by, "delivered@example.com", _offer())
    skipped = await service.invite(by, "skipped@example.com", _offer())
    failed = await service.invite(by, "failed@example.com", _offer())

    await store.set_delivery_state(delivered.invitation.id, next_send_at=None, attempts=1)
    await store.set_delivery_state(skipped.invitation.id, next_send_at=store.clock.now, attempts=0)
    await store.set_delivery_state(failed.invitation.id, next_send_at=None, attempts=3)

    entries, _ = await service.page(by, skip=0, limit=10)
    assert len({_normalized(entry) for entry in entries}) == 1

    entry_fields = {field.name for field in dataclasses.fields(InvitationEntry)}
    issued_fields = {field.name for field in dataclasses.fields(IssuedInvitation)}
    assert "next_send_at" not in entry_fields
    assert "delivery_attempts" not in entry_fields
    assert "next_send_at" not in issued_fields
    assert "delivery_attempts" not in issued_fields


async def test_invite_refuses_a_member_viewer_and_a_stranger(store: _Store) -> None:
    organization_id = await store.organization()
    member = await store.actor(organization_id, role="member")
    viewer = await store.actor(organization_id, role="viewer")
    stranger = ActingMember(organization_id=organization_id, user_id=uuid.uuid4())

    for by in (member, viewer, stranger):
        with pytest.raises(NotAuthorizedError):
            await store.invitations().invite(by, "ada@example.com", _offer())


async def test_invite_refuses_after_the_caller_is_demoted(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id, role="owner")
    service = store.invitations()
    await service.invite(by, "first@example.com", _offer())

    async with store.uow:
        held = await store.repositories.members.get_active_by_organization_and_user(organization_id, by.user_id)
    assert held is not None
    assert held.role == "owner"

    # Updates the row directly, leaving `held` (kept alive here) stale in the identity map,
    # the way a concurrent transaction's own commit would.
    async with store.uow:
        await store._session.execute(
            update(OrganizationMember)
            .where(col(OrganizationMember.id) == held.id)
            .values(role="member")
            .execution_options(synchronize_session=False)
        )

    with pytest.raises(NotAuthorizedError):
        await service.invite(by, "second@example.com", _offer())


async def test_invite_replaces_a_row_that_cleanup_already_deleted(
    store: _Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(policy=_policy(expires_after=timedelta(hours=1)))
    first = await service.invite(by, "ghost@example.com", _offer())
    store.clock.now += timedelta(hours=2)

    real_admit = InvitationLimits.admit

    # Deletes the expired row between the read that marks it as replaceable and the write that
    # deletes it, the way an hourly cleanup sweep or a cancel committing in between would.
    async def admit_once_the_replaced_row_is_already_gone(
        self: InvitationLimits, *, account_id: uuid.UUID, email: str, organization_id: uuid.UUID
    ) -> Admission | InvitationLimitCeiling:
        async with store.uow:
            await store.repositories.invitations.delete_in_organization(first.invitation.id, organization_id)
        return await real_admit(self, account_id=account_id, email=email, organization_id=organization_id)

    monkeypatch.setattr(InvitationLimits, "admit", admit_once_the_replaced_row_is_already_gone)

    second = await service.invite(by, "ghost@example.com", _offer())

    assert second.invitation.id != first.invitation.id


async def test_only_an_owner_offers_the_owner_role(store: _Store) -> None:
    organization_id = await store.organization()
    admin = await store.actor(organization_id, role="admin")
    owner = await store.actor(organization_id, role="owner")
    service = store.invitations()

    with pytest.raises(MembershipUpdateError):
        await service.invite(admin, "ada@example.com", _offer(role="owner"))

    issued = await service.invite(owner, "grace@example.com", _offer(role="owner"))
    assert issued.invitation.role == "owner"


async def test_invite_refuses_where_the_organization_requires_sso(store: _Store) -> None:
    organization_id = await store.organization(sso_required=True)
    by = await store.actor(organization_id)

    with pytest.raises(InvitationsNeedSsoError):
        await store.invitations().invite(by, "ada@example.com", _offer())


async def test_invite_refuses_a_workspace_of_another_organization(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    other_organization_id = await store.organization("Other")
    foreign_workspace_id = await store.workspace(other_organization_id, by.user_id)

    with pytest.raises(WorkspaceNotFoundError):
        await store.invitations().invite(
            by,
            "ada@example.com",
            _offer(workspaces=(OfferedWorkspace(role="member", workspace_id=foreign_workspace_id),)),
        )


async def test_invite_refuses_a_workspace_named_twice(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    workspace_id = await store.workspace(organization_id, by.user_id)
    offer = _offer(
        workspaces=(
            OfferedWorkspace(role="member", workspace_id=workspace_id),
            OfferedWorkspace(role="admin", workspace_id=workspace_id),
        )
    )

    with pytest.raises(InvitationRequestTooLargeError):
        await store.invitations().invite(by, "ada@example.com", offer)


async def test_invite_refuses_more_than_50_workspaces(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    workspaces = tuple(
        OfferedWorkspace(role="member", workspace_id=uuid.uuid4()) for _ in range(MAX_WORKSPACE_ASSIGNMENTS + 1)
    )

    with pytest.raises(InvitationRequestTooLargeError):
        await store.invitations().invite(by, "ada@example.com", _offer(workspaces=workspaces))


async def test_invite_many_refuses_more_than_100_addresses(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    emails = [f"user{i}@example.com" for i in range(MAX_BULK_INVITATIONS + 1)]

    with pytest.raises(InvitationRequestTooLargeError):
        await store.invitations().invite_many(by, emails, _offer())


async def test_invite_refuses_an_active_member_in_any_spelling(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    await store.actor(organization_id, role="member", email="grace@example.com")

    with pytest.raises(OrganizationMemberAlreadyExistsError):
        await store.invitations().invite(by, "GRACE@EXAMPLE.com", _offer())


async def test_invite_refuses_a_pending_invitation_even_when_declined(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations()
    issued = await service.invite(by, "ada@example.com", _offer())
    await store.decline(issued.invitation.id)

    with pytest.raises(AddressAlreadyInvitedError):
        await service.invite(by, "ada@example.com", _offer())


async def test_invite_replaces_an_expired_invitation(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(policy=_policy(expires_after=timedelta(hours=1)))
    first = await service.invite(by, "ada@example.com", _offer())

    store.clock.now += timedelta(hours=2)
    second = await service.invite(by, "ada@example.com", _offer())

    assert second.invitation.id != first.invitation.id
    assert second.invitation.state == InvitationState.PENDING
    row = await store.invitation(second.invitation.id)
    assert row is not None
    assert row.declined_at is None


async def test_invite_replaces_an_expired_declined_invitation(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(policy=_policy(expires_after=timedelta(hours=1)))
    first = await service.invite(by, "ada@example.com", _offer())
    await store.decline(first.invitation.id)

    store.clock.now += timedelta(hours=2)
    second = await service.invite(by, "ada@example.com", _offer())

    assert second.invitation.id != first.invitation.id
    assert second.invitation.state == InvitationState.PENDING
    row = await store.invitation(second.invitation.id)
    assert row is not None
    assert row.declined_at is None


async def test_invite_many_reports_each_refusal_and_invites_the_rest(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations()
    await service.invite(by, "existing@example.com", _offer())

    result = await service.invite_many(
        by,
        ["good-one@example.com", "not-an-address", "existing@example.com", "good-two@example.com"],
        _offer(),
    )

    assert [issued.invitation.email for issued in result.issued] == ["good-one@example.com", "good-two@example.com"]
    assert [refused.email for refused in result.refused] == ["not-an-address", "existing@example.com"]
    assert isinstance(result.refused[0].refusal, InvalidEmailError)
    assert isinstance(result.refused[1].refusal, AddressAlreadyInvitedError)


async def test_invite_many_refuses_a_second_spelling_of_one_address(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)

    result = await store.invitations().invite_many(
        by, ["Ada@Example.com", "ADA@EXAMPLE.COM", "other@example.com"], _offer()
    )

    assert [issued.invitation.email for issued in result.issued] == ["ada@example.com", "other@example.com"]
    assert result.issued[0].email == "Ada@Example.com"
    [refused] = result.refused
    assert refused.email == "ADA@EXAMPLE.COM"
    assert isinstance(refused.refusal, DuplicateInvitationAddressError)


async def test_invite_raises_the_refusal_invite_many_reports(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations()
    await service.invite(by, "existing@example.com", _offer())

    result = await service.invite_many(by, ["existing@example.com"], _offer())
    [refused] = result.refused

    with pytest.raises(type(refused.refusal)) as excinfo:
        await service.invite(by, "existing@example.com", _offer())
    assert str(excinfo.value) == str(refused.refusal)


async def test_invite_many_refuses_the_rest_once_the_count_runs_out(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(policy=_policy(per_day=2, per_address_per_day=10))

    result = await service.invite_many(by, ["one@example.com", "two@example.com", "three@example.com"], _offer())

    assert len(result.issued) == 2
    [refused] = result.refused
    assert refused.email == "three@example.com"
    assert isinstance(refused.refusal, InvitationLimitError)


async def test_a_canceled_invitation_keeps_its_count(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(policy=_policy(per_address_per_day=1))
    issued = await service.invite(by, "ada@example.com", _offer())

    async with store.uow:
        await store.repositories.invitations.delete_in_organization(issued.invitation.id, organization_id)

    with pytest.raises(InvitationLimitError):
        await service.invite(by, "ada@example.com", _offer())


async def test_every_count_comes_back_when_the_write_fails_after_inserting(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    repositories = dataclasses.replace(store.repositories, invitations=_RepositoryThatFailsAfterWriting(store.uow))
    service = Invitations(
        store.uow,
        repositories,
        clock=store.clock,
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(store.uow)),
        mailer=Mailer(_config()),
        policy=_policy(per_day=2, per_address_per_day=10),
        store=store.rate_limit_store,
    )

    with pytest.raises(RuntimeError):
        await service.invite_many(by, ["one@example.com", "two@example.com"], _offer())

    # Neither address's row survived the rollback, so neither count should have either.
    probe = await InvitationLimits(store.rate_limit_store, per_day=1, per_address_per_day=10).admit(
        account_id=by.user_id, email="probe@example.com", organization_id=organization_id
    )
    assert isinstance(probe, Admission)


async def test_a_viewers_bulk_call_never_touches_the_rate_limit_store(store: _Store) -> None:
    organization_id = await store.organization()
    viewer = await store.actor(organization_id, role="viewer")
    spy_store = _SpyRateLimitStore(store.rate_limit_store)
    service = Invitations(
        store.uow,
        store.repositories,
        clock=store.clock,
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(store.uow)),
        mailer=Mailer(_config()),
        policy=_policy(),
        store=spy_store,
    )

    with pytest.raises(NotAuthorizedError):
        await service.invite_many(viewer, ["one@example.com", "two@example.com", "three@example.com"], _offer())

    assert spy_store.hit_calls == 0


async def test_the_owner_role_offered_by_a_non_owner_is_never_counted(store: _Store) -> None:
    organization_id = await store.organization()
    admin = await store.actor(organization_id, role="admin")
    spy_store = _SpyRateLimitStore(store.rate_limit_store)
    service = Invitations(
        store.uow,
        store.repositories,
        clock=store.clock,
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(store.uow)),
        mailer=Mailer(_config()),
        policy=_policy(),
        store=spy_store,
    )

    with pytest.raises(MembershipUpdateError):
        await service.invite_many(
            admin, ["one@example.com", "two@example.com", "three@example.com"], _offer(role="owner")
        )

    assert spy_store.hit_calls == 0


async def test_an_already_a_member_or_already_invited_refusal_is_never_counted(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    spy_store = _SpyRateLimitStore(store.rate_limit_store)
    service = Invitations(
        store.uow,
        store.repositories,
        clock=store.clock,
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(store.uow)),
        mailer=Mailer(_config()),
        policy=_policy(per_day=100, per_address_per_day=100),
        store=spy_store,
    )

    member_address = "member@example.com"
    await store.actor(organization_id, role="member", email=member_address)

    # Seeded straight through the repository, so setting it up touches no rate limit count.
    already_invited_address = "already-invited@example.com"
    async with store.uow:
        await store.repositories.invitations.add_many(
            [
                NewInvitation(
                    created_at=store.clock.now,
                    email=already_invited_address,
                    expires_at=store.clock.now + timedelta(days=7),
                    invited_by_user_id=None,
                    next_send_at=None,
                    organization_id=organization_id,
                    role="member",
                    token_hash="seed",
                    workspace_assignments=[],
                )
            ]
        )

    result = await service.invite_many(by, [member_address, already_invited_address], _offer())

    assert result.issued == []
    refusals = {refused.email: refused.refusal for refused in result.refused}
    assert isinstance(refusals[member_address], OrganizationMemberAlreadyExistsError)
    assert isinstance(refusals[already_invited_address], AddressAlreadyInvitedError)

    # Both refusals were decided by a read, before either address was ever counted.
    assert spy_store.hit_calls == 0


async def test_a_bulk_call_refused_by_sso_is_never_counted(store: _Store) -> None:
    organization_id = await store.organization(sso_required=True)
    by = await store.actor(organization_id)
    spy_store = _SpyRateLimitStore(store.rate_limit_store)
    service = Invitations(
        store.uow,
        store.repositories,
        clock=store.clock,
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(store.uow)),
        mailer=Mailer(_config()),
        policy=_policy(),
        store=spy_store,
    )

    with pytest.raises(InvitationsNeedSsoError):
        await service.invite_many(by, [f"user{i}@example.com" for i in range(5)], _offer())

    assert spy_store.hit_calls == 0


async def test_a_bulk_call_refused_by_a_missing_workspace_is_never_counted(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    other_organization_id = await store.organization("Other")
    foreign_workspace_id = await store.workspace(other_organization_id, by.user_id)
    spy_store = _SpyRateLimitStore(store.rate_limit_store)
    service = Invitations(
        store.uow,
        store.repositories,
        clock=store.clock,
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(store.uow)),
        mailer=Mailer(_config()),
        policy=_policy(),
        store=spy_store,
    )

    with pytest.raises(WorkspaceNotFoundError):
        await service.invite_many(
            by,
            [f"user{i}@example.com" for i in range(5)],
            _offer(workspaces=(OfferedWorkspace(role="member", workspace_id=foreign_workspace_id),)),
        )

    assert spy_store.hit_calls == 0


async def test_a_declined_invitation_is_listed_as_pending(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations()
    issued = await service.invite(by, "ada@example.com", _offer())
    await store.decline(issued.invitation.id)

    entries, total = await service.page(by, skip=0, limit=10)

    [entry] = entries
    assert entry.email == "ada@example.com"
    assert entry.state == InvitationState.PENDING
    assert total == 1


async def test_page_keeps_the_order_the_repository_gives(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(policy=_policy(expires_after=timedelta(hours=1)))

    await service.invite(by, "first@example.com", _offer())
    store.clock.now += timedelta(hours=2)
    await service.invite(by, "second@example.com", _offer())

    entries, _ = await service.page(by, skip=0, limit=10)

    assert [entry.email for entry in entries] == ["first@example.com", "second@example.com"]
    assert entries[0].state == InvitationState.EXPIRED
    assert entries[1].state == InvitationState.PENDING


async def test_page_shows_expired_once_the_lifetime_has_passed(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(policy=_policy(expires_after=timedelta(hours=1)))
    issued = await service.invite(by, "ada@example.com", _offer())

    [before], _ = await service.page(by, skip=0, limit=10)
    assert before.state == InvitationState.PENDING

    store.clock.now += timedelta(hours=2)
    [after], _ = await service.page(by, skip=0, limit=10)
    assert after.state == InvitationState.EXPIRED
    assert after.id == issued.invitation.id


async def test_a_workspace_offer_round_trips_through_the_stored_row(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    workspace_id = await store.workspace(organization_id, by.user_id)
    offer = _offer(workspaces=(OfferedWorkspace(role="admin", workspace_id=workspace_id),))
    service = store.invitations()

    issued = await service.invite(by, "ada@example.com", offer)
    assert issued.invitation.workspaces == (OfferedWorkspace(role="admin", workspace_id=workspace_id),)

    entries, _ = await service.page(by, skip=0, limit=10)
    [entry] = entries
    assert entry.workspaces == (OfferedWorkspace(role="admin", workspace_id=workspace_id),)


async def test_page_refuses_a_viewer(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id, role="viewer")

    with pytest.raises(NotAuthorizedError):
        await store.invitations().page(by, skip=0, limit=10)


async def test_page_returns_only_the_callers_organization(store: _Store) -> None:
    organization_id = await store.organization()
    other_organization_id = await store.organization("Other")
    by = await store.actor(organization_id)
    other_by = await store.actor(other_organization_id)
    service = store.invitations()
    await service.invite(by, "mine@example.com", _offer())
    await service.invite(other_by, "theirs@example.com", _offer())

    entries, total = await service.page(by, skip=0, limit=10)

    assert [entry.email for entry in entries] == ["mine@example.com"]
    assert total == 1


async def test_page_refuses_a_negative_skip(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)

    with pytest.raises(ValueError):
        await store.invitations().page(by, skip=-1, limit=10)


async def test_page_bounds_a_non_positive_or_excessive_limit(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations()
    await service.invite(by, "ada@example.com", _offer())

    entries, total = await service.page(by, skip=0, limit=0)
    assert len(entries) == 1
    assert total == 1

    entries, total = await service.page(by, skip=0, limit=10_000_000)
    assert len(entries) == 1
    assert total == 1


async def test_resend_available_at_follows_the_wait_and_the_limit(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(policy=_policy(resend_limit=2, resend_wait=timedelta(hours=3)))
    issued = await service.invite(by, "ada@example.com", _offer())

    [entry] = (await service.page(by, skip=0, limit=10))[0]
    assert entry.resend_available_at == store.clock.now + timedelta(hours=3)

    await store.set_send_history(issued.invitation.id, send_count=3)

    [entry] = (await service.page(by, skip=0, limit=10))[0]
    assert entry.resend_available_at is None


async def test_cancel_removes_the_invitation_and_its_link(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(mailer=Mailer(_config(mail_transport="none")))
    issued = await service.invite(by, "ada@example.com", _offer())
    assert issued.accept_link is not None
    token_hash = hash_token(issued.accept_link.rsplit("token=", 1)[-1])

    await service.cancel(by, issued.invitation.id)

    assert await store.invitation(issued.invitation.id) is None
    async with store.uow:
        assert await store.repositories.invitations.get_by_token_hash(token_hash) is None


async def test_cancel_of_another_organizations_invitation_is_not_found(store: _Store) -> None:
    organization_id = await store.organization()
    other_organization_id = await store.organization("Other")
    by = await store.actor(organization_id)
    other_by = await store.actor(other_organization_id)
    service = store.invitations()
    issued = await service.invite(by, "ada@example.com", _offer())

    with pytest.raises(InvitationNotFoundError):
        await service.cancel(other_by, issued.invitation.id)


async def test_an_admin_cannot_cancel_an_owner_invitation(store: _Store) -> None:
    organization_id = await store.organization()
    owner = await store.actor(organization_id, role="owner")
    admin = await store.actor(organization_id, role="admin")
    service = store.invitations()
    issued = await service.invite(owner, "ada@example.com", _offer(role="owner"))

    with pytest.raises(MembershipUpdateError):
        await service.cancel(admin, issued.invitation.id)


async def test_an_admin_cannot_resend_an_owner_invitation(store: _Store) -> None:
    organization_id = await store.organization()
    owner = await store.actor(organization_id, role="owner")
    admin = await store.actor(organization_id, role="admin")
    service = store.invitations()
    issued = await service.invite(owner, "ada@example.com", _offer(role="owner"))

    with pytest.raises(MembershipUpdateError):
        await service.resend(admin, issued.invitation.id)


async def test_an_owner_can_cancel_and_resend_an_owner_invitation(store: _Store) -> None:
    organization_id = await store.organization()
    owner = await store.actor(organization_id, role="owner")
    service = store.invitations()
    issued = await service.invite(owner, "ada@example.com", _offer(role="owner"))

    store.clock.now += timedelta(hours=2)
    resent = await service.resend(owner, issued.invitation.id)
    assert resent.invitation.role == "owner"

    await service.cancel(owner, resent.invitation.id)
    assert await store.invitation(resent.invitation.id) is None


async def test_cancel_refuses_a_member(store: _Store) -> None:
    organization_id = await store.organization()
    owner = await store.actor(organization_id, role="owner")
    member = await store.actor(organization_id, role="member")
    issued = await store.invitations().invite(owner, "ada@example.com", _offer())

    with pytest.raises(NotAuthorizedError):
        await store.invitations().cancel(member, issued.invitation.id)


async def test_resend_replaces_the_link(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(mailer=Mailer(_config(mail_transport="none")))
    issued = await service.invite(by, "ada@example.com", _offer())
    assert issued.accept_link is not None
    old_hash = hash_token(issued.accept_link.rsplit("token=", 1)[-1])

    store.clock.now += timedelta(hours=2)
    resent = await service.resend(by, issued.invitation.id)

    assert resent.accept_link is not None
    assert resent.accept_link != issued.accept_link
    async with store.uow:
        assert await store.repositories.invitations.get_by_token_hash(old_hash) is None
        new_hash = hash_token(resent.accept_link.rsplit("token=", 1)[-1])
        assert await store.repositories.invitations.get_by_token_hash(new_hash) is not None


async def test_resend_after_three_failed_sends_sends_again(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    mailer = _RecordingMailer()
    service = store.invitations(mailer=mailer)
    delivery = store.delivery_with(mailer)
    issued = await service.invite(by, "ada@example.com", _offer())

    mailer.result = _NOT_DELIVERED
    for _ in range(MAX_ATTEMPTS):
        assert await delivery.send_due() is True
        store.clock.now += timedelta(hours=1)
    assert len(mailer.sent) == MAX_ATTEMPTS
    row = await store.invitation(issued.invitation.id)
    assert row is not None
    assert row.next_send_at is None

    await service.resend(by, issued.invitation.id)

    mailer.result = _DELIVERED
    assert await delivery.send_due() is True
    assert mailer.sent == ["ada@example.com"] * (MAX_ATTEMPTS + 1)


async def test_a_send_in_flight_records_nothing_after_a_resend(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    mailer = _RecordingMailer()
    service = store.invitations(mailer=mailer)
    delivery = store.delivery_with(mailer)
    issued = await service.invite(by, "ada@example.com", _offer())

    async with store.uow:
        claim = await store.repositories.invitations.claim_due_send(
            now=store.clock.now,
            held_until=store.clock.now + timedelta(minutes=10),
            max_attempts=MAX_ATTEMPTS,
            token_hash="in-flight",
        )
    assert claim is not None

    store.clock.now += timedelta(hours=2)
    await service.resend(by, issued.invitation.id)

    async with store.uow:
        assert await store.repositories.invitations.finish_send(issued.invitation.id, token_hash="in-flight") is False

    mailer.result = _DELIVERED
    assert await delivery.send_due() is True
    assert mailer.sent == ["ada@example.com"]


async def test_resend_after_email_is_lost_during_a_send(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(mailer=Mailer(_config(mail_transport="smtp")))
    issued = await service.invite(by, "ada@example.com", _offer())

    async with store.uow:
        claim = await store.repositories.invitations.claim_due_send(
            now=store.clock.now,
            held_until=store.clock.now + timedelta(minutes=10),
            max_attempts=MAX_ATTEMPTS,
            token_hash="claim-a",
        )
    assert claim is not None

    store.clock.now += timedelta(hours=2)
    service_without_email = store.invitations(mailer=Mailer(_config(mail_transport="none")))
    resent = await service_without_email.resend(by, issued.invitation.id)
    assert resent.accept_link is not None

    row = await store.invitation(issued.invitation.id)
    assert row is not None
    assert row.next_send_at is None
    assert row.token_hash == hash_token(resent.accept_link.rsplit("token=", 1)[-1])

    store.clock.now += timedelta(minutes=20)
    async with store.uow:
        assert (
            await store.repositories.invitations.claim_due_send(
                now=store.clock.now,
                held_until=store.clock.now + timedelta(minutes=10),
                max_attempts=MAX_ATTEMPTS,
                token_hash="probe",
            )
            is None
        )
        assert await store.repositories.invitations.finish_send(issued.invitation.id, token_hash="claim-a") is False
        assert (
            await store.repositories.invitations.retry_send(
                issued.invitation.id, token_hash="claim-a", next_send_at=store.clock.now + timedelta(minutes=5)
            )
            is False
        )


async def test_resend_after_email_is_configured_makes_the_row_due(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service_without_email = store.invitations(mailer=Mailer(_config(mail_transport="none")))
    issued = await service_without_email.invite(by, "ada@example.com", _offer())
    assert issued.accept_link is not None
    old_hash = hash_token(issued.accept_link.rsplit("token=", 1)[-1])

    store.clock.now += timedelta(hours=2)
    service_with_email = store.invitations(mailer=Mailer(_config(mail_transport="smtp")))
    resent = await service_with_email.resend(by, issued.invitation.id)

    assert resent.accept_link is None
    row = await store.invitation(issued.invitation.id)
    assert row is not None
    assert row.next_send_at == store.clock.now
    async with store.uow:
        assert await store.repositories.invitations.get_by_token_hash(old_hash) is None


async def test_resend_of_an_expired_invitation_is_pending_and_sent(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    policy = _policy(expires_after=timedelta(hours=1))
    service = store.invitations(policy=policy)
    issued = await service.invite(by, "ada@example.com", _offer())
    store.clock.now += timedelta(hours=2)
    [before], _ = await service.page(by, skip=0, limit=10)
    assert before.state == InvitationState.EXPIRED

    resent = await service.resend(by, issued.invitation.id)

    assert resent.invitation.state == InvitationState.PENDING
    assert resent.invitation.expires_at == store.clock.now + policy.expires_after
    row = await store.invitation(issued.invitation.id)
    assert row is not None
    assert row.next_send_at == store.clock.now


async def test_resend_of_a_declined_invitation_sends_nothing(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    mailer = _RecordingMailer()
    service = store.invitations(mailer=mailer)
    delivery = store.delivery_with(mailer)
    issued = await service.invite(by, "ada@example.com", _offer())
    await store.decline(issued.invitation.id)

    store.clock.now += timedelta(hours=2)
    resent = await service.resend(by, issued.invitation.id)
    assert resent.invitation.state == InvitationState.PENDING

    assert await delivery.send_due() is True
    assert mailer.sent == []
    row = await store.invitation(issued.invitation.id)
    assert row is not None
    assert row.next_send_at is None


async def test_resend_waits_and_stops_at_the_limit(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    policy = _policy(resend_limit=2, resend_wait=timedelta(hours=1), per_day=100, per_address_per_day=100)
    spy_store = _SpyRateLimitStore(store.rate_limit_store)
    service = Invitations(
        store.uow,
        store.repositories,
        clock=store.clock,
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(store.uow)),
        mailer=Mailer(_config()),
        policy=policy,
        store=spy_store,
    )
    issued = await service.invite(by, "ada@example.com", _offer())
    spy_store.hit_calls = 0

    with pytest.raises(InvitationResendTooSoonError) as too_soon:
        await service.resend(by, issued.invitation.id)
    assert "UTC" in str(too_soon.value)
    assert spy_store.hit_calls == 0

    store.clock.now += policy.resend_wait
    await service.resend(by, issued.invitation.id)
    store.clock.now += policy.resend_wait
    await service.resend(by, issued.invitation.id)

    spy_store.hit_calls = 0
    store.clock.now += policy.resend_wait
    with pytest.raises(InvitationResendLimitError):
        await service.resend(by, issued.invitation.id)
    assert spy_store.hit_calls == 0


async def test_resend_counts_against_the_caller(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    policy = _policy(per_day=2, per_address_per_day=2, resend_wait=timedelta(milliseconds=1))
    service = store.invitations(policy=policy)
    issued = await service.invite(by, "ada@example.com", _offer())

    store.clock.now += timedelta(milliseconds=1)
    await service.resend(by, issued.invitation.id)

    store.clock.now += timedelta(milliseconds=1)
    with pytest.raises(InvitationLimitError):
        await service.resend(by, issued.invitation.id)


async def test_every_count_comes_back_when_a_resends_commit_fails_after_its_write_matches(
    store: _Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    policy = _policy(per_day=2, per_address_per_day=2)
    issued = await store.invitations(policy=policy).invite(by, "ada@example.com", _offer())
    store.clock.now += timedelta(hours=2)

    # request_send's own UPDATE matches, but the commit that would make it durable never runs.
    async def commit_that_fails(self: AsyncSession) -> None:
        msg = "simulated failure during commit, after the resend's write matched"
        raise RuntimeError(msg)

    monkeypatch.setattr(AsyncSession, "commit", commit_that_fails)
    service = store.invitations(policy=policy)

    with pytest.raises(RuntimeError):
        await service.resend(by, issued.invitation.id)

    monkeypatch.undo()

    # The failed resend's own admission came back along with its rolled-back write, so the one
    # already spent by the invite still leaves exactly one more admission of room, not zero.
    probe = await InvitationLimits(store.rate_limit_store, per_day=2, per_address_per_day=2).admit(
        account_id=by.user_id, email="ada@example.com", organization_id=organization_id
    )
    assert isinstance(probe, Admission)


async def test_resend_makes_the_caller_the_inviter(store: _Store) -> None:
    organization_id = await store.organization()
    owner = await store.actor(organization_id, role="owner", full_name="Owner")
    second_admin = await store.actor(organization_id, role="admin", full_name="Second Admin")
    service = store.invitations()
    issued = await service.invite(owner, "ada@example.com", _offer())

    store.clock.now += timedelta(hours=2)
    resent = await service.resend(second_admin, issued.invitation.id)

    row = await store.invitation(resent.invitation.id)
    assert row is not None
    assert row.invited_by_user_id == second_admin.user_id

    async with store.uow:
        facts_reader = InvitationFactsReader(
            domains=OrganizationClaimedDomains(OrganizationDomainRepository(store.uow)),
            organizations=OrganizationRepository(store.uow),
            users=UserRepository(store.uow),
            workspaces=WorkspaceRepository(store.uow),
        )
        facts = await facts_reader.read(row, now=store.clock.now)
    assert facts is not None
    assert facts.inviter_name == "Second Admin"


async def test_resend_refuses_where_the_organization_requires_sso(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations()
    issued = await service.invite(by, "ada@example.com", _offer())
    await store.set_sso_required(organization_id, value=True)

    with pytest.raises(InvitationsNeedSsoError):
        await service.resend(by, issued.invitation.id)


async def test_resend_refuses_after_the_caller_is_removed(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id, role="owner")
    service = store.invitations()
    issued = await service.invite(by, "ada@example.com", _offer())

    async with store.uow:
        member = await store.repositories.members.get_active_by_organization_and_user(organization_id, by.user_id)
        assert member is not None
        await store.repositories.members.update_membership(member, {"status": "removed"})

    with pytest.raises(NotAuthorizedError):
        await service.resend(by, issued.invitation.id)


async def test_address_notes_ignores_whether_the_address_has_an_account(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    await store.actor(organization_id, role="member", email="ada@example.com")

    notes = await store.invitations().address_notes(by, ["ada@example.com"])

    assert notes == [AddressNote(domain="example.com", email="ada@example.com")]


async def test_no_note_for_an_address_at_an_own_proven_domain(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    async with store.uow:
        store._session.add(
            OrganizationDomain(
                organization_id=organization_id,
                domain="example.org",
                verification_token=uuid.uuid4().hex,
                verified_at=store.clock.now,
            )
        )
        await store._session.flush()

    notes = await store.invitations().address_notes(by, ["ada@example.org"])

    assert notes == []


async def test_no_note_for_a_public_email_provider(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)

    notes = await store.invitations().address_notes(by, ["ada@gmail.com"])

    assert notes == []


async def test_a_note_for_any_other_address(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)

    notes = await store.invitations().address_notes(by, ["Ada@Example.com"])

    assert notes == [AddressNote(domain="example.com", email="Ada@Example.com")]


async def test_a_note_for_an_own_domain_whose_proof_has_expired(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    async with store.uow:
        store._session.add(
            OrganizationDomain(
                organization_id=organization_id,
                domain="example.org",
                verification_token=uuid.uuid4().hex,
                verified_at=store.clock.now - DOMAIN_PROOF_TTL - timedelta(days=1),
            )
        )
        await store._session.flush()

    notes = await store.invitations().address_notes(by, ["ada@example.org"])

    assert notes == [AddressNote(domain="example.org", email="ada@example.org")]


async def test_the_note_is_the_same_whether_or_not_another_organization_proved_the_domain(store: _Store) -> None:
    organization_id = await store.organization()
    other_organization_id = await store.organization("Other")
    by = await store.actor(organization_id)
    async with store.uow:
        store._session.add(
            OrganizationDomain(
                organization_id=other_organization_id,
                domain="example.org",
                verification_token=uuid.uuid4().hex,
                verified_at=store.clock.now,
            )
        )
        await store._session.flush()

    notes = await store.invitations().address_notes(by, ["ada@example.org"])

    assert notes == [AddressNote(domain="example.org", email="ada@example.org")]


async def test_no_note_for_text_an_invitation_would_refuse(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)

    notes = await store.invitations().address_notes(by, ["not-an-address"])

    assert notes == []


async def test_address_notes_keeps_order_and_skips_addresses_needing_none(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    async with store.uow:
        store._session.add(
            OrganizationDomain(
                organization_id=organization_id,
                domain="example.org",
                verification_token=uuid.uuid4().hex,
                verified_at=store.clock.now,
            )
        )
        await store._session.flush()

    notes = await store.invitations().address_notes(
        by, ["ada@example.org", "grace@gmail.com", "mae@example.net", "not-an-address"]
    )

    assert notes == [AddressNote(domain="example.net", email="mae@example.net")]


async def test_address_notes_refuses_more_than_100_addresses(store: _Store) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    emails = [f"user{i}@example.com" for i in range(MAX_BULK_INVITATIONS + 1)]

    with pytest.raises(InvitationRequestTooLargeError):
        await store.invitations().address_notes(by, emails)


async def test_address_notes_refuses_a_member(store: _Store) -> None:
    organization_id = await store.organization()
    member = await store.actor(organization_id, role="member")

    with pytest.raises(NotAuthorizedError):
        await store.invitations().address_notes(member, ["ada@example.com"])


async def test_invite_logs_nothing_that_names_the_address_or_a_token(
    store: _Store, logs: pytest.LogCaptureFixture
) -> None:
    organization_id = await store.organization()
    by = await store.actor(organization_id)
    service = store.invitations(mailer=Mailer(_config(mail_transport="none")))

    issued = await service.invite(by, "ada@example.com", _offer())

    assert issued.accept_link is not None
    token = issued.accept_link.rsplit("token=", 1)[-1]
    text = "\n".join(record.getMessage() for record in logs.records)
    assert "ada@example.com" not in text
    assert token not in text


@pytest.mark.parametrize(
    "overrides",
    [
        {"expires_after": timedelta(0)},
        {"per_address_per_day": 0},
        {"per_day": 0},
        {"resend_limit": 0},
        {"resend_wait": timedelta(0)},
    ],
)
async def test_invitation_policy_refuses_a_non_positive_value(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _policy(**overrides)


async def test_invitation_policy_accepts_the_smallest_positive_values() -> None:
    policy = _policy(expires_after=timedelta(microseconds=1), per_address_per_day=1, per_day=1, resend_limit=1)
    assert policy.per_day == 1
