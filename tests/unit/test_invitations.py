"""Creating and listing organization invitations, with a set clock and an in-memory rate limit store, on SQLite."""

import dataclasses
import logging
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
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
    InvitationRequestTooLargeError,
    InvitationsNeedSsoError,
    MembershipUpdateError,
    NotAuthorizedError,
    OrganizationMemberAlreadyExistsError,
    WorkspaceNotFoundError,
)
from gateway.log_config import logger as gateway_logger
from gateway.models.organizations import OrganizationInvitation
from gateway.models.tenancy import (
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
from gateway.repositories.tenancy import OrganizationDomainRepository, UserRepository, WorkspaceRepository
from gateway.services.mail import Mailer
from gateway.services.organizations import OrganizationClaimedDomains
from gateway.services.organizations._invitation_limits import Admission, InvitationLimits
from gateway.services.organizations._invitations import (
    ActingMember,
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

    async def actor(self, organization_id: uuid.UUID, *, role: str = "owner", email: str | None = None) -> ActingMember:
        async with self.uow:
            user = await UserRepository(self.uow).create_local_identity(
                full_name="Actor", active_organization_id=organization_id, email=email
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
