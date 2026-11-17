"""What an invitation link offers an invitee, and declining it."""

import uuid
from collections.abc import AsyncIterator, Collection
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.config import GatewayConfig
from gateway.core.tokens import hash_token
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import (
    InvitationAddressMismatchError,
    InvitationNoLongerValidError,
)
from gateway.models.organizations import WorkspaceAssignment
from gateway.models.tenancy import Organization, OrganizationDomain
from gateway.repositories.organizations import NewInvitation, OrganizationRepositories
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceRepository,
)
from gateway.services.mail import Mailer
from gateway.services.organizations import DeliveryRule, InvitationFactsReader, OrganizationClaimedDomains
from gateway.services.organizations._acceptance import (
    AcceptingPerson,
    InvitationAcceptance,
    _require_the_invited_address,
)

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self.now = _NOW

    def __call__(self) -> datetime:
        return self.now


class _NoBlockedAddresses:
    """A blocklist that matches nothing, for a test that is not about it."""

    async def blocked_among(self, emails: Collection[str]) -> set[str]:
        del emails
        return set()


class _EverythingBlocked:
    """A blocklist that matches every address it is asked about."""

    async def blocked_among(self, emails: Collection[str]) -> set[str]:
        return set(emails)


def _config(*, mail_transport: str = "smtp") -> GatewayConfig:
    """A deployment that can deliver links, unless given a transport that cannot."""
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
        self.repositories = OrganizationRepositories.on(self.uow)

    def acceptance(
        self,
        *,
        blocked: object | None = None,
        mail_transport: str = "smtp",
    ) -> InvitationAcceptance:
        domains = OrganizationClaimedDomains(OrganizationDomainRepository(self.uow))
        blocklist = blocked if blocked is not None else _NoBlockedAddresses()
        return InvitationAcceptance(
            self.uow,
            OrganizationRepositories.on(self.uow),
            blocked=blocklist,  # type: ignore[arg-type]
            clock=self.clock,
            delivery=DeliveryRule(domains, blocklist),  # type: ignore[arg-type]
            facts=InvitationFactsReader(
                domains=domains,
                organizations=OrganizationRepository(self.uow),
                users=UserRepository(self.uow),
                workspaces=WorkspaceRepository(self.uow),
            ),
            mailer=Mailer(_config(mail_transport=mail_transport)),
        )

    async def proven_domain_elsewhere(self, domain: str) -> None:
        """Another organization, holding a live proof of ``domain``."""
        async with self.uow:
            other = Organization(name="Other", slug=uuid.uuid4().hex)
            self._session.add(other)
            await self._session.flush()
            self._session.add(
                OrganizationDomain(
                    organization_id=other.id,
                    domain=domain,
                    verification_token="token",
                    verified_at=self.clock.now,
                )
            )
            await self._session.flush()

    async def account_for(self, email: str) -> None:
        """An account holding ``email``, so a test can prove an answer does not depend on one."""
        async with self.uow:
            await UserRepository(self.uow).create_local_identity(
                full_name="Holder", active_organization_id=uuid.uuid4(), email=email
            )

    async def organization(self, name: str = "Acme") -> Organization:
        async with self.uow:
            row = Organization(name=name, slug=uuid.uuid4().hex)
            self._session.add(row)
            await self._session.flush()
            return row

    async def invitation(
        self,
        organization: Organization,
        *,
        email: str = "ada@example.com",
        token: str = "plain-token",
        expires_in: timedelta = timedelta(days=7),
        workspaces: list[WorkspaceAssignment] | None = None,
    ) -> uuid.UUID:
        async with self.uow:
            row = await self.repositories.invitations.add(
                NewInvitation(
                    created_at=self.clock.now,
                    email=email,
                    expires_at=self.clock.now + expires_in,
                    invited_by_user_id=None,
                    next_send_at=None,
                    organization_id=organization.id,
                    role="member",
                    token_hash=hash_token(token),
                    workspace_assignments=workspaces or [],
                )
            )
            assert row is not None
            return row.id


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


async def test_preview_shows_the_organization_the_role_and_the_workspaces(store: _Store) -> None:
    organization = await store.organization(name="Acme")
    await store.invitation(organization)

    preview = await store.acceptance().preview("plain-token")

    assert preview.invited_email == "ada@example.com"
    assert preview.facts.organization_name == "Acme"
    assert preview.facts.role == "member"


async def test_an_expired_a_declined_and_an_unknown_link_answer_alike(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization, token="expired-token", expires_in=timedelta(days=-1))
    await store.invitation(organization, email="grace@example.com", token="declined-token")
    service = store.acceptance()
    await service.decline("declined-token")

    answers = []
    for token in ("expired-token", "declined-token", "no-such-token"):
        with pytest.raises(InvitationNoLongerValidError) as refusal:
            await service.preview(token)
        answers.append(str(refusal.value))

    assert len(set(answers)) == 1


async def test_preview_refuses_a_blocked_address(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization)

    with pytest.raises(InvitationNoLongerValidError):
        await store.acceptance(blocked=_EverythingBlocked()).preview("plain-token")


async def test_decline_ends_the_link(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization)
    service = store.acceptance()

    await service.decline("plain-token")

    with pytest.raises(InvitationNoLongerValidError):
        await service.preview("plain-token")


async def test_a_second_decline_answers_like_any_invalid_link(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization)
    service = store.acceptance()
    await service.decline("plain-token")

    with pytest.raises(InvitationNoLongerValidError):
        await service.decline("plain-token")


async def test_a_declined_invitation_is_still_there_for_the_admin(store: _Store) -> None:
    organization = await store.organization()
    invitation_id = await store.invitation(organization)
    service = store.acceptance()

    await service.decline("plain-token")

    async with store.uow:
        row = await store.repositories.invitations.get(invitation_id)
    assert row is not None
    assert row.declined_at == _NOW


async def test_the_proven_domain_rule_is_asked_only_where_links_cannot_be_delivered(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization, email="ada@proven.test")
    await store.proven_domain_elsewhere("proven.test")

    # Where Otari delivers the link, another organization's proof decides nothing.
    preview = await store.acceptance().preview("plain-token")
    assert preview.invited_email == "ada@proven.test"

    with pytest.raises(InvitationNoLongerValidError):
        await store.acceptance(mail_transport="console").preview("plain-token")


async def test_preview_answers_alike_whether_or_not_the_address_holds_an_account(store: _Store) -> None:
    organization = await store.organization(name="Acme")
    await store.invitation(organization, email="held@example.com", token="held-token")
    await store.invitation(organization, email="free@example.com", token="free-token")
    await store.account_for("held@example.com")
    service = store.acceptance()

    held = await service.preview("held-token")
    free = await service.preview("free-token")

    assert held.facts == free.facts


async def test_a_session_on_another_address_cannot_accept() -> None:
    with pytest.raises(InvitationAddressMismatchError, match="ada@example.com"):
        _require_the_invited_address(
            "ada@example.com", AcceptingPerson(email="grace@example.com", user_id=uuid.uuid4())
        )


async def test_the_invited_address_matches_in_any_spelling() -> None:
    _require_the_invited_address("ada@example.com", AcceptingPerson(email="  ADA@Example.com  ", user_id=uuid.uuid4()))


async def test_two_values_that_are_not_addresses_never_match() -> None:
    with pytest.raises(InvitationAddressMismatchError):
        _require_the_invited_address("not an address", AcceptingPerson(email="not an address", user_id=uuid.uuid4()))
