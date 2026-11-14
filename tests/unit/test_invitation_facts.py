"""The facts an invitee is told about an invitation, read on SQLite."""

import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.organizations import OrganizationInvitation, WorkspaceAssignment
from gateway.models.tenancy import (
    DOMAIN_PROOF_TTL,
    Organization,
    OrganizationDomain,
    OrganizationMemberRole,
    User,
    Workspace,
)
from gateway.repositories.organizations import DueInvitation
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceRepository,
)
from gateway.services.organizations import OrganizationClaimedDomains
from gateway.services.organizations._invitation_facts import InvitationFacts, InvitationFactsReader

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)


def _terms(
    organization_id: uuid.UUID,
    *,
    invited_by_user_id: uuid.UUID | None = None,
    role: OrganizationMemberRole = "member",
    workspace_assignments: Sequence[WorkspaceAssignment] = (),
) -> DueInvitation:
    return DueInvitation(
        attempt=1,
        declined_at=None,
        email="grace@invitee.test",
        expires_at=_NOW + timedelta(days=7),
        id=uuid.uuid4(),
        invited_by_user_id=invited_by_user_id,
        organization_id=organization_id,
        role=role,
        workspace_assignments=workspace_assignments,
    )


class _Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.uow = UnitOfWork(session)
        self.reader = InvitationFactsReader(
            domains=OrganizationClaimedDomains(OrganizationDomainRepository(self.uow)),
            organizations=OrganizationRepository(self.uow),
            users=UserRepository(self.uow),
            workspaces=WorkspaceRepository(self.uow),
        )

    async def organization(self, name: str = "Acme") -> uuid.UUID:
        row = Organization(name=name, slug=uuid.uuid4().hex)
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def user(
        self,
        *,
        full_name: str | None = "Ada",
        is_active: bool = True,
        verified_at: datetime | None = _NOW,
        vouched_at: datetime | None = None,
    ) -> uuid.UUID:
        row = User(
            email="ada@example.com",
            email_verified_at=verified_at,
            email_vouched_at=vouched_at,
            full_name=full_name,
            is_active=is_active,
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def workspace(self, organization_id: uuid.UUID, name: str) -> WorkspaceAssignment:
        row = Workspace(organization_id=organization_id, name=name)
        self._session.add(row)
        await self._session.flush()
        return {"workspace_id": str(row.id), "role": "member"}

    async def claim(
        self, organization_id: uuid.UUID, domain: str, *, verified_at: datetime | None, enabled: bool = True
    ) -> None:
        self._session.add(
            OrganizationDomain(
                organization_id=organization_id,
                domain=domain,
                verification_token=uuid.uuid4().hex,
                verified_at=verified_at,
                enabled=enabled,
            )
        )
        await self._session.flush()


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


async def test_the_facts_hold_every_value_of_the_invitation(store: _Store) -> None:
    async with store.uow:
        organization_id = await store.organization("Acme")
        inviter_id = await store.user(full_name="Ada")
        await store.claim(organization_id, "acme.com", verified_at=_NOW - timedelta(days=1))
        beta = await store.workspace(organization_id, "beta")
        alpha = await store.workspace(organization_id, "Alpha")

        facts = await store.reader.read(
            _terms(organization_id, invited_by_user_id=inviter_id, role="admin", workspace_assignments=[beta, alpha]),
            now=_NOW,
        )

    assert facts == InvitationFacts(
        inviter_email="ada@example.com",
        inviter_name="Ada",
        organization_name="Acme",
        proven_domains=("acme.com",),
        role="admin",
        workspace_names=("Alpha", "beta"),
    )


async def test_an_organization_that_no_longer_exists_has_no_facts(store: _Store) -> None:
    async with store.uow:
        facts = await store.reader.read(_terms(uuid.uuid4()), now=_NOW)

    assert facts is None


async def test_an_invitation_with_no_inviter_names_nobody(store: _Store) -> None:
    async with store.uow:
        organization_id = await store.organization()

        facts = await store.reader.read(_terms(organization_id), now=_NOW)

    assert facts is not None
    assert (facts.inviter_name, facts.inviter_email) == (None, None)
    assert (facts.proven_domains, facts.workspace_names) == ((), ())


async def test_an_inviter_who_no_longer_exists_names_nobody(store: _Store) -> None:
    async with store.uow:
        organization_id = await store.organization()

        facts = await store.reader.read(_terms(organization_id, invited_by_user_id=uuid.uuid4()), now=_NOW)

    assert facts is not None
    assert (facts.inviter_name, facts.inviter_email) == (None, None)


@pytest.mark.parametrize(
    ("verified_at", "vouched_at"),
    [(None, None), (_NOW, _NOW)],
    ids=["never verified", "verified by an admin's word"],
)
async def test_an_inviter_address_that_the_inviter_has_not_verified_is_absent(
    store: _Store, verified_at: datetime | None, vouched_at: datetime | None
) -> None:
    async with store.uow:
        organization_id = await store.organization()
        inviter_id = await store.user(full_name="Ada", verified_at=verified_at, vouched_at=vouched_at)

        facts = await store.reader.read(_terms(organization_id, invited_by_user_id=inviter_id), now=_NOW)

    assert facts is not None
    assert (facts.inviter_name, facts.inviter_email) == ("Ada", None)


async def test_the_verified_address_of_a_deactivated_inviter_is_absent(store: _Store) -> None:
    async with store.uow:
        organization_id = await store.organization()
        inviter_id = await store.user(is_active=False)

        facts = await store.reader.read(_terms(organization_id, invited_by_user_id=inviter_id), now=_NOW)

    assert facts is not None
    assert (facts.inviter_name, facts.inviter_email) == ("Ada", None)


async def test_only_a_domain_with_a_proof_that_has_not_expired_is_proven(store: _Store) -> None:
    async with store.uow:
        organization_id = await store.organization()
        await store.claim(
            organization_id, "expired.example", verified_at=_NOW - DOMAIN_PROOF_TTL - timedelta(microseconds=1)
        )
        await store.claim(organization_id, "last-day.example", verified_at=_NOW - DOMAIN_PROOF_TTL)
        await store.claim(organization_id, "switched-off.example", verified_at=_NOW, enabled=False)
        await store.claim(organization_id, "unproven.example", verified_at=None)
        await store.claim(await store.organization("Other"), "other.example", verified_at=_NOW)

        facts = await store.reader.read(_terms(organization_id), now=_NOW)

    assert facts is not None
    assert facts.proven_domains == ("last-day.example", "switched-off.example")


async def test_a_workspace_that_is_not_in_the_organization_is_left_out(store: _Store) -> None:
    async with store.uow:
        organization_id = await store.organization()
        kept = await store.workspace(organization_id, "Kept")
        elsewhere = await store.workspace(await store.organization("Other"), "Elsewhere")
        deleted: WorkspaceAssignment = {"workspace_id": str(uuid.uuid4()), "role": "member"}

        facts = await store.reader.read(
            _terms(organization_id, workspace_assignments=[deleted, elsewhere, kept]), now=_NOW
        )

    assert facts is not None
    assert facts.workspace_names == ("Kept",)


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param({"workspace_id": "w1", "role": "member"}, id="text that is not a UUID"),
        pytest.param({"workspace_id": 7, "role": "member"}, id="a number"),
        pytest.param({"workspace_id": None, "role": "member"}, id="no value"),
        pytest.param({"role": "member"}, id="no ID at all"),
    ],
)
async def test_a_workspace_whose_stored_id_is_not_a_uuid_is_left_out(store: _Store, stored: dict[str, object]) -> None:
    async with store.uow:
        organization_id = await store.organization()
        kept = await store.workspace(organization_id, "Kept")
        unreadable = cast("WorkspaceAssignment", stored)

        facts = await store.reader.read(_terms(organization_id, workspace_assignments=[unreadable, kept]), now=_NOW)

    assert facts is not None
    assert facts.workspace_names == ("Kept",)


async def test_the_facts_of_a_stored_invitation_are_read_from_its_row(store: _Store) -> None:
    async with store.uow:
        organization_id = await store.organization("Acme")
        alpha = await store.workspace(organization_id, "Alpha")
        row = OrganizationInvitation(
            email="grace@invitee.test",
            expires_at=_NOW + timedelta(days=7),
            invited_by_user_id=None,
            organization_id=organization_id,
            role="admin",
            token_hash="hash",
            workspace_assignments=[alpha],
        )

        facts = await store.reader.read(row, now=_NOW)

    assert facts is not None
    assert (facts.organization_name, facts.role, facts.workspace_names) == ("Acme", "admin", ("Alpha",))
