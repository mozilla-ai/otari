"""The races the tenancy services resolve, driven concurrently.

Every check-then-act path here is decided by a unique constraint rather than by
its own pre-check, and the pre-check only makes the common case a clean 409. The
fixes for that are one ``except IntegrityError`` each, which is the kind of line
a later refactor removes without a test failing, so these drive the real race
with separate sessions rather than asserting the branch in isolation.
"""

import asyncio
import uuid
from collections.abc import Callable

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from gateway.models.tenancy import (
    ActiveOrganizationMemberCreateRequest,
    Organization,
    User,
    WorkspaceCreate,
)
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy import OrganizationService, WorkspaceService
from gateway.services.tenancy.errors import (
    ForeignTenancyError,
    OrganizationMemberAlreadyExistsError,
    WorkspaceAlreadyExistsError,
    WorkspaceMemberAlreadyExistsError,
)
from gateway.services.tenancy.provisioning_service import (
    BOOTSTRAP_IDENTITY_KEY,
    ensure_bootstrap_identity,
)

pytestmark = pytest.mark.asyncio

_RACERS = 4


async def _seed_owner(db: AsyncSession) -> tuple[Organization, User]:
    organization = await OrganizationRepository(db).create_organization(
        name="Acme",
        slug=f"acme-{uuid.uuid4().hex[:8]}",
        created_by_user_id=None,
    )
    owner = await UserRepository(db).create_local_identity(
        full_name="Owner",
        active_organization_id=organization.id,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=owner.id,
        role="owner",
    )
    await db.commit()
    return organization, owner


async def _race(
    session_factory: async_sessionmaker[AsyncSession],
    attempt: Callable[[AsyncSession], object],
) -> list[object]:
    """Run one attempt per racer, each on its own session, and collect outcomes."""

    async def run_one() -> object:
        async with session_factory() as session:
            try:
                return await attempt(session)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    return list(await asyncio.gather(*(run_one() for _ in range(_RACERS))))


@pytest.fixture
def sessions(postgres_url: str) -> async_sessionmaker[AsyncSession]:
    url = postgres_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://").replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    return async_sessionmaker(create_async_engine(url), expire_on_commit=False)


async def test_concurrent_workspace_creates_conflict_rather_than_fail(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    organization, owner = await _seed_owner(async_db)

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await WorkspaceService(session).create_workspace(
            user=user,
            workspace_create=WorkspaceCreate(name="Research"),
        )

    outcomes = await _race(sessions, attempt)

    created = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, WorkspaceAlreadyExistsError)]
    assert len(created) == 1
    assert len(conflicts) == _RACERS - 1
    _, count = await WorkspaceRepository(async_db).get_by_organization(organization.id, limit=1)
    assert count == 1


async def test_concurrent_member_adds_create_one_identity(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The email unique index decides, and the losers report the conflict."""
    _, owner = await _seed_owner(async_db)

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await OrganizationService(session).create_active_organization_member_for_user(
            user=user,
            request=ActiveOrganizationMemberCreateRequest(email="ada@example.com"),
        )

    outcomes = await _race(sessions, attempt)

    added = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, OrganizationMemberAlreadyExistsError)]
    assert len(added) == 1
    assert len(conflicts) == _RACERS - 1
    assert await UserRepository(async_db).get_by_email("ada@example.com") is not None


async def test_concurrent_workspace_member_adds_conflict(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    _, owner = await _seed_owner(async_db)
    service = OrganizationService(async_db)
    owner_row = await UserRepository(async_db).get(owner.id)
    assert owner_row is not None
    added = await service.create_active_organization_member_for_user(
        user=owner_row,
        request=ActiveOrganizationMemberCreateRequest(email="ada@example.com"),
    )
    workspace = await WorkspaceService(async_db).create_workspace(
        user=owner_row,
        workspace_create=WorkspaceCreate(name="Research"),
    )
    assert added.user_id is not None

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await WorkspaceService(session).add_member(
            user=user,
            workspace_id=workspace.id,
            user_id=added.user_id,  # type: ignore[arg-type]
        )

    outcomes = await _race(sessions, attempt)

    joined = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, WorkspaceMemberAlreadyExistsError)]
    assert len(joined) == 1
    assert len(conflicts) == _RACERS - 1


async def test_provisioning_refuses_to_shadow_an_organization_it_did_not_create(
    test_db: Session,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """A restored or imported tenancy must not be silently made unreachable.

    Provisioning adopts an organization slugged ``default``, which is the one it
    would have made itself. Anything else it would ignore, create its own beside,
    and point the marker at that; every route is scoped to the marked identity's
    organization, so the restored rows become invisible with no list, no switch
    and no by-id route to find them. The platform slugs organizations
    ``{name}-{prefix}``, so a restored hosted organization always lands here.
    """
    async with sessions() as db:
        await _seed_owner(db)

        with pytest.raises(ForeignTenancyError) as raised:
            await ensure_bootstrap_identity(db)

        # The message has to name the organization and the way out, because the
        # marker is not a settable key and nothing else can repoint it.
        assert "Acme" in str(raised.value)
        assert BOOTSTRAP_IDENTITY_KEY in str(raised.value)


async def test_provisioning_still_runs_on_an_empty_database(
    test_db: Session,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The guard must not break first boot, which is the ordinary path."""
    async with sessions() as db:
        operator = await ensure_bootstrap_identity(db)

        assert operator.full_name == "Operator"

