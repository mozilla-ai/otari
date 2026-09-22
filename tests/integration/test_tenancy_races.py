"""The races the tenancy services resolve, driven concurrently.

Every check-then-act path here is decided by a unique constraint rather than by
its own pre-check, and the pre-check only makes the common case a clean 409. The
fixes for that are one ``except IntegrityError`` each, which is the kind of line
a later refactor removes without a test failing, so these drive the real race
with separate sessions rather than asserting the branch in isolation.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import pytest
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.adapters.api_key_format_adapter import DefaultApiKeyFormatAdapter
from gateway.api.routes.scoped_budgets import create_scoped_budget
from gateway.auth.models import hash_key
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.budget_exceptions import OrganizationScopeNotFoundError
from gateway.models.api_keys import APIKey
from gateway.models.budgets import SCOPE_WORKSPACE, SCOPE_WORKSPACE_MEMBER, ScopedBudget, ScopeType
from gateway.models.tenancy import (
    ActiveOrganizationMemberCreateRequest,
    ActiveOrganizationMemberUpdateRequest,  # noqa: E402
    InviteOrganizationMemberRequest,
    Organization,
    User,
    WorkspaceActivationState,
    WorkspaceAssignmentRequest,
    WorkspaceCreate,
    WorkspaceMember,
    WorkspacePublic,
)
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.repositories.budgets import BudgetRepositories
from gateway.repositories.tenancy import (
    InvitationRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.schemas.budgets import (
    CreateScopedBudgetRequest,
    OrganizationBudgetCreate,
    OrganizationScopedBudgetCreate,
    OrganizationScopedBudgetPublic,
    WorkspaceMemberBudgetPolicyCreate,
)
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets import BudgetService, WorkspaceBudgetDefaultService
from gateway.services.password_service import verify_password_async
from gateway.services.tenancy import OrganizationService, WorkspaceService, user_service
from gateway.services.tenancy.errors import (
    EmailAlreadyInUseError,
    ForeignTenancyError,
    InvitationAlreadyPendingError,
    InvitationAlreadyUsedError,
    LastWorkspaceError,
    MembershipUpdateError,
    NotAuthorizedError,
    OrganizationMemberAlreadyExistsError,
    ResetTokenInvalidError,
    VerificationTokenInvalidError,
    WorkspaceAlreadyExistsError,
    WorkspaceMemberAlreadyExistsError,
)
from gateway.services.tenancy.provisioning_service import (
    BOOTSTRAP_IDENTITY_KEY,
    ensure_bootstrap_identity,
)
from gateway.services.tenancy.tokens import generate_token, hash_token
from gateway.services.tenancy.user_service import set_password
from gateway.services.tenancy.workspace_activation_service import (
    ACTIVATION_KEY_NAME,
    WorkspaceActivationService,
)

from .tenancy_helpers import create_budget, create_member

pytestmark = pytest.mark.asyncio

# The open-source format, which is what a service built outside a request would get.
KEY_FORMAT = DefaultApiKeyFormatAdapter(None)

_RACERS = 4

# How long the delete waits for the writer racing it, once that writer has reached its checkpoint.
# Paid in full on every passing run, because the writer blocks on the workspace lock until the delete commits.
# It only has to cover the writer's insert and commit when nothing holds the lock, which is fast.
_WRITER_WINDOW = 0.25

# Bounds a writer that never reaches its checkpoint. A healthy run never waits this long.
_CHECKPOINT_TIMEOUT = 5.0


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
        service = WorkspaceService(session, membership_listener=WorkspaceBudgetDefaultService(session))
        return await service.create_workspace(
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


async def test_concurrent_claims_of_one_address_leave_exactly_one_holder(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two identities claiming the same sign-in address at once.

    ``_claimable_email`` is a preflight, not a lock, so both racers can pass it
    and the unique index on ``user.email`` is what actually decides. The loser
    has to report the conflict its preflight would have reported rather than
    surfacing the driver's integrity error as a 500. Postgres is the engine that
    matters here: the unit suite runs on SQLite, whose error text differs, and
    the detector has to recognize both.
    """
    organization, _ = await _seed_owner(async_db)
    # Count how many losers came through the integrity-error mapping. Without
    # this the test passes whether the racers overlapped or not: a serialized
    # run refuses them at the ``_claimable_email`` preflight, which raises the
    # very same ``EmailAlreadyInUseError``, so the outcome assertions below
    # cannot tell the two routes apart and would green-light a fix that never
    # runs.
    mapped: list[bool] = []
    real_detector = user_service._is_email_conflict

    def counting_detector(exc: IntegrityError) -> bool:
        verdict = real_detector(exc)
        mapped.append(verdict)
        return verdict

    monkeypatch.setattr(user_service, "_is_email_conflict", counting_detector)

    users = UserRepository(async_db)
    racer_ids = [
        (await users.create_local_identity(full_name=f"Claimer {index}", active_organization_id=organization.id)).id
        for index in range(_RACERS)
    ]
    await async_db.commit()
    # One identity per racer, handed out in order. `next` on a plain iterator is
    # safe here because the racers only interleave at their awaits.
    hand_out = iter(racer_ids)

    async def attempt(session: AsyncSession) -> object:
        # Each racer claims *its own* identity, all of them naming one address.
        identity = await UserRepository(session).get(next(hand_out))
        assert identity is not None
        await set_password(
            session,
            identity,
            new_password="a-real-password",
            email="contested@example.com",
        )
        # `set_password` returns None, so the racer that got through reports a
        # marker rather than a value indistinguishable from "nothing happened".
        return "claimed"

    outcomes = await _race(sessions, attempt)

    claimed = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    refused = [outcome for outcome in outcomes if isinstance(outcome, EmailAlreadyInUseError)]
    escaped = [o for o in outcomes if isinstance(o, Exception) and not isinstance(o, EmailAlreadyInUseError)]
    assert not escaped, f"a raw database error reached the caller: {escaped}"
    assert len(claimed) == 1, outcomes
    assert len(refused) == _RACERS - 1, outcomes
    # Every loser reached the unique index rather than the preflight, which is
    # what makes this a test of the mapping and not of the pre-check.
    assert mapped == [True] * (_RACERS - 1), mapped

    holders = (await async_db.execute(select(User).where(col(User.email) == "contested@example.com"))).scalars().all()
    assert len(holders) == 1


async def test_concurrent_member_adds_create_one_identity(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The email unique index decides, and the losers report the conflict."""
    _, owner = await _seed_owner(async_db)

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await OrganizationService(
            session, membership_listener=WorkspaceBudgetDefaultService(session)
        ).create_active_organization_member_for_user(
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
    service = OrganizationService(async_db, membership_listener=WorkspaceBudgetDefaultService(async_db))
    owner_row = await UserRepository(async_db).get(owner.id)
    assert owner_row is not None
    added = await service.create_active_organization_member_for_user(
        user=owner_row,
        request=ActiveOrganizationMemberCreateRequest(email="ada@example.com"),
    )
    workspace_service = WorkspaceService(async_db, membership_listener=WorkspaceBudgetDefaultService(async_db))
    workspace = await workspace_service.create_workspace(
        user=owner_row,
        workspace_create=WorkspaceCreate(name="Research"),
    )
    assert added.user_id is not None

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await WorkspaceService(session, membership_listener=WorkspaceBudgetDefaultService(session)).add_member(
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
            await ensure_bootstrap_identity(db, membership_listener=WorkspaceBudgetDefaultService(db))

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
        operator = await ensure_bootstrap_identity(db, membership_listener=WorkspaceBudgetDefaultService(db))

        assert operator.full_name == "Operator"


async def test_concurrent_demotions_cannot_strip_the_last_owner(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Two owners, each demoted by a different concurrent request.

    Unlike the three races above there is no unique index to lose to, so the
    ``IntegrityError`` guards never fire and only the row lock in
    ``OrganizationRepository.lock`` keeps the count the guard read still true when
    it writes. Without it both requests count two owners and both commit.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme",
        slug=f"acme-{uuid.uuid4().hex[:8]}",
        created_by_user_id=None,
    )
    members = OrganizationMemberRepository(async_db)
    users = UserRepository(async_db)
    owners = []
    for index in range(2):
        owner = await users.create_local_identity(
            full_name=f"Owner {index}",
            active_organization_id=organization.id,
            email=f"owner{index}@example.com",
        )
        membership = await members.create_membership(
            organization_id=organization.id,
            user_id=owner.id,
            role="owner",
        )
        owners.append((owner.id, membership.id))
    await async_db.commit()

    def demote(actor_id: uuid.UUID, target_membership_id: uuid.UUID) -> Callable[[AsyncSession], object]:
        async def attempt(session: AsyncSession) -> object:
            actor = await UserRepository(session).get(actor_id)
            assert actor is not None
            service = OrganizationService(session, membership_listener=WorkspaceBudgetDefaultService(session))
            return await service.update_active_organization_member_for_user(
                user=actor,
                organization_member_id=target_membership_id,
                request=ActiveOrganizationMemberUpdateRequest(role="member"),
            )

        return attempt

    attempts = [
        demote(owners[0][0], owners[1][1]),
        demote(owners[1][0], owners[0][1]),
    ]

    async def run_one(attempt: Callable[[AsyncSession], object]) -> object:
        async with sessions() as session:
            try:
                return await attempt(session)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    outcomes = list(await asyncio.gather(*(run_one(attempt) for attempt in attempts)))

    # Which refusal the loser gets depends on where it was when the winner
    # committed: still inside its own pre-lock reads, so its actor is an owner
    # and the last-owner guard turns it away, or not yet started, so its actor
    # is already a member and the management-role check turns it away first.
    # Both are the same invariant holding, which is what the owner count asserts.
    refused = [outcome for outcome in outcomes if isinstance(outcome, MembershipUpdateError | NotAuthorizedError)]
    assert len(refused) == 1
    assert await OrganizationMemberRepository(async_db).count_active_owners(organization.id) == 1


async def test_concurrent_deletes_cannot_remove_the_last_workspace(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Two workspaces, deleted concurrently. One has to survive."""
    organization, owner = await _seed_owner(async_db)
    service = WorkspaceService(async_db, membership_listener=WorkspaceBudgetDefaultService(async_db))
    workspace_ids = [
        (await service.create_workspace(user=owner, workspace_create=WorkspaceCreate(name=name))).id
        for name in ("One", "Two")
    ]

    async def run_one(workspace_id: uuid.UUID) -> object:
        async with sessions() as session:
            user = await UserRepository(session).get(owner.id)
            assert user is not None
            try:
                service = WorkspaceService(session, membership_listener=WorkspaceBudgetDefaultService(session))
                await service.delete_workspace(user=user, workspace_id=workspace_id)
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc
            return None

    outcomes = list(await asyncio.gather(*(run_one(workspace_id) for workspace_id in workspace_ids)))

    refused = [outcome for outcome in outcomes if isinstance(outcome, LastWorkspaceError)]
    assert len(refused) == 1
    _, remaining = await WorkspaceRepository(async_db).get_by_organization(organization.id, limit=1)
    assert remaining == 1


class _PausingListener:
    """Delegates to the real listener, then holds the delete open until the racing writer settles.

    The orphan only appears when that writer commits between the delete's ceiling
    sweep and the row delete, so that interleaving is pinned rather than raced for.
    """

    def __init__(self, inner: WorkspaceBudgetDefaultService, let_the_writer_run: Callable[[], Awaitable[None]]) -> None:
        self._inner = inner
        self._let_the_writer_run = let_the_writer_run

    async def member_joined(self, member: WorkspaceMember) -> None:
        await self._inner.member_joined(member)

    async def member_removed(self, member: WorkspaceMember) -> None:
        await self._inner.member_removed(member)

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        await self._inner.workspace_deleted(workspace_id, member_ids)
        await self._let_the_writer_run()


async def test_a_join_during_a_workspace_delete_leaves_no_orphaned_ceiling(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ceiling must not outlive the membership it caps.

    ``scoped_budgets.scope_id`` is not a foreign key, so nothing cascades an
    orphan away, it holds a RESTRICT reference that refuses its budget's
    deletion, and no page lists it.
    """
    organization, owner = await _seed_owner(async_db)
    service = WorkspaceService(async_db, membership_listener=WorkspaceBudgetDefaultService(async_db))
    target = await service.create_workspace(user=owner, workspace_create=WorkspaceCreate(name="Target"))
    await service.create_workspace(user=owner, workspace_create=WorkspaceCreate(name="Survivor"))
    joiner = await create_member(async_db, organization, role="member", full_name="Joiner")
    await WorkspaceBudgetDefaultService(async_db).create_default(
        user=owner,
        workspace_id=target.id,
        request=WorkspaceMemberBudgetPolicyCreate(budget_id=await create_budget(async_db, max_budget=10.0)),
    )
    await async_db.commit()

    # A positive control: the final assertion is a negative, and would pass on nothing at all.
    materialized = (
        (await async_db.execute(select(ScopedBudget).where(ScopedBudget.scope_type == "workspace_member")))
        .scalars()
        .all()
    )
    assert len(materialized) == 1, "the default must have given the owner a ceiling before the delete"

    swept = asyncio.Event()
    at_lock = asyncio.Event()

    async def join() -> object:
        await swept.wait()
        async with sessions() as session:
            actor = await UserRepository(session).get(owner.id)
            assert actor is not None
            adder = WorkspaceService(session, membership_listener=WorkspaceBudgetDefaultService(session))
            take_lock = adder.workspaces.lock

            async def signal_then_lock(workspace_id: uuid.UUID) -> None:
                at_lock.set()
                await take_lock(workspace_id)

            monkeypatch.setattr(adder.workspaces, "lock", signal_then_lock)
            try:
                return await adder.add_member(user=actor, workspace_id=target.id, user_id=joiner.id)
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    joining = asyncio.create_task(join())

    async def let_the_join_run() -> None:
        swept.set()
        # The window opens only once the join is at the lock, so its setup time cannot use it up.
        await asyncio.wait_for(at_lock.wait(), timeout=_CHECKPOINT_TIMEOUT)
        await asyncio.wait({joining}, timeout=_WRITER_WINDOW)

    async with sessions() as session:
        actor = await UserRepository(session).get(owner.id)
        assert actor is not None
        deleter = WorkspaceService(
            session,
            membership_listener=_PausingListener(WorkspaceBudgetDefaultService(session), let_the_join_run),
        )
        await deleter.delete_workspace(user=actor, workspace_id=target.id)
    await joining

    ceilings = (
        (await async_db.execute(select(ScopedBudget).where(ScopedBudget.scope_type == "workspace_member")))
        .scalars()
        .all()
    )
    memberships = {
        str(member_id) for member_id in (await async_db.execute(select(col(WorkspaceMember.id)))).scalars().all()
    }
    assert [ceiling.scope_id for ceiling in ceilings if ceiling.scope_id not in memberships] == []


def _budget_service(db: AsyncSession, organizations: OrganizationService | None = None) -> BudgetService:
    uow = UnitOfWork(db)
    return BudgetService(
        uow,
        BudgetRepositories.on(uow),
        organizations or OrganizationService(db, membership_listener=None),
        ApiKeyService(ApiKeyRepository(uow)),
    )


class _RaceOutcome(NamedTuple):
    """What the producer returned, and whether it was still running when the delete's window closed."""

    produced: object
    contended: bool


async def _race_a_workspace_delete(
    *,
    owner: User,
    workspace_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
    produce: Callable[[AsyncSession, asyncio.Event], Awaitable[object]],
) -> _RaceOutcome:
    """Delete the workspace while ``produce`` runs on a session of its own.

    The delete pauses after its ceiling sweep, which is the window an orphan needs.
    ``produce`` sets the event it is passed immediately before the call under test, so the
    pause covers that call alone and not the setup around it.
    ``contended`` is false when the producer finished inside the pause, which is what a
    producer that takes no lock does.
    """
    swept = asyncio.Event()
    at_checkpoint = asyncio.Event()
    finished_early = False

    async def race() -> object:
        await swept.wait()
        async with sessions() as session:
            # The first statement pays the connection cost, which must not come out of the window.
            await session.execute(select(1))
            try:
                return await produce(session, at_checkpoint)
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    racing = asyncio.create_task(race())

    async def let_the_producer_run() -> None:
        nonlocal finished_early
        swept.set()
        await asyncio.wait_for(at_checkpoint.wait(), timeout=_CHECKPOINT_TIMEOUT)
        done, _ = await asyncio.wait({racing}, timeout=_WRITER_WINDOW)
        finished_early = bool(done)

    try:
        async with sessions() as session:
            actor = await UserRepository(session).get(owner.id)
            assert actor is not None
            deleter = WorkspaceService(
                session,
                membership_listener=_PausingListener(WorkspaceBudgetDefaultService(session), let_the_producer_run),
            )
            await deleter.delete_workspace(user=actor, workspace_id=workspace_id)
    finally:
        # Awaited even when the delete raised, so a real failure is not buried under a
        # task whose exception was never retrieved.
        produced = await racing
    return _RaceOutcome(produced=produced, contended=not finished_early)


async def _seed_a_workspace_to_delete(db: AsyncSession) -> tuple[User, WorkspacePublic, WorkspaceMember]:
    """An owner, the workspace a test deletes, a second one so the delete is allowed, and the owner's membership."""
    _, owner = await _seed_owner(db)
    workspaces = WorkspaceService(db, membership_listener=WorkspaceBudgetDefaultService(db))
    target = await workspaces.create_workspace(user=owner, workspace_create=WorkspaceCreate(name="Target"))
    await workspaces.create_workspace(user=owner, workspace_create=WorkspaceCreate(name="Survivor"))
    membership = await WorkspaceMemberRepository(db).get_by_workspace_and_user(target.id, owner.id)
    assert membership is not None
    return owner, target, membership


@pytest.mark.parametrize("scope_type", [SCOPE_WORKSPACE, SCOPE_WORKSPACE_MEMBER])
async def test_an_organization_ceiling_created_during_a_workspace_delete_leaves_no_orphan(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    scope_type: ScopeType,
) -> None:
    """A ceiling must not outlive the workspace or membership it caps.

    Creating one checks the scope exists and then inserts, so a deletion committing
    between the two leaves a ceiling that nothing sweeps and no page lists.
    """
    owner, target, membership = await _seed_a_workspace_to_delete(async_db)
    budget = await _budget_service(async_db).create_organization_budget(
        user=owner,
        request=OrganizationBudgetCreate(name="Cap", max_budget=10.0),
    )
    await async_db.commit()

    scope_id = str(target.id if scope_type == SCOPE_WORKSPACE else membership.id)

    async def create(session: AsyncSession, ready: asyncio.Event) -> object:
        actor = await UserRepository(session).get(owner.id)
        assert actor is not None
        creator = _budget_service(session)
        ready.set()
        return await creator.create_organization_ceiling(
            user=actor,
            request=OrganizationScopedBudgetCreate(
                scope_type=scope_type,
                scope_id=scope_id,
                budget_id=budget.budget_id,
            ),
        )

    race = await _race_a_workspace_delete(
        owner=owner,
        workspace_id=target.id,
        sessions=sessions,
        produce=create,
    )

    gone = {str(target.id), str(membership.id)}
    ceilings = (await async_db.execute(select(ScopedBudget))).scalars().all()
    assert [ceiling.scope_id for ceiling in ceilings if ceiling.scope_id in gone] == []
    # The negative above would pass on a create that never contended. These say it did.
    assert race.contended
    assert isinstance(race.produced, OrganizationScopeNotFoundError)


@pytest.mark.parametrize("scope_type", [SCOPE_WORKSPACE, SCOPE_WORKSPACE_MEMBER])
async def test_a_deployment_ceiling_created_during_a_workspace_delete_leaves_no_orphan(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    scope_type: ScopeType,
) -> None:
    """A ceiling created through the operator's route must not outlive its scope either.

    The route checks the scope exists and then inserts on a session of its own, so a
    deletion committing between the two leaves a ceiling that nothing sweeps.
    """
    owner, target, membership = await _seed_a_workspace_to_delete(async_db)
    budget_id = await create_budget(async_db, max_budget=10.0)
    await async_db.commit()

    scope_id = str(target.id if scope_type == SCOPE_WORKSPACE else membership.id)

    async def create(session: AsyncSession, ready: asyncio.Event) -> object:
        ready.set()
        return await create_scoped_budget(
            CreateScopedBudgetRequest(scope_type=scope_type, scope_id=scope_id, budget_id=budget_id),
            session,
            OrganizationService(session, membership_listener=None),
        )

    race = await _race_a_workspace_delete(
        owner=owner,
        workspace_id=target.id,
        sessions=sessions,
        produce=create,
    )

    gone = {str(target.id), str(membership.id)}
    ceilings = (await async_db.execute(select(ScopedBudget))).scalars().all()
    assert [ceiling.scope_id for ceiling in ceilings if ceiling.scope_id in gone] == []
    # The negative above would pass on a create that never contended. These say it did.
    assert race.contended
    assert isinstance(race.produced, HTTPException)
    assert race.produced.status_code == status.HTTP_404_NOT_FOUND


async def test_a_workspace_delete_sweeps_a_ceiling_that_won_the_lock(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other order the lock allows: the create wins it, and the delete sweeps what it left.

    That only holds while the delete takes the lock before it reads the memberships it
    sweeps. Reverse those two and this is the ordering that strands a ceiling.
    """
    owner, target, _ = await _seed_a_workspace_to_delete(async_db)
    budget = await _budget_service(async_db).create_organization_budget(
        user=owner,
        request=OrganizationBudgetCreate(name="Cap", max_budget=10.0),
    )
    await async_db.commit()

    holding = asyncio.Event()
    may_finish = asyncio.Event()

    async def create() -> object:
        async with sessions() as session:
            actor = await UserRepository(session).get(owner.id)
            assert actor is not None
            organizations = OrganizationService(session, membership_listener=None)
            creator = _budget_service(session, organizations)
            take_lock = organizations.lock_workspace

            async def hold_then_continue(workspace_id: uuid.UUID) -> None:
                await take_lock(workspace_id)
                holding.set()
                await may_finish.wait()

            monkeypatch.setattr(organizations, "lock_workspace", hold_then_continue)
            return await creator.create_organization_ceiling(
                user=actor,
                request=OrganizationScopedBudgetCreate(
                    scope_type=SCOPE_WORKSPACE,
                    scope_id=str(target.id),
                    budget_id=budget.budget_id,
                ),
            )

    creating = asyncio.create_task(create())
    await asyncio.wait_for(holding.wait(), timeout=_CHECKPOINT_TIMEOUT)

    async def delete() -> None:
        async with sessions() as session:
            actor = await UserRepository(session).get(owner.id)
            assert actor is not None
            deleter = WorkspaceService(session, membership_listener=WorkspaceBudgetDefaultService(session))
            await deleter.delete_workspace(user=actor, workspace_id=target.id)

    deleting = asyncio.create_task(delete())
    try:
        blocked, _ = await asyncio.wait({deleting}, timeout=_WRITER_WINDOW)
        assert not blocked, "the delete must wait on the lock the create holds"
        may_finish.set()
        created = await creating
        await deleting
    finally:
        may_finish.set()

    assert isinstance(created, OrganizationScopedBudgetPublic)
    ceilings = (await async_db.execute(select(ScopedBudget))).scalars().all()
    assert [ceiling.scope_id for ceiling in ceilings if ceiling.scope_id == str(target.id)] == []


async def test_concurrent_invites_to_a_suspended_membership_produce_one_pending_invitation(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """A suspended membership has no unique index to lose to either.

    `organization_member_id` on `Invitation` carries no uniqueness (a
    membership can be invited, revoked, and re-invited more than once over its
    life), so nothing catches two concurrent invites to the same suspended
    membership as an `IntegrityError`. Without locking the organization before
    the status check that decides create/revive/refuse, both racers can read
    "suspended", both revive it, and both mint their own live pending
    invitation for the one membership.
    """
    organization, owner = await _seed_owner(async_db)
    owner_row = await UserRepository(async_db).get(owner.id)
    assert owner_row is not None
    added = await OrganizationService(
        async_db, membership_listener=WorkspaceBudgetDefaultService(async_db)
    ).create_active_organization_member_for_user(
        user=owner_row,
        request=ActiveOrganizationMemberCreateRequest(email="grace@example.com"),
    )
    assert added.organization_member_id is not None
    assert added.user_id is not None
    await OrganizationService(
        async_db, membership_listener=WorkspaceBudgetDefaultService(async_db)
    ).remove_active_organization_member_for_user(
        user=owner_row,
        organization_member_id=added.organization_member_id,
    )
    config = GatewayConfig()

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await OrganizationService(
            session, membership_listener=WorkspaceBudgetDefaultService(session)
        ).invite_active_organization_member_for_user(
            user=user,
            request=InviteOrganizationMemberRequest(email="grace@example.com"),
            config=config,
        )

    outcomes = await _race(sessions, attempt)

    invited = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    # Whichever racer wins the lock leaves the membership `invited` with a
    # fresh, unexpired invitation, so every loser re-reads that and raises
    # InvitationAlreadyPendingError, not OrganizationMemberAlreadyExistsError
    # (that one is for an *active* membership, which none of the racers here
    # ever produce: the starting status is `suspended`).
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, InvitationAlreadyPendingError)]
    assert len(invited) == 1
    assert len(conflicts) == _RACERS - 1

    membership = await OrganizationMemberRepository(async_db).get_by_organization_and_user(
        organization.id, added.user_id
    )
    assert membership is not None
    pending = await InvitationRepository(async_db).get_pending_by_organization_members([membership.id])
    assert len(pending) == 1


async def test_concurrent_accepts_of_one_invitation_produce_one_active_membership(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """accept_invitation's pending check is check-then-act too, with a worse failure mode.

    Without locking before the re-check, two concurrent accepts of the same
    token both see `pending`, both flip the membership, and both reach
    `_apply_workspace_assignments`, whose existing-then-create shape lets the
    second racer's insert violate `uq_workspace_member_workspace_user` as an
    uncaught `IntegrityError` on a public, unauthenticated endpoint, rather
    than the mapped `InvitationAlreadyUsedError` every other double-use path
    already answers with.
    """
    organization, owner = await _seed_owner(async_db)
    owner_row = await UserRepository(async_db).get(owner.id)
    assert owner_row is not None
    service = WorkspaceService(async_db, membership_listener=WorkspaceBudgetDefaultService(async_db))
    workspace = await service.create_workspace(
        user=owner_row,
        workspace_create=WorkspaceCreate(name="Research"),
    )
    config = GatewayConfig()
    invited = await OrganizationService(
        async_db, membership_listener=WorkspaceBudgetDefaultService(async_db)
    ).invite_active_organization_member_for_user(
        user=owner_row,
        request=InviteOrganizationMemberRequest(
            email="hank@example.com",
            workspace_assignments=[WorkspaceAssignmentRequest(workspace_id=workspace.id, role="viewer")],
        ),
        config=config,
    )
    token = invited.accept_link.split("token=")[1]

    async def attempt(session: AsyncSession) -> object:
        return await OrganizationService(
            session, membership_listener=WorkspaceBudgetDefaultService(session)
        ).accept_invitation(token)

    outcomes = await _race(sessions, attempt)

    accepted = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    already_used = [outcome for outcome in outcomes if isinstance(outcome, InvitationAlreadyUsedError)]
    assert len(accepted) == 1
    assert len(already_used) == _RACERS - 1

    # async_db's own invite call above left the membership row resident in this
    # session's identity map with status "invited"; with expire_on_commit=False,
    # a plain get() would return that unexpired cached instance rather than
    # querying the row the race committed through separate sessions.
    async_db.expire_all()
    membership = await OrganizationMemberRepository(async_db).get(invited.organization_member_id)
    assert membership is not None
    assert membership.status == "active"
    # Exactly one workspace_member row, not one per racer that reached
    # _apply_workspace_assignments before the lock closed this off.
    workspace_members = await WorkspaceMemberRepository(async_db).get_by_workspaces_and_user(
        {workspace.id: "viewer"}, membership.user_id
    )
    assert len(workspace_members) == 1


async def test_concurrent_accept_and_revoke_of_one_invitation_produce_one_consistent_outcome(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """revoke_organization_member_invitation_for_user races accept_invitation too.

    Both are check-then-act on the same invitation's `pending` status, and
    until revoke's organization lock moved ahead of its own reads, this had a
    worse failure mode than the accept-vs-accept race above: revoke's read of
    the invitation and membership happened before it took any lock, so a
    revoke that started just ahead of a winning accept could sit on the lock
    call inside `_validate_membership_update`, wake up once that accept had
    fully committed, and then unconditionally overwrite the accept's
    `active`/`accepted` state with its own stale, pre-lock `suspended`/
    `cancelled` write. Both operations would report success, and the invitee
    who just accepted would silently lose the membership they were told they
    had.
    """
    organization, owner = await _seed_owner(async_db)
    owner_row = await UserRepository(async_db).get(owner.id)
    assert owner_row is not None
    invited = await OrganizationService(
        async_db, membership_listener=WorkspaceBudgetDefaultService(async_db)
    ).invite_active_organization_member_for_user(
        user=owner_row,
        request=InviteOrganizationMemberRequest(email="ivy@example.com"),
        config=GatewayConfig(),
    )
    token = invited.accept_link.split("token=")[1]
    assert invited.invitation_id is not None

    async def accept(session: AsyncSession) -> object:
        return await OrganizationService(
            session, membership_listener=WorkspaceBudgetDefaultService(session)
        ).accept_invitation(token)

    async def revoke(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        assert invited.invitation_id is not None
        await OrganizationService(
            session, membership_listener=WorkspaceBudgetDefaultService(session)
        ).revoke_organization_member_invitation_for_user(
            user=user,
            invitation_id=invited.invitation_id,
        )
        return None

    async def run_one(attempt: Callable[[AsyncSession], object]) -> object:
        async with sessions() as session:
            try:
                return await attempt(session)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    outcomes = list(await asyncio.gather(*(run_one(attempt) for attempt in (accept, revoke))))

    # Exactly one side has to lose, and lose with the mapped error: accept and
    # revoke are different operations, but they are still racing the same
    # pending-to-something-else transition, so only one may win it.
    losses = [outcome for outcome in outcomes if isinstance(outcome, InvitationAlreadyUsedError)]
    assert len(losses) == 1

    async_db.expire_all()
    invitation = await InvitationRepository(async_db).get(invited.invitation_id)
    membership = await OrganizationMemberRepository(async_db).get(invited.organization_member_id)
    assert invitation is not None
    assert membership is not None
    # The invitation and its paired membership have to agree on which
    # operation won, never a mix showing one operation's write and the other's
    # leftover state.
    assert (invitation.status, membership.status) in {
        ("accepted", "active"),
        ("cancelled", "suspended"),
    }


async def test_concurrent_verifications_of_one_token_verify_exactly_once(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """verify_email's token lookup is check-then-act too, same shape as accept_invitation.

    Without locking before the re-check, two concurrent verifications of the
    same token both read a live, unexpired hash (nothing has committed yet to
    see) and both proceed, so the loser's write silently re-runs the same
    columns the winner already cleared instead of raising the mapped
    ``VerificationTokenInvalidError`` every other double-use path answers with.
    """
    organization, _ = await _seed_owner(async_db)
    users = UserRepository(async_db)
    identity = await users.create_local_identity(
        full_name="Vera", active_organization_id=organization.id, email="vera@example.com"
    )
    identity_id = identity.id
    token = generate_token()
    identity.email_verification_token_hash = hash_token(token)
    identity.email_verification_token_expires_at = datetime.now(UTC) + timedelta(hours=1)
    await async_db.commit()

    async def attempt(session: AsyncSession) -> object:
        return await user_service.verify_email(session, token=token)

    outcomes = await _race(sessions, attempt)

    verified = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    invalid = [outcome for outcome in outcomes if isinstance(outcome, VerificationTokenInvalidError)]
    assert len(verified) == 1
    assert len(invalid) == _RACERS - 1

    async_db.expire_all()
    row = await users.get(identity_id)
    assert row is not None
    assert row.email_verified_at is not None
    assert row.email_verification_token_hash is None
    assert row.email_verification_token_expires_at is None


async def test_concurrent_resets_of_one_token_change_the_password_exactly_once(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """reset_password's token lookup races the same way, with a worse failure mode.

    Under READ COMMITTED, two concurrent resets of the same token both read a
    live, unexpired hash and both proceed, each hashing a different new
    password and each receiving success: the last write wins, and the caller
    who reset it first has no way to know their password did not stick.
    """
    organization, _ = await _seed_owner(async_db)
    users = UserRepository(async_db)
    identity = await users.create_local_identity(
        full_name="Rex", active_organization_id=organization.id, email="rex@example.com"
    )
    identity_id = identity.id
    token = generate_token()
    identity.password_reset_token_hash = hash_token(token)
    identity.password_reset_token_expires_at = datetime.now(UTC) + timedelta(hours=1)
    await async_db.commit()

    async def run_one(index: int) -> object:
        async with sessions() as session:
            try:
                await user_service.reset_password(session, token=token, new_password=f"racer-password-{index}")
                return index
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    outcomes = list(await asyncio.gather(*(run_one(index) for index in range(_RACERS))))

    won = [outcome for outcome in outcomes if isinstance(outcome, int)]
    invalid = [outcome for outcome in outcomes if isinstance(outcome, ResetTokenInvalidError)]
    assert len(won) == 1
    assert len(invalid) == _RACERS - 1

    async_db.expire_all()
    row = await users.get(identity_id)
    assert row is not None
    assert row.password_reset_token_hash is None
    assert row.password_reset_token_expires_at is None
    assert row.hashed_password is not None
    assert await verify_password_async(f"racer-password-{won[0]}", row.hashed_password)


async def test_concurrent_first_issuance_leaves_one_setup_key(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Every racer asking a fresh workspace for its setup key gets the same one.

    The guide's promise is one "Setup guide" key per workspace, rotated in place,
    and on a first call there is no state row to lock, so the serialization point
    is the insert of that row rather than a pre-check. Minting the key before
    creating it (which is what this used to do) let each racer commit a key of
    its own, leaving a workspace with several live credentials nobody asked for
    and a state row naming whichever committed last.
    """
    organization, owner = await _seed_owner(async_db)
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Engineering",
        organization_id=organization.id,
        created_by_user_id=owner.id,
    )
    await WorkspaceMemberRepository(async_db).create(
        workspace_id=workspace.id,
        user_id=owner.id,
        role="owner",
    )
    await async_db.commit()

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await WorkspaceActivationService(session, GatewayConfig(), KEY_FORMAT).issue_api_key(
            user=user,
            workspace_id=workspace.id,
        )

    workspace_id = workspace.id
    outcomes = await _race(sessions, attempt)

    # Every racer, not merely one: a losing call has to come back with a key
    # rather than with the integrity error it hit on the way, which is the half
    # of this a "one row survived" assertion cannot see.
    issued = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    assert len(issued) == _RACERS, f"some racers failed: {outcomes}"

    # Expired first, and every value read out into a local before the next
    # await: this session predates the racers' commits, and a lazily refreshed
    # attribute on an ``AsyncSession`` raises ``MissingGreenlet`` rather than
    # reloading.
    async_db.expire_all()
    keys = (await async_db.execute(select(APIKey).where(col(APIKey.workspace_id) == workspace_id))).scalars().all()
    key_ids = [key.id for key in keys]
    key_names = [key.key_name for key in keys]
    key_hash = keys[0].key_hash
    state = await async_db.get(WorkspaceActivationState, workspace_id)
    assert state is not None
    named_key = state.api_key_id

    # One row, and every racer that got a key got that row's, so no plaintext was
    # handed out for a credential the workspace does not carry.
    assert key_names == [ACTIVATION_KEY_NAME]
    assert {issue.key_id for issue in issued} == set(key_ids)  # type: ignore[attr-defined]
    assert named_key == key_ids[0]

    # Exactly one of the plaintexts authenticates, and that is the design rather
    # than a gap in it: each call rotates the row, so the last writer's key is
    # the live one and the earlier ones are invalidated exactly as a page reload
    # invalidates the key before it. Asserting it here keeps that a stated
    # property, so a future change that quietly handed out several live keys for
    # one row would fail rather than look like an improvement.
    live = [issue for issue in issued if hash_key(issue.key) == key_hash]  # type: ignore[attr-defined]
    assert len(live) == 1
