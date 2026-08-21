"""Workspace per-member budget defaults: materialization, CRUD, and authorization.

Exercised at the service layer, matching `test_tenancy_authorization.py`: the
API can only ever act as the one superuser operator identity a standalone
deployment has, so the rules that matter most (a non-management member
refused, a foreign workspace refused) are only reachable by calling the
services with identities built at whatever role a case needs.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.models.entities import Budget, ScopedBudget, WorkspaceBudgetDefault
from gateway.models.tenancy import (
    ActiveOrganizationMemberCreateRequest,
    Organization,
    User,
    Workspace,
    WorkspaceAssignmentRequest,
)
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy import OrganizationService, WorkspaceService
from gateway.services.tenancy.errors import (
    NotAuthorizedError,
    WorkspaceBudgetDefaultAlreadyExistsError,
    WorkspaceBudgetDefaultNotFoundError,
    WorkspaceNotFoundError,
)
from gateway.services.tenancy.workspace_budget_default_service import (
    WorkspaceBudgetDefaultService,
    WorkspaceMemberBudgetPolicyCreate,
    WorkspaceMemberBudgetPolicyUpdate,
)

pytestmark = pytest.mark.asyncio


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(
    db: AsyncSession,
    organization: Organization,
    *,
    role: str,
    full_name: str,
) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role=role,
    )
    return user


async def _workspace(db: AsyncSession, organization: Organization, *, name: str, owner: User) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name,
        organization_id=organization.id,
        created_by_user_id=owner.id,
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    return workspace


async def _member_budget(
    db: AsyncSession, member_id: uuid.UUID, *, provider_key_id: str | None = None
) -> ScopedBudget | None:
    stmt = select(ScopedBudget).where(
        ScopedBudget.scope_type == "workspace_member",
        ScopedBudget.scope_id == str(member_id),
    )
    stmt = stmt.where(
        ScopedBudget.provider_key_id == provider_key_id
        if provider_key_id is not None
        else ScopedBudget.provider_key_id.is_(None)
    )
    return (await db.execute(stmt)).scalars().first()


async def _budget(
    db: AsyncSession,
    *,
    max_budget: float | None = None,
    budget_duration_sec: int | None = None,
    name: str | None = None,
) -> str:
    """A budget for a default to hand out, returning its id.

    A default no longer carries a limit of its own: it names a ``budgets`` row,
    which is what lets the Budgets page say a limit is a workspace's default.
    """
    budget = Budget(name=name, max_budget=max_budget, budget_duration_sec=budget_duration_sec)
    db.add(budget)
    await db.flush()
    return budget.budget_id


async def test_create_materializes_onto_existing_members_but_skips_an_override(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="acme-create")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    other = await _member(async_db, org, role="member", full_name="Other")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    workspace_members = WorkspaceMemberRepository(async_db)
    other_member = await workspace_members.create(workspace_id=workspace.id, user_id=other.id, role="member")

    # `other` already has a ceiling for this scope; the default must not touch it.
    override = ScopedBudget(scope_type="workspace_member", scope_id=str(other_member.id), max_budget=999.0)
    async_db.add(override)
    await async_db.commit()

    service = WorkspaceBudgetDefaultService(async_db)
    created = await service.create_default(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, name="Default", max_budget=50.0, budget_duration_sec=86400)),
    )
    assert created.max_budget == 50.0

    owner_member = await workspace_members.get_by_workspace_and_user(workspace.id, owner.id)
    assert owner_member is not None
    owner_budget = await _member_budget(async_db, owner_member.id)
    assert owner_budget is not None
    assert owner_budget.max_budget == 50.0
    assert owner_budget.budget_duration_sec == 86400

    other_budget = await _member_budget(async_db, other_member.id)
    assert other_budget is not None
    assert other_budget.max_budget == 999.0, "an existing member-specific ceiling must win over the template"


async def test_member_added_afterwards_is_materialized_on_join(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="acme-join")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    joiner = await _member(async_db, org, role="member", full_name="Joiner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    service = WorkspaceBudgetDefaultService(async_db)
    await service.create_default(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, max_budget=25.0)),
    )

    workspace_service = WorkspaceService(async_db)
    added = await workspace_service.add_member(user=owner, workspace_id=workspace.id, user_id=joiner.id)

    budget = await _member_budget(async_db, added.id)
    assert budget is not None
    assert budget.max_budget == 25.0


async def test_member_added_via_organization_workspace_assignment_is_materialized(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="acme-assign")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    service = WorkspaceBudgetDefaultService(async_db)
    await service.create_default(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, max_budget=15.0)),
    )

    organization_service = OrganizationService(async_db)
    result = await organization_service.create_active_organization_member_for_user(
        user=owner,
        request=ActiveOrganizationMemberCreateRequest(
            email="new-hire@example.test",
            role="member",
            workspace_assignments=[WorkspaceAssignmentRequest(workspace_id=workspace.id, role="member")],
        ),
    )

    assert result.user_id is not None
    workspace_member = await WorkspaceMemberRepository(async_db).get_by_workspace_and_user(
        workspace.id, result.user_id
    )
    assert workspace_member is not None
    budget = await _member_budget(async_db, workspace_member.id)
    assert budget is not None
    assert budget.max_budget == 15.0


async def test_reviving_a_suspended_workspace_membership_is_materialized(async_db: AsyncSession) -> None:
    """A member revived from suspended, not just one freshly created, gets caught up.

    There is no service-level producer of a suspended `WorkspaceMember` row in
    this edition yet (workspace removal deletes rather than suspends), so the
    row is built directly through the repository, the way the row itself would
    look if one arrives (an import, or a future suspend action) and a default
    was created while it was suspended.
    """
    org = await _organization(async_db, slug="acme-revive")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    target = await UserRepository(async_db).create_local_identity(
        full_name="Revived",
        email="revived@example.test",
        active_organization_id=org.id,
    )
    suspended_member = await WorkspaceMemberRepository(async_db).create(
        workspace_id=workspace.id,
        user_id=target.id,
        role="member",
        status="suspended",
    )
    await async_db.commit()

    service = WorkspaceBudgetDefaultService(async_db)
    await service.create_default(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, max_budget=35.0)),
    )
    # The default fans out to active members only; the suspended row gets
    # nothing from it yet.
    assert await _member_budget(async_db, suspended_member.id) is None

    organization_service = OrganizationService(async_db)
    await organization_service.create_active_organization_member_for_user(
        user=owner,
        request=ActiveOrganizationMemberCreateRequest(
            email="revived@example.test",
            role="member",
            workspace_assignments=[WorkspaceAssignmentRequest(workspace_id=workspace.id, role="member")],
        ),
    )

    revived = await WorkspaceMemberRepository(async_db).get_by_workspace_and_user(workspace.id, target.id)
    assert revived is not None
    assert revived.status == "active"
    budget = await _member_budget(async_db, revived.id)
    assert budget is not None, "a member revived from suspended must be materialized, not just reactivated"
    assert budget.max_budget == 35.0


async def test_reapplying_an_active_assignment_does_not_rematerialize_a_deleted_override(
    async_db: AsyncSession,
) -> None:
    """Re-applying an assignment to an already-active member is not a join.

    `_apply_workspace_assignments`'s revive branch also runs for a membership
    that was already active (an idempotent re-post, or a second invitation
    naming the same workspace); only a row that was actually inactive is a
    real join. Without the status gate, re-applying the assignment would
    resurrect a per-member ceiling an admin deliberately deleted through
    `/v1/scoped-budgets`.
    """
    org = await _organization(async_db, slug="acme-reapply")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    member_user = await _member(async_db, org, role="member", full_name="Member")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    workspace_member = await WorkspaceMemberRepository(async_db).create(
        workspace_id=workspace.id, user_id=member_user.id, role="member"
    )
    await async_db.commit()

    service = WorkspaceBudgetDefaultService(async_db)
    await service.create_default(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, max_budget=50.0)),
    )
    budget = await _member_budget(async_db, workspace_member.id)
    assert budget is not None

    # An admin deletes the member's own ceiling directly.
    await async_db.delete(budget)
    await async_db.commit()
    assert await _member_budget(async_db, workspace_member.id) is None

    organization_service = OrganizationService(async_db)
    await organization_service._apply_workspace_assignments(  # noqa: SLF001 - exercising the internal gate directly
        user_id=member_user.id,
        assignments=[WorkspaceAssignmentRequest(workspace_id=workspace.id, role="member")],
    )
    await async_db.commit()

    assert await _member_budget(async_db, workspace_member.id) is None, (
        "re-applying an already-active assignment must not re-materialize a deleted override"
    )


async def test_update_is_not_retroactive(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="acme-update")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    later_joiner = await _member(async_db, org, role="member", full_name="Later")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    service = WorkspaceBudgetDefaultService(async_db)
    default = await service.create_default(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, max_budget=10.0)),
    )
    owner_member = await WorkspaceMemberRepository(async_db).get_by_workspace_and_user(workspace.id, owner.id)
    assert owner_member is not None
    owner_budget_before = await _member_budget(async_db, owner_member.id)
    assert owner_budget_before is not None
    assert owner_budget_before.max_budget == 10.0

    await service.update_default(
        user=owner,
        workspace_id=workspace.id,
        default_id=default.id,
        request=WorkspaceMemberBudgetPolicyUpdate(budget_id=await _budget(async_db, max_budget=20.0)),
    )

    owner_budget_after = await _member_budget(async_db, owner_member.id)
    assert owner_budget_after is not None
    assert owner_budget_after.max_budget == 10.0, "an already-materialized ceiling must not be rewritten"

    workspace_service = WorkspaceService(async_db)
    joined = await workspace_service.add_member(user=owner, workspace_id=workspace.id, user_id=later_joiner.id)
    joiner_budget = await _member_budget(async_db, joined.id)
    assert joiner_budget is not None
    assert joiner_budget.max_budget == 20.0, "a member joining after the edit must get the new value"


async def test_delete_preserves_materialized_rows_and_stops_future_ones(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="acme-delete")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    later_joiner = await _member(async_db, org, role="member", full_name="Later")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    service = WorkspaceBudgetDefaultService(async_db)
    default = await service.create_default(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, max_budget=30.0)),
    )
    owner_member = await WorkspaceMemberRepository(async_db).get_by_workspace_and_user(workspace.id, owner.id)
    assert owner_member is not None

    await service.delete_default(user=owner, workspace_id=workspace.id, default_id=default.id)

    owner_budget = await _member_budget(async_db, owner_member.id)
    assert owner_budget is not None, "spend history on an already-materialized ceiling survives the delete"

    workspace_service = WorkspaceService(async_db)
    joined = await workspace_service.add_member(user=owner, workspace_id=workspace.id, user_id=later_joiner.id)
    assert await _member_budget(async_db, joined.id) is None, "a member joining after the delete gets nothing from it"

    with pytest.raises(WorkspaceBudgetDefaultNotFoundError):
        await service._get_or_404(workspace, default.id)  # noqa: SLF001 - asserting the row is actually gone


async def test_duplicate_aggregate_default_conflicts(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="acme-conflict")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    service = WorkspaceBudgetDefaultService(async_db)
    await service.create_default(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, max_budget=10.0)),
    )

    with pytest.raises(WorkspaceBudgetDefaultAlreadyExistsError):
        await service.create_default(
            user=owner,
            workspace_id=workspace.id,
            request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, max_budget=20.0)),
        )


async def test_non_management_member_may_list_but_not_write(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="acme-authz")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    plain = await _member(async_db, org, role="member", full_name="Plain")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=plain.id, role="member")

    service = WorkspaceBudgetDefaultService(async_db)
    default = await service.create_default(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, max_budget=10.0)),
    )

    listed = await service.list_defaults(user=plain, workspace_id=workspace.id)
    assert listed.count == 1

    with pytest.raises(NotAuthorizedError):
        await service.create_default(
            user=plain,
            workspace_id=workspace.id,
            request=WorkspaceMemberBudgetPolicyCreate(budget_id=await _budget(async_db, max_budget=99.0)),
        )

    with pytest.raises(NotAuthorizedError):
        await service.update_default(
            user=plain,
            workspace_id=workspace.id,
            default_id=default.id,
            request=WorkspaceMemberBudgetPolicyUpdate(budget_id=await _budget(async_db, max_budget=99.0)),
        )

    with pytest.raises(NotAuthorizedError):
        await service.delete_default(user=plain, workspace_id=workspace.id, default_id=default.id)


async def test_foreign_workspace_is_not_found(async_db: AsyncSession) -> None:
    org_a = await _organization(async_db, slug="acme-foreign-a")
    org_b = await _organization(async_db, slug="acme-foreign-b")
    owner_a = await _member(async_db, org_a, role="owner", full_name="Owner A")
    owner_b = await _member(async_db, org_b, role="owner", full_name="Owner B")
    workspace_b = await _workspace(async_db, org_b, name="Elsewhere", owner=owner_b)

    service = WorkspaceBudgetDefaultService(async_db)
    with pytest.raises(WorkspaceNotFoundError):
        await service.list_defaults(user=owner_a, workspace_id=workspace_b.id)


# =============================================================================
# _insert_member_budgets' savepoint fallback, exercised directly: a batch
# collision must fall back to a per-row retry rather than 500ing the request
# or leaving the session unusable.
# =============================================================================


async def test_materialize_batch_recovers_from_a_missed_collision(async_db: AsyncSession) -> None:
    """A batch collision falls back to a per-row retry instead of failing outright.

    Reproduces the shape of the race `_insert_member_budgets` exists to
    survive: by the time the insert runs, a ceiling already exists for one of
    the ids in the batch (standing in for a concurrent direct
    ``POST /v1/scoped-budgets`` for that member, which the batch's own
    existence check, run a moment earlier, would not yet have seen).
    Called directly rather than through `materialize_for_default`, which
    otherwise wraps the same existence check around every id and would just
    filter the colliding one out before ever attempting to insert it.
    """
    org = await _organization(async_db, slug="acme-collision")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    other = await _member(async_db, org, role="member", full_name="Other")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    workspace_members = WorkspaceMemberRepository(async_db)
    other_member = await workspace_members.create(workspace_id=workspace.id, user_id=other.id, role="member")
    owner_member = await workspace_members.get_by_workspace_and_user(workspace.id, owner.id)
    assert owner_member is not None

    default = WorkspaceBudgetDefault(
        workspace_id=workspace.id, budget_id=await _budget(async_db, max_budget=40.0)
    )
    async_db.add(default)
    await async_db.flush()

    # The row `_insert_member_budgets` will collide with, inserted directly
    # (not through the check-then-insert path this test bypasses).
    collision = ScopedBudget(scope_type="workspace_member", scope_id=str(other_member.id), max_budget=999.0)
    async_db.add(collision)
    await async_db.flush()

    service = WorkspaceBudgetDefaultService(async_db)
    created = await service._insert_member_budgets(  # noqa: SLF001 - exercising the fallback directly
        [owner_member.id, other_member.id], default, await service._budget_for(default)  # noqa: SLF001
    )

    assert {budget.scope_id for budget in created} == {str(owner_member.id)}
    owner_budget = await _member_budget(async_db, owner_member.id)
    assert owner_budget is not None
    assert owner_budget.max_budget == 40.0
    other_budget = await _member_budget(async_db, other_member.id)
    assert other_budget is not None
    assert other_budget.max_budget == 999.0, "the pre-existing row must survive the collision untouched"

    # The broken version failed exactly here: PendingRollbackError on the next
    # statement, because the failed batch flush had already dirtied the outer
    # transaction rather than just the savepoint.
    await async_db.commit()


# =============================================================================
# The race between creating a default and a member joining, driven concurrently
# =============================================================================


@pytest_asyncio.fixture
async def sessions(postgres_url: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory on its own engine, disposed after the test.

    Undisposed, each test using this leaves an asyncpg connection pool alive
    until garbage collection, which tends to surface later as a
    connection-limit failure or "event loop is closed" noise in an unrelated
    test rather than as a failure here.
    """
    url = postgres_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://").replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    engine = create_async_engine(url)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def test_concurrent_default_create_and_member_add_both_land(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """A default created at the same moment a member joins must reach that member either way.

    Without `WorkspaceRepository.lock` serializing the two paths, each can read
    the other's pre-write state under READ COMMITTED: the create reads the
    members before the join commits, and the join reads the defaults before the
    create commits, and both still succeed, leaving the new member with no
    ceiling from a default that, from the outside, looks like it was already
    there when they joined.
    """
    org = await _organization(async_db, slug="acme-race")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    joiner = await _member(async_db, org, role="member", full_name="Joiner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    # The racing sessions below are separate connections and must see this
    # graph committed, not merely flushed on `async_db`. The budget the default
    # will name is part of that graph.
    budget_id = await _budget(async_db, max_budget=40.0)
    await async_db.commit()

    async def create_default_attempt() -> object:
        async with sessions() as session:
            user = await UserRepository(session).get(owner.id)
            assert user is not None
            try:
                return await WorkspaceBudgetDefaultService(session).create_default(
                    user=user,
                    workspace_id=workspace.id,
                    request=WorkspaceMemberBudgetPolicyCreate(budget_id=budget_id),
                )
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    async def add_member_attempt() -> object:
        async with sessions() as session:
            user = await UserRepository(session).get(owner.id)
            assert user is not None
            try:
                return await WorkspaceService(session).add_member(
                    user=user,
                    workspace_id=workspace.id,
                    user_id=joiner.id,
                )
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    outcomes = await asyncio.gather(create_default_attempt(), add_member_attempt())
    for outcome in outcomes:
        assert not isinstance(outcome, Exception), outcome

    joined_member = await WorkspaceMemberRepository(async_db).get_by_workspace_and_user(workspace.id, joiner.id)
    assert joined_member is not None
    budget = await _member_budget(async_db, joined_member.id)
    assert budget is not None, "the new member must get the default whichever transaction committed first"
    assert budget.max_budget == 40.0
