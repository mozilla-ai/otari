"""The budget repositories answer the organization surface's queries on a Unit of Work."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import OutsideUnitOfWorkError, UnitOfWork
from gateway.exceptions.budget_exceptions import (
    BudgetStillReferencedError,
    MemberBudgetPolicyAlreadyExistsError,
    SpendCeilingAlreadyExistsError,
)
from gateway.models.api_keys import APIKey
from gateway.models.budgets import (
    SCOPE_API_TOKEN,
    SCOPE_ORG_MEMBER,
    SCOPE_ORGANIZATION,
    SCOPE_WORKSPACE,
    SCOPE_WORKSPACE_MEMBER,
    Budget,
    BudgetResetLog,
    ScopedBudget,
    WorkspaceBudgetDefault,
)
from gateway.models.tenancy import Organization, User, Workspace
from gateway.models.users import User as ApiUser
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.repositories.budgets import (
    BudgetRepositories,
    BudgetRepository,
    ScopedBudgetRepository,
    ScopeIdSets,
    WorkspaceBudgetDefaultRepository,
)
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets import BudgetService
from gateway.services.tenancy.organization_service import OrganizationService

pytestmark = pytest.mark.asyncio


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _budget(db: AsyncSession, organization: Organization | None, *, name: str) -> Budget:
    budget = Budget(organization_id=None if organization is None else organization.id, name=name)
    db.add(budget)
    await db.flush()
    return budget


async def _ceiling(db: AsyncSession, budget: Budget, *, scope_type: str, scope_id: str) -> ScopedBudget:
    ceiling = ScopedBudget(scope_type=scope_type, scope_id=scope_id, budget_id=budget.budget_id)
    db.add(ceiling)
    await db.flush()
    return ceiling


async def _scopes_of(db: AsyncSession, organization: Organization) -> tuple[User, ScopeIdSets]:
    owner = await UserRepository(db).create_local_identity(
        full_name=f"{organization.slug} owner",
        active_organization_id=organization.id,
        is_superuser=False,
    )
    organization_member = await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=owner.id, role="owner"
    )
    workspace = await WorkspaceRepository(db).create_workspace(
        name=f"{organization.slug} workspace", organization_id=organization.id, created_by_user_id=owner.id
    )
    workspace_member = await WorkspaceMemberRepository(db).create(
        workspace_id=workspace.id, user_id=owner.id, role="owner"
    )
    key_id = f"sk-{organization.slug}"
    db.add(APIKey(id=key_id, key_hash=f"hash-{key_id}", workspace_id=workspace.id))
    await db.flush()
    return owner, ScopeIdSets(
        organization_ids=(str(organization.id),),
        workspace_ids=(str(workspace.id),),
        organization_member_ids=(str(organization_member.id),),
        workspace_member_ids=(str(workspace_member.id),),
        api_key_ids=(key_id,),
    )


async def _one_ceiling_per_scope(db: AsyncSession, budget: Budget, scopes: ScopeIdSets) -> list[str]:
    ceilings = [
        await _ceiling(db, budget, scope_type=SCOPE_ORGANIZATION, scope_id=scopes.organization_ids[0]),
        await _ceiling(db, budget, scope_type=SCOPE_WORKSPACE, scope_id=scopes.workspace_ids[0]),
        await _ceiling(db, budget, scope_type=SCOPE_ORG_MEMBER, scope_id=scopes.organization_member_ids[0]),
        await _ceiling(db, budget, scope_type=SCOPE_WORKSPACE_MEMBER, scope_id=scopes.workspace_member_ids[0]),
        await _ceiling(db, budget, scope_type=SCOPE_API_TOKEN, scope_id=scopes.api_key_ids[0]),
    ]
    return [ceiling.id for ceiling in ceilings]


async def _workspace(db: AsyncSession, organization: Organization, *, name: str) -> Workspace:
    owner = await UserRepository(db).create_local_identity(
        full_name=f"{name} owner", active_organization_id=organization.id
    )
    return await WorkspaceRepository(db).create_workspace(
        name=name, organization_id=organization.id, created_by_user_id=owner.id
    )


async def _policy(
    db: AsyncSession,
    workspace: Workspace,
    budget: Budget,
    *,
    provider_key_id: str | None = None,
    created_at: datetime | None = None,
) -> WorkspaceBudgetDefault:
    policy = WorkspaceBudgetDefault(
        workspace_id=workspace.id,
        budget_id=budget.budget_id,
        provider_key_id=provider_key_id,
        **({} if created_at is None else {"created_at": created_at}),
    )
    db.add(policy)
    await db.flush()
    return policy


def _member_ceiling(member_id: uuid.UUID, budget_id: str, *, provider_key_id: str | None = None) -> ScopedBudget:
    return ScopedBudget(
        scope_type=SCOPE_WORKSPACE_MEMBER,
        scope_id=str(member_id),
        provider_key_id=provider_key_id,
        budget_id=budget_id,
    )


async def test_get_by_id_and_organization_answers_only_the_owners_budget(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    globex = await _organization(async_db, slug="globex")
    own = await _budget(async_db, acme, name="Acme monthly")
    foreign = await _budget(async_db, globex, name="Globex monthly")
    deployment = await _budget(async_db, None, name="Deployment")
    uow = UnitOfWork(async_db)

    async with uow:
        budgets = BudgetRepository(uow)
        found = await budgets.get_by_id_and_organization(own.budget_id, acme.id)
        assert found is not None
        assert found.budget_id == own.budget_id
        assert await budgets.get_by_id_and_organization(foreign.budget_id, acme.id) is None
        assert await budgets.get_by_id_and_organization(deployment.budget_id, acme.id) is None
        assert await budgets.get_by_id_and_organization("missing", acme.id) is None


async def test_list_by_organization_pages_oldest_first_and_count_matches(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    globex = await _organization(async_db, slug="globex")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for offset, name in enumerate(["first", "second", "third"]):
        async_db.add(Budget(organization_id=acme.id, name=name, created_at=start + timedelta(days=offset)))
    await _budget(async_db, globex, name="elsewhere")
    await async_db.flush()
    uow = UnitOfWork(async_db)

    async with uow:
        budgets = BudgetRepository(uow)
        page = await budgets.list_by_organization(acme.id, skip=0, limit=2)
        assert [budget.name for budget in page] == ["first", "second"]
        rest = await budgets.list_by_organization(acme.id, skip=2, limit=2)
        assert [budget.name for budget in rest] == ["third"]
        assert await budgets.count_by_organization(acme.id) == 3
        assert await budgets.count_by_organization(globex.id) == 1


async def test_list_by_organization_breaks_a_created_at_tie_on_the_id(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    same_moment = datetime(2026, 1, 1, tzinfo=UTC)
    for budget_id in ("b-2", "b-1", "b-3"):
        async_db.add(Budget(budget_id=budget_id, organization_id=acme.id, created_at=same_moment))
    await async_db.flush()
    uow = UnitOfWork(async_db)

    async with uow:
        page = await BudgetRepository(uow).list_by_organization(acme.id, skip=0, limit=10)
        assert [budget.budget_id for budget in page] == ["b-1", "b-2", "b-3"]


async def test_count_member_policies_and_users_for_budget(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    _, scopes = await _scopes_of(async_db, acme)
    held = await _budget(async_db, acme, name="held")
    free = await _budget(async_db, acme, name="free")
    async_db.add(WorkspaceBudgetDefault(workspace_id=uuid.UUID(scopes.workspace_ids[0]), budget_id=held.budget_id))
    async_db.add(ApiUser(user_id="capped-a", budget_id=held.budget_id))
    async_db.add(ApiUser(user_id="capped-b", budget_id=held.budget_id))
    await async_db.flush()
    uow = UnitOfWork(async_db)

    async with uow:
        budgets = BudgetRepository(uow)
        policies = WorkspaceBudgetDefaultRepository(uow)
        assert await policies.count_for_budget(held.budget_id) == 1
        assert await policies.count_for_budget(free.budget_id) == 0
        assert await budgets.count_users_for_budget(held.budget_id) == 2
        assert await budgets.count_users_for_budget(free.budget_id) == 0


async def test_add_stages_a_budget_with_its_generated_values(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    uow = UnitOfWork(async_db)

    async with uow:
        budget = await BudgetRepository(uow).add(Budget(organization_id=acme.id, name="new"))
        assert budget.budget_id
        assert budget.created_at is not None

    async with uow:
        assert await BudgetRepository(uow).count_by_organization(acme.id) == 1


async def test_remove_deletes_a_budget_nothing_names(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="doomed")
    uow = UnitOfWork(async_db)

    async with uow:
        budgets = BudgetRepository(uow)
        await budgets.remove(budget)
        assert await budgets.count_by_organization(acme.id) == 0


async def test_remove_raises_while_a_reset_record_names_the_budget(async_db: AsyncSession) -> None:
    """A reset log's ``budget_id`` is NOT NULL, so the delete fails at the flush rather than detaching the row.

    The seed is committed first, so the budget is a persistent row that the failed flush expires rather than expunges.
    """
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="logged")
    async_db.add(ApiUser(user_id="detached-user", budget_id=None))
    await async_db.flush()
    async_db.add(
        BudgetResetLog(
            user_id="detached-user",
            budget_id=budget.budget_id,
            previous_spend=Decimal("1.5"),
            reset_at=datetime.now(UTC),
        )
    )
    await async_db.commit()
    uow = UnitOfWork(async_db)

    with pytest.raises(BudgetStillReferencedError):
        async with uow:
            await BudgetRepository(uow).remove(budget)


async def test_count_for_budget_and_count_for_budgets(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    twice = await _budget(async_db, acme, name="twice")
    once = await _budget(async_db, acme, name="once")
    never = await _budget(async_db, acme, name="never")
    await _ceiling(async_db, twice, scope_type=SCOPE_ORGANIZATION, scope_id=str(acme.id))
    await _ceiling(async_db, twice, scope_type=SCOPE_WORKSPACE, scope_id="ws-1")
    await _ceiling(async_db, once, scope_type=SCOPE_WORKSPACE, scope_id="ws-2")
    uow = UnitOfWork(async_db)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        assert await ceilings.count_for_budget(twice.budget_id) == 2
        assert await ceilings.count_for_budget(never.budget_id) == 0
        counts = await ceilings.count_for_budgets([twice.budget_id, once.budget_id, never.budget_id])
        assert counts == {twice.budget_id: 2, once.budget_id: 1}
        assert await ceilings.count_for_budgets([]) == {}


async def test_list_in_scopes_matches_what_the_organization_surface_lists(async_db: AsyncSession) -> None:
    """Every scope kind is filtered on its own ID set, and another organization's ceilings stay out."""
    acme = await _organization(async_db, slug="acme")
    globex = await _organization(async_db, slug="globex")
    acme_owner, acme_scopes = await _scopes_of(async_db, acme)
    _, globex_scopes = await _scopes_of(async_db, globex)
    acme_budget = await _budget(async_db, acme, name="acme")
    globex_budget = await _budget(async_db, globex, name="globex")
    expected = await _one_ceiling_per_scope(async_db, acme_budget, acme_scopes)
    await _one_ceiling_per_scope(async_db, globex_budget, globex_scopes)
    await async_db.commit()
    uow = UnitOfWork(async_db)
    service = BudgetService(
        uow,
        BudgetRepositories.on(uow),
        OrganizationService(async_db, membership_listener=None),
        ApiKeyService(ApiKeyRepository(uow)),
    )
    surface = await service.list_organization_ceilings(user=acme_owner)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        rows = await ceilings.list_in_scopes(acme_scopes, skip=0, limit=10)
        assert [ceiling.id for ceiling, _ in rows] == expected
        assert [ceiling.id for ceiling, _ in rows] == [row.id for row in surface.data]
        assert all(budget.budget_id == acme_budget.budget_id for _, budget in rows)
        assert await ceilings.count_in_scopes(acme_scopes) == 5
        assert await ceilings.count_in_scopes(acme_scopes) == surface.count

        page = await ceilings.list_in_scopes(acme_scopes, skip=3, limit=10)
        assert [ceiling.id for ceiling, _ in page] == expected[3:]

        empty = ScopeIdSets((), (), (), (), ())
        assert await ceilings.list_in_scopes(empty, skip=0, limit=10) == []
        assert await ceilings.count_in_scopes(empty) == 0


async def test_has_ceiling_is_per_scope_and_provider(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    async_db.add(ScopedBudget(scope_type=SCOPE_WORKSPACE, scope_id="ws-1", budget_id=budget.budget_id))
    async_db.add(
        ScopedBudget(scope_type=SCOPE_WORKSPACE, scope_id="ws-2", provider_key_id="pk-a", budget_id=budget.budget_id)
    )
    await async_db.flush()
    uow = UnitOfWork(async_db)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        assert await ceilings.has_ceiling(SCOPE_WORKSPACE, "ws-1", None)
        assert not await ceilings.has_ceiling(SCOPE_WORKSPACE, "ws-1", "pk-a")
        assert await ceilings.has_ceiling(SCOPE_WORKSPACE, "ws-2", "pk-a")
        assert not await ceilings.has_ceiling(SCOPE_WORKSPACE, "ws-2", None)
        assert not await ceilings.has_ceiling(SCOPE_WORKSPACE, "ws-2", "pk-b")
        assert not await ceilings.has_ceiling(SCOPE_ORGANIZATION, "ws-1", None)


async def test_add_stages_a_ceiling_and_refuses_a_duplicate(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget_id = (await _budget(async_db, acme, name="acme")).budget_id
    uow = UnitOfWork(async_db)

    async with uow:
        ceiling = await ScopedBudgetRepository(uow).add(
            ScopedBudget(scope_type=SCOPE_WORKSPACE, scope_id="ws-1", budget_id=budget_id)
        )
        assert ceiling.id
        assert ceiling.created_at is not None
    ceiling_id = ceiling.id

    with pytest.raises(SpendCeilingAlreadyExistsError):
        async with uow:
            await ScopedBudgetRepository(uow).add(
                ScopedBudget(scope_type=SCOPE_WORKSPACE, scope_id="ws-1", budget_id=budget_id)
            )

    async with uow:
        narrowed = await ScopedBudgetRepository(uow).add(
            ScopedBudget(scope_type=SCOPE_WORKSPACE, scope_id="ws-1", provider_key_id="pk-a", budget_id=budget_id)
        )
        assert narrowed.id != ceiling_id


async def test_remove_deletes_a_ceiling(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    ceiling = await _ceiling(async_db, budget, scope_type=SCOPE_WORKSPACE, scope_id="ws-1")
    uow = UnitOfWork(async_db)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        await ceilings.remove(ceiling)
        assert await ceilings.count_for_budget(budget.budget_id) == 0


async def test_retime_for_budget_rewrites_the_window_and_keeps_the_counters(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    retimed = await _budget(async_db, acme, name="retimed")
    other = await _budget(async_db, acme, name="other")
    async_db.add(
        ScopedBudget(
            scope_type=SCOPE_WORKSPACE, scope_id="ws-1", budget_id=retimed.budget_id, current_spend=Decimal("2.5")
        )
    )
    async_db.add(ScopedBudget(scope_type=SCOPE_WORKSPACE, scope_id="ws-2", budget_id=other.budget_id))
    await async_db.flush()
    start = datetime(2026, 2, 1, tzinfo=UTC)
    end = datetime(2026, 3, 1, tzinfo=UTC)
    uow = UnitOfWork(async_db)

    async with uow:
        await ScopedBudgetRepository(uow).retime_for_budget(retimed.budget_id, period_start=start, period_end=end)

    async_db.expire_all()
    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        rows = await ceilings.list_in_scopes(ScopeIdSets((), ("ws-1", "ws-2"), (), (), ()), skip=0, limit=10)
        by_scope = {ceiling.scope_id: ceiling for ceiling, _ in rows}
        assert (by_scope["ws-1"].period_start, by_scope["ws-1"].period_end) == (start, end)
        assert by_scope["ws-1"].current_spend == Decimal("2.5")
        assert (by_scope["ws-2"].period_start, by_scope["ws-2"].period_end) == (None, None)

    async with uow:
        await ScopedBudgetRepository(uow).retime_for_budget(retimed.budget_id, period_start=None, period_end=None)

    async_db.expire_all()
    async with uow:
        only_first = ScopeIdSets((), ("ws-1",), (), (), ())
        rows = await ScopedBudgetRepository(uow).list_in_scopes(only_first, skip=0, limit=10)
        assert (rows[0][0].period_start, rows[0][0].period_end) == (None, None)


async def test_for_workspace_lists_only_that_workspaces_policies(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    workspace = await _workspace(async_db, acme, name="acme one")
    other = await _workspace(async_db, acme, name="acme two")
    mine = [
        await _policy(async_db, workspace, budget),
        await _policy(async_db, workspace, budget, provider_key_id="pk-a"),
    ]
    await _policy(async_db, other, budget)
    uow = UnitOfWork(async_db)

    async with uow:
        policies = WorkspaceBudgetDefaultRepository(uow)
        assert sorted(policy.id for policy in await policies.for_workspace(workspace.id)) == sorted(
            policy.id for policy in mine
        )
        assert await policies.for_workspace(uuid.uuid4()) == []


async def test_page_for_workspace_pages_oldest_first_and_counts_every_policy(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    workspace = await _workspace(async_db, acme, name="acme one")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    expected = [
        (await _policy(async_db, workspace, budget, provider_key_id=key, created_at=start + timedelta(days=offset))).id
        for offset, key in enumerate([None, "pk-a", "pk-b"])
    ]
    uow = UnitOfWork(async_db)

    async with uow:
        policies = WorkspaceBudgetDefaultRepository(uow)
        page, count = await policies.page_for_workspace(workspace.id, skip=0, limit=2)
        assert [policy.id for policy in page] == expected[:2]
        assert count == 3
        rest, _ = await policies.page_for_workspace(workspace.id, skip=2, limit=2)
        assert [policy.id for policy in rest] == expected[2:]
        assert await policies.page_for_workspace(uuid.uuid4(), skip=0, limit=2) == ([], 0)


async def test_get_in_workspace_answers_only_a_policy_on_that_workspace(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    workspace = await _workspace(async_db, acme, name="acme one")
    other = await _workspace(async_db, acme, name="acme two")
    mine = await _policy(async_db, workspace, budget)
    foreign = await _policy(async_db, other, budget)
    uow = UnitOfWork(async_db)

    async with uow:
        policies = WorkspaceBudgetDefaultRepository(uow)
        found = await policies.get_in_workspace(mine.id, workspace.id)
        assert found is not None
        assert found.id == mine.id
        assert await policies.get_in_workspace(foreign.id, workspace.id) is None
        assert await policies.get_in_workspace("missing", workspace.id) is None


async def test_add_stages_a_policy_and_refuses_a_second_for_the_same_provider(async_db: AsyncSession) -> None:
    """The refused block rolls back and expires every attached row, so the IDs are held as plain values."""
    acme = await _organization(async_db, slug="acme")
    budget_id = (await _budget(async_db, acme, name="acme")).budget_id
    workspace_id = (await _workspace(async_db, acme, name="acme one")).id
    await async_db.commit()
    uow = UnitOfWork(async_db)

    async with uow:
        policy = await WorkspaceBudgetDefaultRepository(uow).add(
            WorkspaceBudgetDefault(workspace_id=workspace_id, budget_id=budget_id)
        )
        assert policy.id
        assert policy.created_at is not None
    policy_id = policy.id

    with pytest.raises(MemberBudgetPolicyAlreadyExistsError):
        async with uow:
            await WorkspaceBudgetDefaultRepository(uow).add(
                WorkspaceBudgetDefault(workspace_id=workspace_id, budget_id=budget_id)
            )

    async with uow:
        narrowed = await WorkspaceBudgetDefaultRepository(uow).add(
            WorkspaceBudgetDefault(workspace_id=workspace_id, budget_id=budget_id, provider_key_id="pk-a")
        )
        assert narrowed.id != policy_id


async def test_remove_deletes_a_policy(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    workspace = await _workspace(async_db, acme, name="acme one")
    policy = await _policy(async_db, workspace, budget)
    uow = UnitOfWork(async_db)

    async with uow:
        policies = WorkspaceBudgetDefaultRepository(uow)
        await policies.remove(policy)
        assert await policies.for_workspace(workspace.id) == []


async def test_get_many_returns_the_budgets_it_finds_keyed_on_id(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    first = await _budget(async_db, acme, name="first")
    second = await _budget(async_db, acme, name="second")
    uow = UnitOfWork(async_db)

    async with uow:
        budgets = BudgetRepository(uow)
        found = await budgets.get_many([first.budget_id, second.budget_id, "missing"])
        assert set(found) == {first.budget_id, second.budget_id}
        assert found[first.budget_id].name == "first"
        assert await budgets.get_many([]) == {}


async def test_member_ceiling_is_per_membership_and_provider(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    capped = uuid.uuid4()
    narrowed = uuid.uuid4()
    uncapped = uuid.uuid4()
    async_db.add(_member_ceiling(capped, budget.budget_id))
    async_db.add(_member_ceiling(narrowed, budget.budget_id, provider_key_id="pk-a"))
    await async_db.flush()
    uow = UnitOfWork(async_db)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        found = await ceilings.member_ceiling(capped, None)
        assert found is not None
        assert found.scope_id == str(capped)
        assert await ceilings.member_ceiling(capped, "pk-a") is None
        assert await ceilings.member_ceiling(narrowed, "pk-a") is not None
        assert await ceilings.member_ceiling(narrowed, None) is None
        assert await ceilings.member_ceiling(uncapped, None) is None


async def test_members_with_ceiling_answers_only_those_capped_for_that_provider(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    capped = uuid.uuid4()
    narrowed = uuid.uuid4()
    uncapped = uuid.uuid4()
    async_db.add(_member_ceiling(capped, budget.budget_id))
    async_db.add(_member_ceiling(narrowed, budget.budget_id, provider_key_id="pk-a"))
    await async_db.flush()
    members = [capped, narrowed, uncapped]
    uow = UnitOfWork(async_db)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        assert await ceilings.members_with_ceiling(members, None) == {capped}
        assert await ceilings.members_with_ceiling(members, "pk-a") == {narrowed}
        assert await ceilings.members_with_ceiling(members, "pk-b") == set()
        assert await ceilings.members_with_ceiling([], None) == set()


async def test_insert_member_ceilings_skips_a_membership_already_capped(async_db: AsyncSession) -> None:
    """A collision on one membership costs that row, not the batch, and is not reported to the caller."""
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    already = uuid.uuid4()
    fresh = uuid.uuid4()
    async_db.add(_member_ceiling(already, budget.budget_id))
    await async_db.flush()
    uow = UnitOfWork(async_db)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        inserted = await ceilings.insert_member_ceilings(
            [_member_ceiling(already, budget.budget_id), _member_ceiling(fresh, budget.budget_id)]
        )
        assert [ceiling.scope_id for ceiling in inserted] == [str(fresh)]
        assert await ceilings.count_for_budget(budget.budget_id) == 2
        assert await ceilings.insert_member_ceilings([]) == []


async def test_insert_member_ceilings_stages_the_whole_batch_when_nothing_collides(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    members = [uuid.uuid4(), uuid.uuid4()]
    uow = UnitOfWork(async_db)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        inserted = await ceilings.insert_member_ceilings(
            [_member_ceiling(member, budget.budget_id) for member in members]
        )
        assert {ceiling.scope_id for ceiling in inserted} == {str(member) for member in members}
        assert await ceilings.count_for_budget(budget.budget_id) == 2


async def test_delete_for_member_removes_only_that_memberships_ceilings(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    leaver = uuid.uuid4()
    stayer = uuid.uuid4()
    async_db.add(_member_ceiling(leaver, budget.budget_id))
    async_db.add(_member_ceiling(leaver, budget.budget_id, provider_key_id="pk-a"))
    async_db.add(_member_ceiling(stayer, budget.budget_id))
    await async_db.flush()
    uow = UnitOfWork(async_db)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        await ceilings.delete_for_member(leaver)
        assert await ceilings.members_with_ceiling([leaver, stayer], None) == {stayer}
        assert await ceilings.member_ceiling(leaver, "pk-a") is None
        assert await ceilings.count_for_budget(budget.budget_id) == 1


async def test_delete_for_workspace_removes_its_own_and_its_memberships_ceilings(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    doomed = uuid.uuid4()
    surviving = uuid.uuid4()
    member = uuid.uuid4()
    other_member = uuid.uuid4()
    await _ceiling(async_db, budget, scope_type=SCOPE_WORKSPACE, scope_id=str(doomed))
    await _ceiling(async_db, budget, scope_type=SCOPE_WORKSPACE, scope_id=str(surviving))
    await _ceiling(async_db, budget, scope_type=SCOPE_ORGANIZATION, scope_id=str(doomed))
    async_db.add(_member_ceiling(member, budget.budget_id))
    async_db.add(_member_ceiling(other_member, budget.budget_id))
    await async_db.flush()
    uow = UnitOfWork(async_db)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        await ceilings.delete_for_workspace(doomed, [member])
        assert not await ceilings.has_ceiling(SCOPE_WORKSPACE, str(doomed), None)
        assert await ceilings.has_ceiling(SCOPE_WORKSPACE, str(surviving), None)
        assert await ceilings.has_ceiling(SCOPE_ORGANIZATION, str(doomed), None)
        assert await ceilings.members_with_ceiling([member, other_member], None) == {other_member}


async def test_delete_for_workspace_removes_the_workspace_ceiling_with_no_members_named(
    async_db: AsyncSession,
) -> None:
    acme = await _organization(async_db, slug="acme")
    budget = await _budget(async_db, acme, name="acme")
    doomed = uuid.uuid4()
    await _ceiling(async_db, budget, scope_type=SCOPE_WORKSPACE, scope_id=str(doomed))
    uow = UnitOfWork(async_db)

    async with uow:
        ceilings = ScopedBudgetRepository(uow)
        await ceilings.delete_for_workspace(doomed, [])
        assert not await ceilings.has_ceiling(SCOPE_WORKSPACE, str(doomed), None)


async def test_on_builds_every_repository_on_the_unit_of_work(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    uow = UnitOfWork(async_db)
    repositories = BudgetRepositories.on(uow)

    with pytest.raises(OutsideUnitOfWorkError):
        await repositories.budgets.count_by_organization(acme.id)
    with pytest.raises(OutsideUnitOfWorkError):
        await repositories.ceilings.count_for_budget("any")
    with pytest.raises(OutsideUnitOfWorkError):
        await repositories.member_policies.for_workspace(uuid.uuid4())

    async with uow:
        assert await repositories.budgets.count_by_organization(acme.id) == 0
        assert await repositories.ceilings.count_for_budget("any") == 0
        assert await repositories.member_policies.for_workspace(uuid.uuid4()) == []


async def test_insert_member_ceilings_raises_when_a_ceiling_names_no_budget(async_db: AsyncSession) -> None:
    """Only a membership already capped is skipped; any other refusal is the caller's to see."""
    acme = await _organization(async_db, slug="acme")
    budget_id = (await _budget(async_db, acme, name="acme")).budget_id
    await async_db.commit()
    uow = UnitOfWork(async_db)

    with pytest.raises(IntegrityError):
        async with uow:
            await ScopedBudgetRepository(uow).insert_member_ceilings(
                [_member_ceiling(uuid.uuid4(), budget_id), _member_ceiling(uuid.uuid4(), "no-such-budget")]
            )

    async with uow:
        assert await ScopedBudgetRepository(uow).count_for_budget(budget_id) == 0


async def test_add_leaves_the_step_usable_after_it_refuses_a_duplicate(async_db: AsyncSession) -> None:
    """The refusal rolls back its own savepoint only, so the caller's step carries on and commits."""
    acme = await _organization(async_db, slug="acme")
    budget_id = (await _budget(async_db, acme, name="acme")).budget_id
    workspace_id = (await _workspace(async_db, acme, name="acme one")).id
    await async_db.commit()
    uow = UnitOfWork(async_db)

    async with uow:
        policies = WorkspaceBudgetDefaultRepository(uow)
        await policies.add(WorkspaceBudgetDefault(workspace_id=workspace_id, budget_id=budget_id))
        with pytest.raises(MemberBudgetPolicyAlreadyExistsError):
            await policies.add(WorkspaceBudgetDefault(workspace_id=workspace_id, budget_id=budget_id))
        await policies.add(
            WorkspaceBudgetDefault(workspace_id=workspace_id, budget_id=budget_id, provider_key_id="pk-a")
        )
        assert len(await policies.for_workspace(workspace_id)) == 2

    async with uow:
        assert len(await WorkspaceBudgetDefaultRepository(uow).for_workspace(workspace_id)) == 2


async def test_add_raises_when_a_policy_names_no_budget(async_db: AsyncSession) -> None:
    """Only a policy already in place is reported as a duplicate; any other refusal is the caller's to see."""
    acme = await _organization(async_db, slug="acme")
    workspace_id = (await _workspace(async_db, acme, name="acme one")).id
    await async_db.commit()
    uow = UnitOfWork(async_db)

    with pytest.raises(IntegrityError):
        async with uow:
            await WorkspaceBudgetDefaultRepository(uow).add(
                WorkspaceBudgetDefault(workspace_id=workspace_id, budget_id="no-such-budget")
            )

    async with uow:
        assert await WorkspaceBudgetDefaultRepository(uow).for_workspace(workspace_id) == []
