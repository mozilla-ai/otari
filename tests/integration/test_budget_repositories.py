"""The budget repositories answer the organization surface's queries on a Unit of Work."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import OutsideUnitOfWorkError, UnitOfWork
from gateway.exceptions.budget_exceptions import BudgetStillReferencedError, SpendCeilingAlreadyExistsError
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
from gateway.models.tenancy import Organization, User
from gateway.models.users import User as ApiUser
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.repositories.budgets import BudgetRepositories, BudgetRepository, ScopedBudgetRepository, ScopeIdSets
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
        assert await budgets.count_member_policies_for_budget(held.budget_id) == 1
        assert await budgets.count_member_policies_for_budget(free.budget_id) == 0
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



async def test_on_builds_every_repository_on_the_unit_of_work(async_db: AsyncSession) -> None:
    acme = await _organization(async_db, slug="acme")
    uow = UnitOfWork(async_db)
    repositories = BudgetRepositories.on(uow)

    with pytest.raises(OutsideUnitOfWorkError):
        await repositories.budgets.count_by_organization(acme.id)
    with pytest.raises(OutsideUnitOfWorkError):
        await repositories.ceilings.count_for_budget("any")

    async with uow:
        assert await repositories.budgets.count_by_organization(acme.id) == 0
        assert await repositories.ceilings.count_for_budget("any") == 0
