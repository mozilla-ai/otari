"""An organization's own spend budgets and the ceilings that enforce them.

A budget is the figure and the period it is spent over, owned by the organization named on it.
A ceiling names a budget and caps one identity inside the organization at that figure.
Three rules hold across every use case here.
The caller's organization comes from their identity, never from the request.
Only a management role may read or write, because a cap says what colleagues may spend.
Every scope must resolve into the caller's organization and every budget must be owned by it,
and both refusals are 404, so another tenant's row cannot be told from one that was never created.
A budget with no organization belongs to the deployment: nothing here lists, offers or repoints one,
and a ceiling that names one is still listed, with ``manageable`` false, because it caps this organization's spend.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from gateway.exceptions import TenancyValidationError
from gateway.exceptions.budget_exceptions import (
    BudgetStillReferencedError,
    OrganizationBudgetHeldElsewhereError,
    OrganizationBudgetInUseError,
    OrganizationBudgetNotFoundError,
    OrganizationScopedBudgetAlreadyExistsError,
    OrganizationScopedBudgetNotFoundError,
    OrganizationScopeNotFoundError,
    SpendCeilingAlreadyExistsError,
)
from gateway.models.budgets import (
    SCOPE_ORGANIZATION,
    SCOPE_TYPES,
    SCOPE_WORKSPACE,
    SCOPE_WORKSPACE_MEMBER,
    Budget,
    ScopedBudget,
)
from gateway.models.money import to_usd_or_none
from gateway.models.tenancy import Organization, User
from gateway.repositories.budgets import BudgetRepositories
from gateway.schemas.budgets import (
    AppliedEntity,
    AppliedEntityPublic,
    OrganizationBudgetCreate,
    OrganizationBudgetPublic,
    OrganizationBudgetsPublic,
    OrganizationBudgetUpdate,
    OrganizationScopedBudgetCreate,
    OrganizationScopedBudgetPublic,
    OrganizationScopedBudgetsPublic,
    OrganizationScopedBudgetUpdate,
)
from gateway.services.budgets._periods import (
    CYCLE_FIELD_ORDER,
    CycleSettings,
    budget_window,
    cycle_settings_of,
    settle_cycle,
    validate_cycle_settings,
)
from gateway.services.budgets._retiming import cadence_of
from gateway.services.budgets._scopes import ScopeOwnership, lock_workspace_for_scope, uuid_or_none
from gateway.services.tenancy.organization_service import OrganizationService

_MAX_LIST_LIMIT = 1000


def _require_valid_cycle(settings: CycleSettings) -> None:
    """Refuse a cadence carrying the wrong settings for its cycle, naming the field."""
    try:
        validate_cycle_settings(settings)
    except ValueError as error:
        raise TenancyValidationError(str(error)) from error


def _current_window(budget: Budget) -> tuple[datetime | None, datetime | None]:
    """Return the window a ceiling on this budget occupies now.

    An aligned budget opens on the current calendar boundary, so the first period is a partial one.
    """
    window = budget_window(datetime.now(UTC), budget)
    return window if window is not None else (None, None)


class _OrganizationSurface:
    """The organization's budget and ceiling use cases, each run inside the caller's Unit of Work block."""

    def __init__(
        self,
        repositories: BudgetRepositories,
        scopes: ScopeOwnership,
        organizations: OrganizationService,
    ) -> None:
        self._repositories = repositories
        self._scopes = scopes
        self._organizations = organizations

    async def _get_managed_organization(self, user: User) -> Organization:
        """Return the caller's organization once they are proven to manage its spend."""
        organization = await self._organizations.get_active_organization_for_user(user)
        await self._organizations.require_active_organization_management_access(user=user, organization=organization)
        return organization

    async def _applied_to(
        self, organization: Organization, budget_ids: Sequence[str]
    ) -> dict[str, list[AppliedEntityPublic]]:
        """Return the entities inside this organization that each of these budgets applies to.

        An operator can point another organization's scope at this organization's budget,
        so the ceilings are filtered to this organization's scopes and that ID never leaves it.
        """
        scopes = await self._scopes.get_scope_ids_in(organization.id)
        ceilings = await self._repositories.ceilings.list_for_budgets_in_scopes(budget_ids, scopes)
        workspace_ids = {
            workspace_id
            for ceiling in ceilings
            if ceiling.scope_type == SCOPE_WORKSPACE and (workspace_id := uuid_or_none(ceiling.scope_id))
        }
        names = await self._organizations.get_workspace_names_in_organization(organization.id, workspace_ids)
        names[organization.id] = organization.name

        def name_of(ceiling: ScopedBudget) -> str | None:
            scope_id = uuid_or_none(ceiling.scope_id)
            if ceiling.scope_type in (SCOPE_ORGANIZATION, SCOPE_WORKSPACE) and scope_id:
                return names.get(scope_id)
            return None

        applied: dict[str, list[AppliedEntityPublic]] = {budget_id: [] for budget_id in budget_ids}
        for ceiling in ceilings:
            applied[ceiling.budget_id].append(
                AppliedEntityPublic(
                    scope_type=ceiling.scope_type,
                    scope_id=ceiling.scope_id,
                    provider_key_id=ceiling.provider_key_id,
                    model=ceiling.model,
                    name=name_of(ceiling),
                )
            )
        return applied

    async def _require_no_existing_ceiling(self, request: OrganizationScopedBudgetCreate) -> None:
        if await self._repositories.ceilings.has_ceiling(
            request.scope_type, request.scope_id, request.provider_key_id, request.model
        ):
            raise OrganizationScopedBudgetAlreadyExistsError(request.scope_type, request.scope_id)

    async def _require_own_budget(self, *, organization: Organization, budget_id: str) -> Budget:
        """Return a budget this organization owns, or refuse it as not found."""
        budget = await self._repositories.budgets.get_by_id_and_organization(budget_id, organization.id)
        if budget is None:
            raise OrganizationBudgetNotFoundError(budget_id)
        return budget

    async def _require_own_ceiling(self, *, organization: Organization, ceiling_id: str) -> ScopedBudget:
        """Return a ceiling whose scope sits in this organization, or refuse it as not found.

        Ownership is resolved through the scope, not the budget,
        so a ceiling on a deployment budget is still this organization's.
        """
        ceiling = await self._repositories.ceilings.get(ceiling_id)
        if ceiling is None:
            raise OrganizationScopedBudgetNotFoundError(ceiling_id)
        # A stored scope type this build does not know resolves to no owner, which refuses rather than leaks.
        if ceiling.scope_type not in SCOPE_TYPES:
            raise OrganizationScopedBudgetNotFoundError(ceiling_id)
        owner = await self._scopes.get_organization_id_for(ceiling.scope_type, ceiling.scope_id)
        if owner != organization.id:
            raise OrganizationScopedBudgetNotFoundError(ceiling_id)
        return ceiling

    async def _require_scope_in_organization(
        self,
        *,
        organization: Organization,
        scope_type: str,
        scope_id: str,
    ) -> None:
        """Refuse a scope that resolves to nothing or into another organization, with one not-found answer for both."""
        if scope_type not in SCOPE_TYPES:
            raise TenancyValidationError(f"Unknown scope type: {scope_type}")
        owner = await self._scopes.get_organization_id_for(scope_type, scope_id)
        if owner != organization.id:
            raise OrganizationScopeNotFoundError(scope_type, scope_id)

    async def _replace_entities(
        self, *, organization: Organization, budget: Budget, entities: list[AppliedEntity]
    ) -> int:
        """Make these entities exactly the ones this budget applies to, and return how many that is.

        An entity the budget already applied to keeps its ceiling, and with it the spend this window has recorded:
        dropping and re-adding it would hand back a budget already spent.
        Runs in the caller's block, so a refusal part-way through rolls back the budget write as well.
        A fixed number of queries however many entities there are: the workspaces are locked in one ordered
        statement before anything is read or written, so two saves cannot deadlock each other or a workspace
        deletion, and ownership is checked against the organization's scope IDs in memory.
        """
        scopes = await self._scopes.get_scope_ids_in(organization.id)
        held = {
            (ceiling.scope_type, ceiling.scope_id, ceiling.provider_key_id, ceiling.model): ceiling
            for ceiling in await self._repositories.ceilings.list_for_budgets_in_scopes([budget.budget_id], scopes)
        }
        wanted = {entity.key() for entity in entities}
        added = [entity for entity in entities if entity.key() not in held]
        removed = [ceiling for key, ceiling in held.items() if key not in wanted]
        if not added and not removed:
            return len(wanted)

        touched: list[tuple[str, str]] = [(entity.scope_type, entity.scope_id) for entity in added]
        touched += [(ceiling.scope_type, ceiling.scope_id) for ceiling in removed]
        await self._organizations.lock_workspaces(
            {
                workspace_id
                for kind, scope_id in touched
                if kind == SCOPE_WORKSPACE and (workspace_id := uuid_or_none(scope_id))
            },
            {
                member_id
                for kind, scope_id in touched
                if kind == SCOPE_WORKSPACE_MEMBER and (member_id := uuid_or_none(scope_id))
            },
        )
        # Re-read under the locks, so a workspace deleted while this waited is not still offered.
        scopes = await self._scopes.get_scope_ids_in(organization.id)
        # Membership in the organization's own ID strings, which is also what refuses a
        # non-canonical spelling of a UUID: it would store a ceiling nothing matches.
        for entity in added:
            if entity.scope_id not in scopes.ids_of(entity.scope_type):
                raise OrganizationScopeNotFoundError(entity.scope_type, entity.scope_id)
        taken = {
            (ceiling.scope_type, ceiling.scope_id, ceiling.provider_key_id, ceiling.model)
            for ceiling in await self._repositories.ceilings.list_on_scope_ids([entity.scope_id for entity in added])
            if ceiling.budget_id != budget.budget_id
        }
        for entity in added:
            if entity.key() in taken:
                raise OrganizationScopedBudgetAlreadyExistsError(entity.scope_type, entity.scope_id)

        await self._repositories.ceilings.remove_many([ceiling.id for ceiling in removed])
        period_start, period_end = _current_window(budget)
        try:
            await self._repositories.ceilings.add_many(
                [
                    ScopedBudget(
                        scope_type=entity.scope_type,
                        scope_id=entity.scope_id,
                        provider_key_id=entity.provider_key_id,
                        model=entity.model,
                        budget_id=budget.budget_id,
                        period_start=period_start,
                        period_end=period_end,
                    )
                    for entity in added
                ]
            )
        except SpendCeilingAlreadyExistsError:
            raise OrganizationScopedBudgetAlreadyExistsError(added[0].scope_type, added[0].scope_id) from None
        return len(wanted)

    async def create_budget(self, *, user: User, request: OrganizationBudgetCreate) -> OrganizationBudgetPublic:
        organization = await self._get_managed_organization(user)
        _require_valid_cycle(CycleSettings(*(getattr(request, name) for name in CYCLE_FIELD_ORDER)))
        budget = await self._repositories.budgets.add(
            Budget(
                organization_id=organization.id,
                name=request.name,
                max_budget=to_usd_or_none(request.max_budget),
                token_limit=request.token_limit,
                request_limit=request.request_limit,
                reset_cycle=request.reset_cycle,
                reset_every_n=request.reset_every_n,
                reset_anchor_at=request.reset_anchor_at,
                reset_weekdays=request.reset_weekdays,
                reset_month_day=request.reset_month_day,
                reset_month=request.reset_month,
            )
        )
        if not request.applied_to:
            return OrganizationBudgetPublic.from_model(
                budget, organization_id=organization.id, ceiling_count=0, applied_to=[]
            )
        applied = await self._replace_entities(organization=organization, budget=budget, entities=request.applied_to)
        return OrganizationBudgetPublic.from_model(
            budget,
            organization_id=organization.id,
            ceiling_count=applied,
            applied_to=(await self._applied_to(organization, [budget.budget_id]))[budget.budget_id],
        )

    async def create_ceiling(
        self,
        *,
        user: User,
        request: OrganizationScopedBudgetCreate,
    ) -> OrganizationScopedBudgetPublic:
        organization = await self._get_managed_organization(user)
        # Checked here as well as below, because the lock between them answers an
        # unknown scope type with an assertion rather than a validation error.
        if request.scope_type not in SCOPE_TYPES:
            raise TenancyValidationError(f"Unknown scope type: {request.scope_type}")
        # The lock precedes the check, so a concurrent workspace deletion cannot
        # commit between the check and the insert.
        await lock_workspace_for_scope(self._organizations, request.scope_type, request.scope_id)
        await self._require_scope_in_organization(
            organization=organization,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
        )
        budget = await self._require_own_budget(organization=organization, budget_id=request.budget_id)
        # The pre-check names the clash, and the unique index closes the race it leaves with the same 409.
        await self._require_no_existing_ceiling(request)
        period_start, period_end = _current_window(budget)
        try:
            ceiling = await self._repositories.ceilings.add(
                ScopedBudget(
                    scope_type=request.scope_type,
                    scope_id=request.scope_id,
                    provider_key_id=request.provider_key_id,
                    model=request.model,
                    budget_id=budget.budget_id,
                    name=request.name,
                    period_start=period_start,
                    period_end=period_end,
                )
            )
        except SpendCeilingAlreadyExistsError:
            raise OrganizationScopedBudgetAlreadyExistsError(request.scope_type, request.scope_id) from None
        return OrganizationScopedBudgetPublic.from_model(ceiling, budget, organization_id=organization.id)

    async def delete_budget(self, *, user: User, budget_id: str) -> None:
        """Delete a budget of the organization's, refusing while anything names it.

        Ceilings and member policies are counted so the refusal can say which.
        A gateway user's assignment is counted but not named, because the admin cannot act on gateway users,
        and without the count the ORM would null the assignment out silently.
        The budget's reset history goes with it, as it does on the deployment's delete.
        """
        organization = await self._get_managed_organization(user)
        budget = await self._require_own_budget(organization=organization, budget_id=budget_id)
        ceilings = await self._repositories.ceilings.count_for_budget(budget.budget_id)
        defaults = await self._repositories.member_policies.count_for_budget(budget.budget_id)
        if ceilings or defaults:
            raise OrganizationBudgetInUseError(budget.budget_id, ceilings=ceilings, defaults=defaults)
        if await self._repositories.budgets.count_users_for_budget(budget.budget_id):
            raise OrganizationBudgetHeldElsewhereError(budget.budget_id)
        await self._repositories.budgets.remove_reset_logs(budget.budget_id)
        try:
            await self._repositories.budgets.remove(budget)
        except BudgetStillReferencedError:
            raise OrganizationBudgetHeldElsewhereError(budget_id) from None

    async def delete_ceiling(self, *, user: User, ceiling_id: str) -> None:
        """Remove a ceiling. A reservation still held against it settles into nothing."""
        organization = await self._get_managed_organization(user)
        ceiling = await self._require_own_ceiling(organization=organization, ceiling_id=ceiling_id)
        await self._repositories.ceilings.remove(ceiling)

    async def list_budgets(self, *, user: User, skip: int, limit: int) -> OrganizationBudgetsPublic:
        organization = await self._get_managed_organization(user)
        limit = min(limit, _MAX_LIST_LIMIT)
        count = await self._repositories.budgets.count_by_organization(organization.id)
        budgets = await self._repositories.budgets.list_by_organization(organization.id, skip=skip, limit=limit)
        budget_ids = [budget.budget_id for budget in budgets]
        held = await self._repositories.ceilings.count_for_budgets(budget_ids)
        applied = await self._applied_to(organization, budget_ids)
        return OrganizationBudgetsPublic(
            data=[
                OrganizationBudgetPublic.from_model(
                    budget,
                    organization_id=organization.id,
                    ceiling_count=held.get(budget.budget_id, 0),
                    applied_to=applied[budget.budget_id],
                )
                for budget in budgets
            ],
            count=count,
        )

    async def list_ceilings(self, *, user: User, skip: int, limit: int) -> OrganizationScopedBudgetsPublic:
        organization = await self._get_managed_organization(user)
        limit = min(limit, _MAX_LIST_LIMIT)
        scopes = await self._scopes.get_scope_ids_in(organization.id)
        count = await self._repositories.ceilings.count_in_scopes(scopes)
        rows = await self._repositories.ceilings.list_in_scopes(scopes, skip=skip, limit=limit)
        return OrganizationScopedBudgetsPublic(
            data=[
                OrganizationScopedBudgetPublic.from_model(ceiling, budget, organization_id=organization.id)
                for ceiling, budget in rows
            ],
            count=count,
        )

    async def update_budget(
        self,
        *,
        user: User,
        budget_id: str,
        request: OrganizationBudgetUpdate,
    ) -> OrganizationBudgetPublic:
        """Change a budget of the organization's, retiming every ceiling naming it when its cadence changes.

        Retiming is keyed on the cadence, so a rename or a new figure does not restart a window part-way through.
        The counters are not zeroed: spend already recorded stays,
        and a hold taken before the change is released against the same counter.
        """
        organization = await self._get_managed_organization(user)
        budget = await self._require_own_budget(organization=organization, budget_id=budget_id)
        cadence_before = cadence_of(budget)
        changes: dict[str, Any] = request.model_dump(exclude_unset=True, exclude={"applied_to"})
        if "max_budget" in changes:
            changes["max_budget"] = to_usd_or_none(changes["max_budget"])
        # The resulting set is what the CHECK constraints refuse, and no submitted
        # field alone looks wrong: a cycle change that leaves the previous cycle's
        # settings behind is two valid-looking fields and an impossible row.
        settled = settle_cycle(
            cycle_settings_of(budget),
            CycleSettings(*(changes.get(name) for name in CYCLE_FIELD_ORDER)),
            changes.keys() & set(CYCLE_FIELD_ORDER),
        )
        _require_valid_cycle(settled)
        changes.update(dict(zip(CYCLE_FIELD_ORDER, settled, strict=True)))
        budget = await self._repositories.budgets.update(budget, changes)
        if cadence_of(budget) != cadence_before:
            period_start, period_end = _current_window(budget)
            await self._repositories.ceilings.retime_for_budget(
                budget.budget_id, period_start=period_start, period_end=period_end
            )
        if request.applied_to is not None:
            await self._replace_entities(organization=organization, budget=budget, entities=request.applied_to)
        return OrganizationBudgetPublic.from_model(
            budget,
            organization_id=organization.id,
            ceiling_count=await self._repositories.ceilings.count_for_budget(budget.budget_id),
            applied_to=(await self._applied_to(organization, [budget.budget_id]))[budget.budget_id],
        )

    async def update_ceiling(
        self,
        *,
        user: User,
        ceiling_id: str,
        request: OrganizationScopedBudgetUpdate,
    ) -> OrganizationScopedBudgetPublic:
        """Relabel a ceiling, or point it at a budget the organization owns.

        Repointing restarts the window from now and keeps the spend already recorded.
        A ceiling naming a deployment budget may be moved onto one of the organization's own,
        which is how it becomes manageable.
        """
        organization = await self._get_managed_organization(user)
        ceiling = await self._require_own_ceiling(organization=organization, ceiling_id=ceiling_id)
        budget = await self._repositories.budgets.get(ceiling.budget_id)
        if budget is None:
            raise OrganizationScopedBudgetNotFoundError(ceiling_id)
        changes: dict[str, Any] = {}
        if "name" in request.model_fields_set:
            changes["name"] = request.name
        if request.budget_id is not None and request.budget_id != ceiling.budget_id:
            budget = await self._require_own_budget(organization=organization, budget_id=request.budget_id)
            changes["budget_id"] = budget.budget_id
            changes["period_start"], changes["period_end"] = _current_window(budget)
        ceiling = await self._repositories.ceilings.update(ceiling, changes)
        return OrganizationScopedBudgetPublic.from_model(ceiling, budget, organization_id=organization.id)
