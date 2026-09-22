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

from datetime import UTC, datetime
from typing import Any

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
from gateway.models.budgets import SCOPE_TYPES, Budget, ScopedBudget
from gateway.models.money import to_usd_or_none
from gateway.models.tenancy import Organization, User
from gateway.repositories.budgets import BudgetRepositories
from gateway.schemas.budgets import (
    OrganizationBudgetCreate,
    OrganizationBudgetPublic,
    OrganizationBudgetsPublic,
    OrganizationBudgetUpdate,
    OrganizationScopedBudgetCreate,
    OrganizationScopedBudgetPublic,
    OrganizationScopedBudgetsPublic,
    OrganizationScopedBudgetUpdate,
)
from gateway.services.budgets._periods import period_window
from gateway.services.budgets._retiming import cadence_of
from gateway.services.budgets._scopes import ScopeOwnership
from gateway.services.tenancy.errors import TenancyValidationError
from gateway.services.tenancy.organization_service import OrganizationService

_MAX_LIST_LIMIT = 1000


def _require_single_period_source(duration: int | None, alignment: str | None) -> None:
    """Refuse a budget that resets on a duration and on a calendar boundary at once."""
    if duration is not None and alignment is not None:
        raise TenancyValidationError("A budget resets on budget_duration_sec or on reset_alignment, not both")


def _current_window(budget: Budget) -> tuple[datetime | None, datetime | None]:
    """Return the window a ceiling on this budget occupies now.

    An aligned budget opens on the current calendar boundary, so the first period is a partial one.
    """
    window = period_window(datetime.now(UTC), duration=budget.budget_duration_sec, alignment=budget.reset_alignment)
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

    async def _require_no_existing_ceiling(self, request: OrganizationScopedBudgetCreate) -> None:
        if await self._repositories.ceilings.has_ceiling(request.scope_type, request.scope_id, request.provider_key_id):
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

    async def create_budget(self, *, user: User, request: OrganizationBudgetCreate) -> OrganizationBudgetPublic:
        organization = await self._get_managed_organization(user)
        _require_single_period_source(request.budget_duration_sec, request.reset_alignment)
        budget = await self._repositories.budgets.add(
            Budget(
                organization_id=organization.id,
                name=request.name,
                max_budget=to_usd_or_none(request.max_budget),
                token_limit=request.token_limit,
                request_limit=request.request_limit,
                budget_duration_sec=request.budget_duration_sec,
                reset_alignment=request.reset_alignment,
            )
        )
        return OrganizationBudgetPublic.from_model(budget, organization_id=organization.id, ceiling_count=0)

    async def create_ceiling(
        self,
        *,
        user: User,
        request: OrganizationScopedBudgetCreate,
    ) -> OrganizationScopedBudgetPublic:
        organization = await self._get_managed_organization(user)
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
        A reset record refuses at flush time instead, and is reported the same way.
        """
        organization = await self._get_managed_organization(user)
        budget = await self._require_own_budget(organization=organization, budget_id=budget_id)
        ceilings = await self._repositories.ceilings.count_for_budget(budget.budget_id)
        defaults = await self._repositories.budgets.count_member_policies_for_budget(budget.budget_id)
        if ceilings or defaults:
            raise OrganizationBudgetInUseError(budget.budget_id, ceilings=ceilings, defaults=defaults)
        if await self._repositories.budgets.count_users_for_budget(budget.budget_id):
            raise OrganizationBudgetHeldElsewhereError(budget.budget_id)
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
        held = await self._repositories.ceilings.count_for_budgets([budget.budget_id for budget in budgets])
        return OrganizationBudgetsPublic(
            data=[
                OrganizationBudgetPublic.from_model(
                    budget,
                    organization_id=organization.id,
                    ceiling_count=held.get(budget.budget_id, 0),
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
        cadence_before = cadence_of(budget.budget_duration_sec, budget.reset_alignment)
        changes: dict[str, Any] = request.model_dump(exclude_unset=True)
        if "max_budget" in changes:
            changes["max_budget"] = to_usd_or_none(changes["max_budget"])
        # The resulting pair is what the CHECK constraint refuses, and neither submitted field alone looks wrong.
        _require_single_period_source(
            changes.get("budget_duration_sec", budget.budget_duration_sec),
            changes.get("reset_alignment", budget.reset_alignment),
        )
        budget = await self._repositories.budgets.update(budget, changes)
        if cadence_of(budget.budget_duration_sec, budget.reset_alignment) != cadence_before:
            period_start, period_end = _current_window(budget)
            await self._repositories.ceilings.retime_for_budget(
                budget.budget_id, period_start=period_start, period_end=period_end
            )
        return OrganizationBudgetPublic.from_model(
            budget,
            organization_id=organization.id,
            ceiling_count=await self._repositories.ceilings.count_for_budget(budget.budget_id),
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

