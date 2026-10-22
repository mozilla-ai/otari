"""The deployment operator's use cases over the deployment's own budgets."""

from gateway.exceptions.budget_exceptions import (
    BudgetStillReferencedError,
    DeploymentBudgetEnforcedError,
    DeploymentBudgetIsMemberDefaultError,
    DeploymentBudgetNotFoundError,
    DeploymentBudgetNotReplaceableError,
    DeploymentBudgetOwnedByOrganizationError,
    DeploymentBudgetStillReferencedError,
)
from gateway.models.money import to_usd_or_none
from gateway.repositories.budgets import BudgetRepositories
from gateway.schemas.budgets import BudgetResponse, CreateBudgetRequest
from gateway.services.budgets._organization_surface import _current_window, _require_valid_cycle
from gateway.services.budgets._periods import CYCLE_FIELD_ORDER, CycleSettings
from gateway.services.budgets._retiming import cadence_of


class _DeploymentSurface:
    """Replace or delete a budget the deployment owns."""

    def __init__(self, repositories: BudgetRepositories) -> None:
        self._repositories = repositories

    async def delete_budget(self, budget_id: str) -> None:
        """Delete a deployment budget with its reset history, refusing while a workspace or ceiling names it.

        The workspace-default and ceiling foreign keys are ``RESTRICT``, so the database would refuse either anyway,
        but as an error naming nothing to change. Gateway users assigned the budget are left uncapped.
        """
        budget = await self._repositories.budgets.get(budget_id)
        if budget is None:
            raise DeploymentBudgetNotFoundError(budget_id)
        if budget.organization_id is not None:
            raise DeploymentBudgetOwnedByOrganizationError()
        if workspaces := await self._repositories.member_policies.workspace_names_for_budget(budget_id):
            raise DeploymentBudgetIsMemberDefaultError(workspaces)
        if ceilings := await self._repositories.ceilings.count_for_budget(budget_id):
            raise DeploymentBudgetEnforcedError(ceilings)
        await self._repositories.budgets.remove_reset_logs(budget_id)
        try:
            await self._repositories.budgets.remove(budget)
        except BudgetStillReferencedError:
            raise DeploymentBudgetStillReferencedError(budget_id) from None

    async def put_budget(self, budget_id: str, request: CreateBudgetRequest) -> tuple[BudgetResponse, bool]:
        """Create the deployment budget ``budget_id`` or replace it, saying whether it was created.

        Every field takes the request's value, so a field left out is cleared, and a budget a concurrent request
        created first is replaced like any other. A replaced budget's ceilings follow a change of reset period.
        """
        _require_valid_cycle(CycleSettings(*(getattr(request, name) for name in CYCLE_FIELD_ORDER)))
        budgets = self._repositories.budgets
        budget = await budgets.get(budget_id)
        created = False
        if budget is None:
            created = await budgets.add_if_absent(budget_id)
            budget = await budgets.get(budget_id)
            if budget is None:
                raise RuntimeError("A budget's insert conflicted with a row that is not there")
        if budget.organization_id is not None:
            raise DeploymentBudgetNotReplaceableError(budget_id)

        cadence_before = cadence_of(budget)
        changes = request.model_dump()
        changes["max_budget"] = to_usd_or_none(request.max_budget)
        budget = await budgets.update(budget, changes)
        if created:
            return BudgetResponse.from_model(budget), True
        if cadence_of(budget) != cadence_before:
            period_start, period_end = _current_window(budget)
            await self._repositories.ceilings.retime_for_budget(
                budget_id, period_start=period_start, period_end=period_end
            )
        user_count, total_spend, total_reserved = await budgets.usage(budget_id)
        return BudgetResponse.from_model(
            budget, user_count=user_count, total_spend=total_spend, total_reserved=total_reserved
        ), False
