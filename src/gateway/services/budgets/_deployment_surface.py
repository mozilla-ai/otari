"""The deployment operator's use cases over the deployment's own budgets."""

from gateway.exceptions.budget_exceptions import (
    BudgetStillReferencedError,
    DeploymentBudgetEnforcedError,
    DeploymentBudgetIsMemberDefaultError,
    DeploymentBudgetNotFoundError,
    DeploymentBudgetOwnedByOrganizationError,
    DeploymentBudgetStillReferencedError,
)
from gateway.repositories.budgets import BudgetRepositories


class _DeploymentSurface:
    """Delete a budget the deployment owns."""

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
