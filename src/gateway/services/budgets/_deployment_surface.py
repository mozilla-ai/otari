"""Create deployment-managed ceilings inside the caller's Unit of Work."""

from datetime import UTC, datetime

from gateway.exceptions.budget_exceptions import (
    DeploymentBudgetNotFoundError,
    DeploymentScopedBudgetAlreadyExistsError,
    DeploymentScopeNotFoundError,
    SpendCeilingAlreadyExistsError,
)
from gateway.models.budgets import ScopedBudget, ScopeType
from gateway.repositories.budgets import BudgetRepositories
from gateway.schemas.budgets import CreateScopedBudgetRequest, ScopedBudgetResponse
from gateway.services.budgets._periods import period_window
from gateway.services.budgets._scopes import ScopeOwnership

_SCOPE_SUBJECTS: dict[ScopeType, str] = {
    "organization": "Organization",
    "workspace": "Workspace",
    "workspace_member": "Workspace membership",
    "org_member": "Organization membership",
    "api_token": "API key",
}


class _DeploymentSurface:
    """The ceiling creation step of the deployment-operator surface."""

    def __init__(self, repositories: BudgetRepositories, scopes: ScopeOwnership) -> None:
        self._repositories = repositories
        self._scopes = scopes

    async def create_ceiling(self, request: CreateScopedBudgetRequest) -> ScopedBudgetResponse:
        # Lock before checking existence, and hold through the insert and commit.
        await self._scopes.lock_for_ceiling(request.scope_type, request.scope_id)
        if await self._scopes.get_organization_id_for(request.scope_type, request.scope_id) is None:
            raise DeploymentScopeNotFoundError(_SCOPE_SUBJECTS[request.scope_type], request.scope_id)
        budget = await self._repositories.budgets.get(request.budget_id)
        if budget is None:
            raise DeploymentBudgetNotFoundError(request.budget_id)
        window = period_window(datetime.now(UTC), duration=budget.budget_duration_sec, alignment=budget.reset_alignment)
        period_start, period_end = window if window is not None else (None, None)
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
            raise DeploymentScopedBudgetAlreadyExistsError from None
        return ScopedBudgetResponse.from_model(ceiling, budget)
