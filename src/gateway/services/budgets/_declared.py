"""Apply the budgets config.yml declares, and the ceilings its keys name."""

from gateway.exceptions.budget_exceptions import EndUserBudgetNotFoundError
from gateway.models.budgets import SCOPE_API_TOKEN
from gateway.repositories.budgets import BudgetRepositories
from gateway.services.budgets._organization_surface import _current_window


async def attach_key_ceiling(repositories: BudgetRepositories, key_id: str, budget_id: str) -> None:
    """Cap an API key as a whole at ``budget_id``, through its scoped budget across every provider.

    A key with no such ceiling gets one, with its window opening now. A ceiling on
    another budget is pointed at this one, restarting its window and keeping the
    spend it has recorded, as repointing through the API does. A ceiling already
    on this budget is left alone, so a restart never resets a running period.
    """
    budget = await repositories.budgets.get(budget_id)
    if budget is None or budget.organization_id is not None:
        # The same answer the end-user budgets get: a tenant's budget is not one a deployment key is capped at.
        raise EndUserBudgetNotFoundError(budget_id)
    ceiling = await repositories.ceilings.scope_ceiling(SCOPE_API_TOKEN, key_id)
    if ceiling is None:
        period_start, period_end = _current_window(budget)
        if await repositories.ceilings.add_scope_ceiling_if_absent(
            scope_type=SCOPE_API_TOKEN,
            scope_id=key_id,
            budget_id=budget_id,
            period_start=period_start,
            period_end=period_end,
        ):
            return
        ceiling = await repositories.ceilings.scope_ceiling(SCOPE_API_TOKEN, key_id)
        if ceiling is None:
            raise RuntimeError("A ceiling's insert conflicted with a row that is not there")
    if ceiling.budget_id != budget_id:
        period_start, period_end = _current_window(budget)
        await repositories.ceilings.update(
            ceiling, {"budget_id": budget_id, "period_start": period_start, "period_end": period_end}
        )
