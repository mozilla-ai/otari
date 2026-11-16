"""A budget's per-minute limits, as the budget service hands them to admission."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.rate_limit import BudgetMinuteLimits
from gateway.services.budgets import BudgetService


def _service(found: tuple[str, int | None, int | None] | None) -> tuple[BudgetService, AsyncMock]:
    repositories = MagicMock()
    repositories.budgets.minute_limits_for_user = AsyncMock(return_value=found)
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    service = BudgetService(uow, repositories, MagicMock(), MagicMock(), MagicMock(), users=MagicMock())
    return service, repositories.budgets.minute_limits_for_user


@pytest.mark.asyncio
async def test_a_budgets_minute_limits_reach_admission() -> None:
    service, lookup = _service(("b", 5, None))

    assert await service.minute_limits("u", strategy="for_update") == BudgetMinuteLimits(budget_id="b", rpm=5, tpm=None)
    lookup.assert_awaited_once_with("u")


@pytest.mark.asyncio
async def test_a_disabled_budget_strategy_applies_no_minute_limits() -> None:
    service, lookup = _service(("b", 5, None))

    assert await service.minute_limits("u", strategy=" Disabled ") is None
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_spend_is_staged_by_the_budget_service() -> None:
    repositories = MagicMock()
    spend = repositories.budgets.record_external_spend = AsyncMock()
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    service = BudgetService(uow, repositories, MagicMock(), MagicMock(), MagicMock(), users=MagicMock())

    await service.record_external_spend("u", Decimal("1.25"))

    spend.assert_awaited_once_with("u", Decimal("1.25"))
    uow.__aenter__.assert_awaited_once()
    uow.__aexit__.assert_awaited_once()
