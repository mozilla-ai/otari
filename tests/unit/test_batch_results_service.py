from dataclasses import dataclass
from decimal import Decimal
from typing import cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.usage import UsageLog
from gateway.services.budgets import BudgetService
from gateway.services.inference import BatchResultsService


def _usage_row() -> UsageLog:
    return UsageLog(id="batch-row", model="test-model", endpoint="/v1/batches/results", status="success")


@dataclass
class _BatchClaim:
    claim_accounting: AsyncMock


def _service() -> tuple[BatchResultsService, AsyncSession, _BatchClaim, AsyncMock, AsyncMock, AsyncMock]:
    session = AsyncMock(spec=AsyncSession)
    uow = UnitOfWork(session)
    claim = _BatchClaim(AsyncMock(return_value=True))
    usage_logs = AsyncMock()
    budgets = AsyncMock(spec=BudgetService)
    writer = AsyncMock()
    return BatchResultsService(uow, claim, usage_logs, budgets), session, claim, usage_logs, budgets, writer


async def _account(service: BatchResultsService, writer: AsyncMock) -> bool:
    accounted = await service.account(
        batch_id="batch-1",
        recorded=True,
        logs=[_usage_row()],
        writer=writer,
        cost=Decimal("1.25"),
        total_tokens=10,
        user_id="user-1",
        batch_exempt=False,
        charge=None,
        model="test-model",
    )
    return bool(accounted)


@pytest.mark.asyncio
async def test_batch_accounting_commits_once_and_notifies_after_commit() -> None:
    service, session, claim, usage_logs, budgets, writer = _service()
    events: list[str] = []
    cast(AsyncMock, usage_logs.add_many).side_effect = lambda _rows: events.append("usage")
    cast(AsyncMock, session.commit).side_effect = lambda: events.append("commit")
    cast(AsyncMock, writer.after_commit).side_effect = lambda _rows: events.append("after_commit")

    assert await _account(service, writer) is True

    claim.claim_accounting.assert_awaited_once_with("batch-1")
    cast(AsyncMock, usage_logs.add_many).assert_awaited_once()
    cast(AsyncMock, budgets.record_external_spend).assert_awaited_once_with("user-1", Decimal("1.25"))
    cast(AsyncMock, session.commit).assert_awaited_once()
    cast(AsyncMock, writer.after_commit).assert_awaited_once()
    assert events == ["usage", "commit", "after_commit"]


@pytest.mark.asyncio
async def test_duplicate_batch_claim_skips_writes_and_after_commit() -> None:
    service, session, claim, usage_logs, budgets, writer = _service()
    claim.claim_accounting.return_value = False

    assert await _account(service, writer) is False

    claim.claim_accounting.assert_awaited_once_with("batch-1")
    cast(AsyncMock, usage_logs.add_many).assert_not_awaited()
    cast(AsyncMock, budgets.record_external_spend).assert_not_awaited()
    cast(AsyncMock, session.commit).assert_awaited_once()
    cast(AsyncMock, writer.after_commit).assert_not_awaited()


@pytest.mark.asyncio
async def test_usage_write_failure_rolls_back_claim_and_skips_settlement() -> None:
    service, session, claim, usage_logs, budgets, writer = _service()
    failure = RuntimeError("usage write failed")
    cast(AsyncMock, usage_logs.add_many).side_effect = failure

    with pytest.raises(RuntimeError, match="usage write failed"):
        await _account(service, writer)

    claim.claim_accounting.assert_awaited_once_with("batch-1")
    cast(AsyncMock, session.commit).assert_not_awaited()
    cast(AsyncMock, session.rollback).assert_awaited_once()
    cast(AsyncMock, budgets.record_external_spend).assert_not_awaited()
    cast(AsyncMock, writer.after_commit).assert_not_awaited()
