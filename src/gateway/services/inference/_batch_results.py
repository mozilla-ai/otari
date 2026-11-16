"""Account completed batch results as one transactional business step."""

from decimal import Decimal
from typing import Protocol

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.usage import UsageLog
from gateway.services.budgets import BudgetService
from gateway.services.inference._deployment_funding import DeploymentCharge


class _BatchUsageWriter(Protocol):
    async def put(self, log: UsageLog) -> bool: ...

    async def after_commit(self, logs: list[UsageLog]) -> None: ...


class _BatchClaim(Protocol):
    async def claim_accounting(self, batch_id: str) -> bool: ...


class _UsageLogRepository(Protocol):
    async def add_many(self, rows: list[UsageLog]) -> None: ...


class BatchResultsService:
    """Claim, write, settle and publish completed batch usage."""

    def __init__(
        self,
        uow: UnitOfWork,
        batches: _BatchClaim,
        usage_logs: _UsageLogRepository,
        budgets: BudgetService,
    ) -> None:
        self._uow = uow
        self._batches = batches
        self._usage_logs = usage_logs
        self._budgets = budgets

    async def account(
        self,
        *,
        batch_id: str,
        recorded: bool,
        logs: list[UsageLog],
        writer: _BatchUsageWriter,
        cost: Decimal | None,
        total_tokens: int,
        user_id: str | None,
        batch_exempt: bool,
        charge: DeploymentCharge | None,
        model: str,
    ) -> bool:
        """Account a batch once, keeping claim, rows and settlement in one UoW block."""
        if not recorded:
            for log in logs:
                await writer.put(log)
            return bool(logs)

        async with self._uow:
            if not await self._batches.claim_accounting(batch_id):
                return False
            await self._usage_logs.add_many(logs)
            if charge is not None:
                await charge.settle(
                    cost,
                    tools={},
                    model=model,
                    description=f"batch:{model}",
                    expects_cost=bool(total_tokens),
                    savepoint=self._uow.savepoint,
                )
            if cost and user_id and not batch_exempt:
                await self._budgets.record_external_spend(user_id, cost)

        await writer.after_commit(logs)
        return True
