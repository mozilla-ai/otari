import uuid
from collections.abc import Sequence
from decimal import Decimal
from typing import Never

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.budget_exceptions import BudgetStillReferencedError
from gateway.models.budgets import Budget, BudgetResetLog
from gateway.models.users import User
from gateway.repositories.base_repository import BaseRepository


class BudgetRepository(BaseRepository[Budget, Never, Never]):
    """Query and stage budgets in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, Budget)

    async def add(self, budget: Budget) -> Budget:
        """Stage a new budget and return it with its generated values."""
        self.db.add(budget)
        await self.db.flush()
        await self.db.refresh(budget)
        return budget

    async def add_if_absent(self, budget: Budget) -> bool:
        """Stage a budget under the id it carries, returning False when a concurrent request created that id first.

        The insert runs in a SAVEPOINT, so losing that race rolls back this row alone.
        """
        try:
            async with self.db.begin_nested():
                self.db.add(budget)
        except IntegrityError:
            return False
        return True

    async def usage(self, budget_id: str) -> tuple[int, float, float]:
        """Count the active users on a budget and sum their spend and reservations."""
        row = (
            await self.db.execute(
                select(
                    func.count(),
                    # Decimal defaults: ``coalesce(numeric, double precision)`` would sum exact counters as floats.
                    func.coalesce(func.sum(User.spend), Decimal(0)),
                    func.coalesce(func.sum(User.reserved), Decimal(0)),
                ).where(User.budget_id == budget_id, User.deleted_at.is_(None))
            )
        ).one()
        return int(row[0]), float(row[1]), float(row[2])

    async def count_by_organization(self, organization_id: uuid.UUID) -> int:
        """Count the organization's budgets."""
        result = await self.db.execute(
            select(func.count()).select_from(Budget).where(Budget.organization_id == organization_id)
        )
        return result.scalar_one()

    async def count_users_for_budget(self, budget_id: str) -> int:
        """Count the gateway users assigned this budget."""
        result = await self.db.execute(select(func.count()).select_from(User).where(User.budget_id == budget_id))
        return result.scalar_one()

    async def minute_limits_for_user(self, user_id: str) -> tuple[str, int | None, int | None] | None:
        """``(budget_id, rpm_limit, tpm_limit)`` of the user's own budget, or None when it limits neither."""
        row = (
            await self.db.execute(
                select(Budget.budget_id, Budget.rpm_limit, Budget.tpm_limit)
                .join(User, User.budget_id == Budget.budget_id)
                .where(User.user_id == user_id, User.deleted_at.is_(None))
            )
        ).first()
        if row is None or (row.rpm_limit is None and row.tpm_limit is None):
            return None
        return row.budget_id, row.rpm_limit, row.tpm_limit

    async def get_by_id_and_organization(self, budget_id: str, organization_id: uuid.UUID) -> Budget | None:
        """Return the budget with this ID when this organization owns it, otherwise None."""
        result = await self.db.execute(
            select(Budget).where(Budget.budget_id == budget_id, Budget.organization_id == organization_id)
        )
        return result.scalar_one_or_none()

    async def get_many(self, budget_ids: Sequence[str]) -> dict[str, Budget]:
        """Return the budgets with these IDs, keyed on ID, omitting an ID that names none."""
        result = await self.db.execute(select(Budget).where(Budget.budget_id.in_(budget_ids)))
        return {budget.budget_id: budget for budget in result.scalars().all()}

    async def list_by_organization(self, organization_id: uuid.UUID, *, skip: int, limit: int) -> list[Budget]:
        """Return a page of the organization's budgets, oldest first."""
        result = await self.db.execute(
            select(Budget)
            .where(Budget.organization_id == organization_id)
            .order_by(Budget.created_at, Budget.budget_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def remove_reset_logs(self, budget_id: str) -> None:
        """Stage the deletion of the budget's reset history.

        ``budget_reset_logs.budget_id`` is NOT NULL with no ``ondelete``, so a budget that has ever reset
        cannot be removed until its history is.
        """
        await self.db.execute(delete(BudgetResetLog).where(BudgetResetLog.budget_id == budget_id))

    async def remove(self, budget: Budget) -> None:
        """Stage the deletion of a budget.

        Raises:
            BudgetStillReferencedError: the database refused the delete because a row still names the budget.
        """
        # A failed flush expires the row, so the ID is read before it.
        budget_id = budget.budget_id
        await self.db.delete(budget)
        try:
            await self.db.flush()
        except IntegrityError:
            raise BudgetStillReferencedError(budget_id) from None
