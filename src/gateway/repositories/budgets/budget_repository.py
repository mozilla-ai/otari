import uuid
from typing import Never

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.budget_exceptions import BudgetStillReferencedError
from gateway.models.budgets import Budget, WorkspaceBudgetDefault
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

    async def count_by_organization(self, organization_id: uuid.UUID) -> int:
        """Count the organization's budgets."""
        result = await self.db.execute(
            select(func.count()).select_from(Budget).where(Budget.organization_id == organization_id)
        )
        return result.scalar_one()

    async def count_member_policies_for_budget(self, budget_id: str) -> int:
        """Count the workspace member budget policies that name this budget."""
        result = await self.db.execute(
            select(func.count())
            .select_from(WorkspaceBudgetDefault)
            .where(WorkspaceBudgetDefault.budget_id == budget_id)
        )
        return result.scalar_one()

    async def count_users_for_budget(self, budget_id: str) -> int:
        """Count the gateway users assigned this budget."""
        result = await self.db.execute(select(func.count()).select_from(User).where(User.budget_id == budget_id))
        return result.scalar_one()

    async def get_by_id_and_organization(self, budget_id: str, organization_id: uuid.UUID) -> Budget | None:
        """Return the budget with this ID when this organization owns it, otherwise None."""
        result = await self.db.execute(
            select(Budget).where(Budget.budget_id == budget_id, Budget.organization_id == organization_id)
        )
        return result.scalar_one_or_none()

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

    async def remove(self, budget: Budget) -> None:
        """Stage the deletion of a budget.

        Raises:
            BudgetStillReferencedError: the database refused the delete because a row still names the budget.
        """
        await self.db.delete(budget)
        try:
            await self.db.flush()
        except IntegrityError:
            raise BudgetStillReferencedError(budget.budget_id) from None
