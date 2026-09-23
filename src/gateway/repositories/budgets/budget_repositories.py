from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.budgets.budget_repository import BudgetRepository
from gateway.repositories.budgets.scoped_budget_repository import ScopedBudgetRepository
from gateway.repositories.budgets.workspace_budget_default_repository import WorkspaceBudgetDefaultRepository


@dataclass(frozen=True)
class BudgetRepositories:
    """The budgets domain's repositories, all on one Unit of Work."""

    budgets: BudgetRepository
    ceilings: ScopedBudgetRepository
    member_policies: WorkspaceBudgetDefaultRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(
            budgets=BudgetRepository(uow),
            ceilings=ScopedBudgetRepository(uow),
            member_policies=WorkspaceBudgetDefaultRepository(uow),
        )
