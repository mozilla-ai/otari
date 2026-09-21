"""The generic repository serves a declarative table as well as a SQLModel one.

mypy does most of the checking here, because the runtime does not enforce the bound of a type argument.
"""

import pytest
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.budgets import BudgetResetLog
from gateway.repositories.base_repository import BaseRepository

pytestmark = pytest.mark.asyncio


class _Empty(BaseModel):
    pass


class _BudgetResetLogReader(BaseRepository[BudgetResetLog, _Empty, _Empty]):
    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, BudgetResetLog)


async def test_a_repository_counts_a_declarative_table(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        assert await _BudgetResetLogReader(uow).count() == 0
