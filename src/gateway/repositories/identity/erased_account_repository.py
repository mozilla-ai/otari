import uuid
from typing import Never

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.identity import ErasedAccount
from gateway.repositories.base_repository import BaseRepository


class ErasedAccountRepository(BaseRepository[ErasedAccount, Never, Never]):
    """Record the accounts deleted at their own request."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, ErasedAccount)

    async def add(self, account_id: uuid.UUID) -> None:
        """Stage the record that ``account_id`` was erased."""
        self.db.add(ErasedAccount(id=account_id))
        await self.db.flush()
