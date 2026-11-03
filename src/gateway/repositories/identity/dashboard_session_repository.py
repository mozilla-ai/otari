import uuid
from collections.abc import Collection
from datetime import datetime
from typing import Never

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import DashboardSession
from gateway.repositories.base_repository import BaseRepository

_DELETE_CHUNK = 1000


class DashboardSessionRepository(BaseRepository[DashboardSession, Never, Never]):
    """Stage dashboard session rows."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, DashboardSession)

    async def prune_expired(self, now: datetime) -> None:
        """Stage the deletion of every session that expired before ``now``."""
        await self.db.execute(delete(DashboardSession).where(DashboardSession.expires_at < now))

    async def delete_for_user(self, user_id: uuid.UUID) -> None:
        """Stage the deletion of every session one account holds."""
        await self.db.execute(delete(DashboardSession).where(DashboardSession.user_id == user_id))

    async def delete_for_users(self, user_ids: Collection[uuid.UUID]) -> None:
        """Stage the deletion of every session the accounts in ``user_ids`` hold.

        Chunked, so a domain matching many accounts stays under the driver's bind-parameter limit.
        """
        ids = list(user_ids)
        for start in range(0, len(ids), _DELETE_CHUNK):
            chunk = ids[start : start + _DELETE_CHUNK]
            await self.db.execute(delete(DashboardSession).where(col(DashboardSession.user_id).in_(chunk)))

    async def add(self, *, token_hash: str, user_id: uuid.UUID, created_at: datetime, expires_at: datetime) -> None:
        """Stage a session for an account."""
        self.db.add(
            DashboardSession(token_hash=token_hash, user_id=user_id, created_at=created_at, expires_at=expires_at)
        )
