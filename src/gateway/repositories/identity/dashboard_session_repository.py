import uuid
from datetime import datetime
from typing import Never

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import DashboardSession
from gateway.repositories.base_repository import BaseRepository


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

    async def add(self, *, token_hash: str, user_id: uuid.UUID, created_at: datetime, expires_at: datetime) -> None:
        """Stage a session for an account."""
        self.db.add(
            DashboardSession(token_hash=token_hash, user_id=user_id, created_at=created_at, expires_at=expires_at)
        )
