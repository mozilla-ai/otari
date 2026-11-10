"""Data access for ``memory_keys``. Flushes, never commits."""

import uuid
from collections.abc import Collection, Sequence

from sqlalchemy import select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.memory import MemoryKey
from gateway.repositories.memory.fenced_repository import FencedRepository


class KeyRepository(FencedRepository[MemoryKey]):
    """Store a workspace's wrapped data keys, and find the ones its rows were sealed under."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, MemoryKey)

    async def newest(self, workspace_id: uuid.UUID) -> MemoryKey | None:
        """The workspace's most recently made key, or None before it has one."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryKey)
            .where(MemoryKey.workspace_id == workspace_id)
            .order_by(MemoryKey.created_at.desc(), MemoryKey.id)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def add(self, workspace_id: uuid.UUID, *, wrapper: str, wrapped: str) -> MemoryKey:
        """Stage a new key for the workspace."""
        await self._fence(workspace_id)
        key = MemoryKey(workspace_id=workspace_id, wrapper=wrapper, wrapped=wrapped)
        self.db.add(key)
        await self.db.flush()
        return key

    async def by_ids(self, workspace_id: uuid.UUID, key_ids: Collection[uuid.UUID]) -> Sequence[MemoryKey]:
        """The workspace's keys among ``key_ids``; an id of another workspace's key finds nothing."""
        if not key_ids:
            return []
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryKey).where(MemoryKey.workspace_id == workspace_id, MemoryKey.id.in_(list(key_ids)))
        )
        return result.scalars().all()
