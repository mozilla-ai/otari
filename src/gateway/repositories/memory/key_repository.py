"""Data access for ``memory_keys``. Flushes, never commits."""

import uuid
from collections.abc import Collection, Sequence
from datetime import datetime

from sqlalchemy import delete, select

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

    async def delete_unused(
        self, workspace_id: uuid.UUID, *, in_use: Collection[uuid.UUID], made_before: datetime
    ) -> int:
        """Delete the workspace's keys made before ``made_before`` that seal nothing, all but its newest.

        The newest still seals what comes next, and a key younger than ``made_before`` may be one another
        replica is sealing under right now. Once its key is gone, a row deleted before it opens from no
        backup either.
        """
        newest = await self.newest(workspace_id)
        keep = {*in_use, *([newest.id] if newest is not None else [])}
        query = delete(MemoryKey).where(MemoryKey.workspace_id == workspace_id, MemoryKey.created_at < made_before)
        if keep:
            query = query.where(MemoryKey.id.not_in(list(keep)))
        result = await self.db.execute(query.returning(MemoryKey.id))
        return len(result.scalars().all())

    async def delete_all(self, workspace_id: uuid.UUID) -> list[uuid.UUID]:
        """Delete every key of the workspace, once nothing is sealed under them; the ids deleted."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryKey).where(MemoryKey.workspace_id == workspace_id).returning(MemoryKey.id)
        )
        return list(result.scalars().all())
