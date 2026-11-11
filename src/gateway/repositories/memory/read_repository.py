"""Data access for ``memory_reads``. Flushes, never commits."""

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import delete, func, select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.memory import MemoryRead
from gateway.repositories.memory.fenced_repository import FencedRepository


class ReadRepository(FencedRepository[MemoryRead]):
    """Record who recalled which entries. A record keeps no content."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, MemoryRead)

    async def record(
        self,
        workspace_id: uuid.UUID,
        *,
        reader_user_id: uuid.UUID | None,
        api_key_id: str | None,
        tool: str,
        entry_ids: Sequence[uuid.UUID],
    ) -> MemoryRead:
        """Stage one recall's record."""
        await self._fence(workspace_id)
        row = MemoryRead(
            workspace_id=workspace_id,
            reader_user_id=reader_user_id,
            api_key_id=api_key_id,
            tool=tool,
            entry_ids=[str(entry_id) for entry_id in entry_ids],
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def count_by_tool(self, workspace_id: uuid.UUID, *, since: datetime, until: datetime) -> dict[str, int]:
        """How many recalls of the workspace each tool made in the window."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryRead.tool, func.count())
            .where(MemoryRead.workspace_id == workspace_id, MemoryRead.read_at >= since, MemoryRead.read_at < until)
            .group_by(MemoryRead.tool)
        )
        return {tool: count for tool, count in result.all()}

    async def delete_older(self, workspace_id: uuid.UUID, *, read_before: datetime) -> int:
        """Delete the workspace's read records made before ``read_before``."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryRead)
            .where(MemoryRead.workspace_id == workspace_id, MemoryRead.read_at < read_before)
            .returning(MemoryRead.id)
        )
        return len(result.scalars().all())

    async def delete_all(self, workspace_id: uuid.UUID) -> int:
        """Delete every read record of the workspace."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryRead).where(MemoryRead.workspace_id == workspace_id).returning(MemoryRead.id)
        )
        return len(result.scalars().all())
