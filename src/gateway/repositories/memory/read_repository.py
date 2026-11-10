"""Data access for ``memory_reads``. Flushes, never commits."""

import uuid
from collections.abc import Sequence

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
