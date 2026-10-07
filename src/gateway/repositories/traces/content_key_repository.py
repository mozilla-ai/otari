"""Data access for ``trace_content_keys``, each session's wrapped data key. Flushes, never commits."""

import uuid
from collections.abc import Collection
from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import delete, select, tuple_
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult

from gateway.core.sql import dialect_name, utc_bound
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import TraceContentKey
from gateway.repositories.base_repository import BaseRepository


class ContentKeyRepository(BaseRepository[TraceContentKey, Never, Never]):
    """Keep one wrapped data key per session, read it back, and destroy it with its session."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceContentKey)

    async def add_if_absent(self, *, workspace_id: uuid.UUID, trace_id: str, key_ref: str, wrapped: bytes) -> None:
        """Stage a session's key unless it has one; a concurrent writer's key wins."""
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        await self.db.execute(
            insert(TraceContentKey)
            .values(workspace_id=workspace_id, trace_id=trace_id, key_ref=key_ref, wrapped=wrapped)
            .on_conflict_do_nothing(index_elements=[TraceContentKey.workspace_id, TraceContentKey.trace_id])
        )
        await self.db.flush()

    async def find(self, workspace_id: uuid.UUID, trace_id: str) -> TraceContentKey | None:
        return await self.db.get(TraceContentKey, (workspace_id, trace_id))

    async def delete_for_traces(self, workspace_id: uuid.UUID, trace_ids: Collection[str]) -> int:
        if not trace_ids:
            return 0
        return await self._delete(
            (TraceContentKey.workspace_id == workspace_id) & TraceContentKey.trace_id.in_(list(trace_ids))
        )

    async def created_before(self, before: datetime, *, limit: int) -> list[tuple[uuid.UUID, str]]:
        """Sessions whose key is older than ``before``; the caller decides which of them are gone."""
        bound = utc_bound(before)
        assert bound is not None
        result = await self.db.execute(
            select(TraceContentKey.workspace_id, TraceContentKey.trace_id)
            .where(TraceContentKey.created_at < bound)
            .limit(limit)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def delete_sessions(self, keys: Collection[tuple[uuid.UUID, str]]) -> int:
        if not keys:
            return 0
        return await self._delete(tuple_(TraceContentKey.workspace_id, TraceContentKey.trace_id).in_(list(keys)))

    async def delete_for_workspace(self, workspace_id: uuid.UUID) -> int:
        return await self._delete(TraceContentKey.workspace_id == workspace_id)

    async def _delete(self, condition: Any) -> int:
        result = cast("CursorResult[Any]", await self.db.execute(delete(TraceContentKey).where(condition)))
        await self.db.flush()
        return int(result.rowcount or 0)
