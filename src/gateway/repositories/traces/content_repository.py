"""Data access for ``trace_span_content``, each span's reference into the object store. Flushes, never commits."""

import uuid
from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import delete, select, tuple_
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult

from gateway.core.sql import dialect_name, utc_bound
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import TraceSpanContent
from gateway.repositories.base_repository import BaseRepository

# One stored blob: its span's key, and where the store keeps it.
ContentRef = tuple[uuid.UUID, str, str, str]


class ContentRepository(BaseRepository[TraceSpanContent, Never, Never]):
    """Reference each span's stored content, read one back, and list what an expiry or purge must delete."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceSpanContent)

    async def insert_new(self, rows: Sequence[dict[str, Any]]) -> set[str]:
        """Stage references for spans that have none yet, and return the span ids this call inserted."""
        if not rows:
            return set()
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        result = await self.db.execute(
            insert(TraceSpanContent)
            .values(list(rows))
            .on_conflict_do_nothing(
                index_elements=[TraceSpanContent.workspace_id, TraceSpanContent.trace_id, TraceSpanContent.span_id]
            )
            .returning(TraceSpanContent.span_id)
        )
        inserted = set(result.scalars().all())
        await self.db.flush()
        return inserted

    async def find(
        self, workspace_ids: Collection[uuid.UUID] | None, trace_id: str, span_id: str
    ) -> TraceSpanContent | None:
        """One span's content reference inside the scope (None is every workspace), or None."""
        query = select(TraceSpanContent).where(
            TraceSpanContent.trace_id == trace_id, TraceSpanContent.span_id == span_id
        )
        if workspace_ids is not None:
            query = query.where(TraceSpanContent.workspace_id.in_(list(workspace_ids)))
        return (await self.db.execute(query.limit(1))).scalar_one_or_none()

    async def span_ids_with_content(self, workspace_id: uuid.UUID, trace_id: str) -> set[str]:
        result = await self.db.execute(
            select(TraceSpanContent.span_id).where(
                TraceSpanContent.workspace_id == workspace_id, TraceSpanContent.trace_id == trace_id
            )
        )
        return set(result.scalars().all())

    async def refs_before(self, before: datetime, *, limit: int) -> list[ContentRef]:
        bound = utc_bound(before)
        assert bound is not None
        return await self._refs(TraceSpanContent.created_at < bound, limit=limit)

    async def refs_for_workspace(self, workspace_id: uuid.UUID, *, limit: int) -> list[ContentRef]:
        return await self._refs(TraceSpanContent.workspace_id == workspace_id, limit=limit)

    async def refs_for_traces(self, keys: Collection[tuple[uuid.UUID, str]], *, limit: int) -> list[ContentRef]:
        if not keys:
            return []
        return await self._refs(
            tuple_(TraceSpanContent.workspace_id, TraceSpanContent.trace_id).in_(list(keys)), limit=limit
        )

    async def delete_refs(self, refs: Collection[ContentRef]) -> int:
        if not refs:
            return 0
        keys = [(workspace_id, trace_id, span_id) for workspace_id, trace_id, span_id, _ in refs]
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                delete(TraceSpanContent).where(
                    tuple_(TraceSpanContent.workspace_id, TraceSpanContent.trace_id, TraceSpanContent.span_id).in_(keys)
                )
            ),
        )
        await self.db.flush()
        return int(result.rowcount or 0)

    async def _refs(self, condition: Any, *, limit: int) -> list[ContentRef]:
        result = await self.db.execute(
            select(
                TraceSpanContent.workspace_id,
                TraceSpanContent.trace_id,
                TraceSpanContent.span_id,
                TraceSpanContent.storage_ref,
            )
            .where(condition)
            .limit(limit)
        )
        return [(row[0], row[1], row[2], row[3]) for row in result.all()]
