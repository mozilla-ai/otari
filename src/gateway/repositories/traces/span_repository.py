"""Data access for ``trace_spans``. Flushes, never commits; a span once stored is never updated."""

import uuid
from collections.abc import Sequence
from typing import Any, Never

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import TraceSpan
from gateway.repositories.base_repository import BaseRepository


class SpanRepository(BaseRepository[TraceSpan, Never, Never]):
    """Insert a trace's spans once each, and read them back in start order."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceSpan)

    async def insert_new(self, rows: Sequence[dict[str, Any]]) -> set[str]:
        """Stage the spans not stored yet and return the ids of those this call inserted.

        ``ON CONFLICT DO NOTHING ... RETURNING`` answers with only the rows that went
        in, so a repeated span is skipped without failing the batch and the caller
        grows a trace's totals by exactly what is new.
        """
        if not rows:
            return set()
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        result = await self.db.execute(
            insert(TraceSpan)
            .values(list(rows))
            .on_conflict_do_nothing(index_elements=[TraceSpan.workspace_id, TraceSpan.trace_id, TraceSpan.span_id])
            .returning(TraceSpan.span_id)
        )
        inserted = set(result.scalars().all())
        await self.db.flush()
        return inserted

    async def for_trace(self, workspace_id: uuid.UUID, trace_id: str, *, limit: int) -> Sequence[TraceSpan]:
        """Return up to ``limit`` of a trace's spans, earliest first.

        A client-run tool has no start, so it sorts by its end, which is when the
        request carrying its result arrived.
        """
        result = await self.db.execute(
            select(TraceSpan)
            .where(TraceSpan.workspace_id == workspace_id, TraceSpan.trace_id == trace_id)
            .order_by(func.coalesce(TraceSpan.start_time, TraceSpan.end_time), TraceSpan.span_id)
            .limit(limit)
        )
        return result.scalars().all()
