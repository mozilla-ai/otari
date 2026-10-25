"""Data access for ``trace_spans``. Flushes, never commits; a span once stored changes only by losing its content."""

import uuid
from collections.abc import Collection, Sequence
from typing import Any, Never

from sqlalchemy import func, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import TraceSpan
from gateway.repositories.base_repository import BaseRepository
from gateway.repositories.traces.trace_repository import TraceKey


class SpanRepository(BaseRepository[TraceSpan, Never, Never]):
    """Insert a trace's spans once each, and read them back in start order."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceSpan)

    async def clear_content(self, workspace_id: uuid.UUID, *, limit: int) -> int:
        """Drop the sealed content of up to ``limit`` spans in the workspace, returning how many held any."""
        held = (
            select(TraceSpan.workspace_id, TraceSpan.trace_id, TraceSpan.span_id)
            .where(TraceSpan.workspace_id == workspace_id, TraceSpan.content_sealed.is_not(None))
            .limit(limit)
        )
        result = await self.db.execute(
            update(TraceSpan)
            .where(tuple_(TraceSpan.workspace_id, TraceSpan.trace_id, TraceSpan.span_id).in_(held))
            .values(content_sealed=None)
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def insert_new(self, rows: Sequence[dict[str, Any]]) -> set[tuple[uuid.UUID, str, str]]:
        """Stage the spans not stored yet, in one statement, and return the keys of those it inserted.

        ``ON CONFLICT DO NOTHING ... RETURNING`` answers with only the rows that went
        in, so a repeated span is skipped without failing the batch and the caller
        grows a trace's totals by exactly what is new. Rows go in key order, so two
        writers cannot lock the same spans in opposite orders.
        """
        if not rows:
            return set()
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        ordered = sorted(rows, key=lambda row: (str(row["workspace_id"]), row["trace_id"], row["span_id"]))
        result = await self.db.execute(
            insert(TraceSpan)
            .values(ordered)
            .on_conflict_do_nothing(index_elements=[TraceSpan.workspace_id, TraceSpan.trace_id, TraceSpan.span_id])
            .returning(TraceSpan.workspace_id, TraceSpan.trace_id, TraceSpan.span_id)
        )
        inserted = {(row[0], row[1], row[2]) for row in result.all()}
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

    async def trace_of_request(self, workspace_ids: Collection[uuid.UUID] | None, request_id: str) -> TraceKey | None:
        """Return the key of the trace holding a gateway request's spans, or None when none is stored.

        ``workspace_ids`` is the read's scope: None is every workspace, an empty set none.
        """
        conditions = [TraceSpan.request_id == request_id]
        if workspace_ids is not None:
            if not workspace_ids:
                return None
            conditions.append(TraceSpan.workspace_id.in_(list(workspace_ids)))
        result = await self.db.execute(select(TraceSpan.workspace_id, TraceSpan.trace_id).where(*conditions).limit(1))
        row = result.first()
        return (row[0], row[1]) if row is not None else None
