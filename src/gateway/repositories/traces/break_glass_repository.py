"""Data access for ``trace_break_glass_reads``. Flushes, never commits; a record changes only by an erasure."""

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Never

from sqlalchemy import delete, select, update

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import TraceBreakGlassRead
from gateway.repositories.base_repository import BaseRepository
from gateway.types.traces import BreakGlass


class BreakGlassRepository(BaseRepository[TraceBreakGlassRead, Never, Never]):
    """Record break-glass openings, page through them, and delete those past their retention."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceBreakGlassRead)

    async def record(
        self, opening: BreakGlass, *, workspace_id: uuid.UUID, trace_id: str, session_user_id: str | None
    ) -> TraceBreakGlassRead:
        """Stage one record: who opened content and why, and which session they opened."""
        row = TraceBreakGlassRead(
            kind=opening.kind,
            organization_id=opening.organization_id,
            workspace_id=workspace_id,
            trace_id=trace_id,
            session_user_id=session_user_id,
            reader_user_id=opening.reader_user_id,
            reader_email=opening.reader_email,
            reason=opening.reason,
            reference=opening.reference,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def page(
        self, *, organization_id: uuid.UUID | None, kind: str | None, limit: int, offset: int = 0
    ) -> Sequence[TraceBreakGlassRead]:
        """A page of records, newest first, narrowed to an organization and a kind where given."""
        query = select(TraceBreakGlassRead)
        if organization_id is not None:
            query = query.where(TraceBreakGlassRead.organization_id == organization_id)
        if kind is not None:
            query = query.where(TraceBreakGlassRead.kind == kind)
        result = await self.db.execute(
            query.order_by(TraceBreakGlassRead.read_at.desc(), TraceBreakGlassRead.id).limit(limit).offset(offset)
        )
        return result.scalars().all()

    async def forget_session_user(self, user_id: str) -> None:
        """Clear whose session a record opened, for that user's erasure; the opening itself stays recorded."""
        await self.db.execute(
            update(TraceBreakGlassRead)
            .where(TraceBreakGlassRead.session_user_id == user_id)
            .values(session_user_id=None)
        )

    async def delete_before(self, cutoff: datetime) -> int:
        """Delete every record older than ``cutoff``, returning how many."""
        result = await self.db.execute(delete(TraceBreakGlassRead).where(TraceBreakGlassRead.read_at < cutoff))
        return int(getattr(result, "rowcount", 0) or 0)
