"""Data access for ``trace_break_glass_reads``. Flushes, never commits; a record changes only by an erasure."""

import uuid
from collections.abc import Sequence
from typing import Never

from sqlalchemy import select, update

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import TraceBreakGlassRead
from gateway.repositories.base_repository import BaseRepository


class BreakGlassRepository(BaseRepository[TraceBreakGlassRead, Never, Never]):
    """Record break-glass openings, page through them, and delete those past their retention."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceBreakGlassRead)

    async def record(self, **values: object) -> TraceBreakGlassRead:
        """Stage one record."""
        row = TraceBreakGlassRead(**values)
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
