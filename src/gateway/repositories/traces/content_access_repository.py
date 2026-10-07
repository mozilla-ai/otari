"""Data access for ``trace_content_access``, the record of every content read. Flushes, never commits."""

import uuid
from collections.abc import Sequence
from typing import Never

from sqlalchemy import select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import TraceContentAccess
from gateway.repositories.base_repository import BaseRepository


class ContentAccessRepository(BaseRepository[TraceContentAccess, Never, Never]):
    """Record one read of captured content, and list a workspace's reads, newest first."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceContentAccess)

    async def record(
        self,
        *,
        workspace_id: uuid.UUID,
        trace_id: str,
        span_id: str,
        reader_kind: str,
        reader: str,
        reason: str | None,
    ) -> None:
        self.db.add(
            TraceContentAccess(
                workspace_id=workspace_id,
                trace_id=trace_id,
                span_id=span_id,
                reader_kind=reader_kind,
                reader=reader,
                reason=reason,
            )
        )
        await self.db.flush()

    async def for_workspace(self, workspace_id: uuid.UUID, *, limit: int, offset: int) -> Sequence[TraceContentAccess]:
        result = await self.db.execute(
            select(TraceContentAccess)
            .where(TraceContentAccess.workspace_id == workspace_id)
            .order_by(TraceContentAccess.accessed_at.desc(), TraceContentAccess.id)
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()
