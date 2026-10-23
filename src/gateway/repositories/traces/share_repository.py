"""Data access for ``trace_content_shares`` and ``trace_content_reads``. Flushes, never commits."""

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Never

from sqlalchemy import select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import TraceContentRead, TraceContentShare
from gateway.repositories.base_repository import BaseRepository


class ContentShareRepository(BaseRepository[TraceContentShare, Never, Never]):
    """Store a session's shares and the reads made through them, and find the one a reader may use."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceContentShare)

    async def add(
        self,
        *,
        workspace_id: uuid.UUID,
        trace_id: str,
        audience: str,
        note: str | None,
        shared_by_user_id: uuid.UUID,
        expires_at: datetime,
    ) -> TraceContentShare:
        """Stage a new share."""
        share = TraceContentShare(
            workspace_id=workspace_id,
            trace_id=trace_id,
            audience=audience,
            note=note,
            shared_by_user_id=shared_by_user_id,
            expires_at=expires_at,
        )
        self.db.add(share)
        await self.db.flush()
        return share

    async def for_trace(self, workspace_id: uuid.UUID, trace_id: str) -> Sequence[TraceContentShare]:
        """Every share the session has had, newest first."""
        result = await self.db.execute(
            select(TraceContentShare)
            .where(TraceContentShare.workspace_id == workspace_id, TraceContentShare.trace_id == trace_id)
            .order_by(TraceContentShare.created_at.desc())
        )
        return result.scalars().all()

    async def active(
        self, workspace_id: uuid.UUID, trace_id: str, audience: str, now: datetime
    ) -> TraceContentShare | None:
        """The newest share of the session to ``audience`` that has neither expired nor been revoked."""
        result = await self.db.execute(
            select(TraceContentShare)
            .where(
                TraceContentShare.workspace_id == workspace_id,
                TraceContentShare.trace_id == trace_id,
                TraceContentShare.audience == audience,
                TraceContentShare.revoked_at.is_(None),
                TraceContentShare.expires_at > now,
            )
            .order_by(TraceContentShare.created_at.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def find(self, workspace_id: uuid.UUID, trace_id: str, share_id: uuid.UUID) -> TraceContentShare | None:
        """The session's share by id, or None when the session holds no such share."""
        result = await self.db.execute(
            select(TraceContentShare).where(
                TraceContentShare.id == share_id,
                TraceContentShare.workspace_id == workspace_id,
                TraceContentShare.trace_id == trace_id,
            )
        )
        return result.scalars().first()

    async def revoke(self, share: TraceContentShare, now: datetime) -> None:
        """Stage the share's end, unless it already ended."""
        if share.revoked_at is None:
            share.revoked_at = now
            await self.db.flush()

    async def record_read(self, share_id: uuid.UUID, reader_user_id: uuid.UUID, reader_email: str) -> None:
        """Stage the record of one read through a share."""
        self.db.add(TraceContentRead(share_id=share_id, reader_user_id=reader_user_id, reader_email=reader_email))
        await self.db.flush()

    async def reads(self, share_ids: Sequence[uuid.UUID]) -> Sequence[TraceContentRead]:
        """The reads made through these shares, newest first."""
        if not share_ids:
            return []
        result = await self.db.execute(
            select(TraceContentRead)
            .where(TraceContentRead.share_id.in_(share_ids))
            .order_by(TraceContentRead.read_at.desc())
        )
        return result.scalars().all()
