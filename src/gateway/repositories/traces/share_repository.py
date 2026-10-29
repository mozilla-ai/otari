"""Data access for ``trace_content_shares`` and ``trace_content_reads``. Flushes, never commits."""

import uuid
from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Never

from sqlalchemy import case, func, select, update

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
    ) -> TraceContentShare:
        """Stage a new share."""
        share = TraceContentShare(
            workspace_id=workspace_id,
            trace_id=trace_id,
            audience=audience,
            note=note,
            shared_by_user_id=shared_by_user_id,
        )
        self.db.add(share)
        await self.db.flush()
        return share

    async def for_trace(
        self, workspace_id: uuid.UUID, trace_id: str, *, limit: int, offset: int = 0
    ) -> Sequence[TraceContentShare]:
        """A page of the shares the session has had: the running ones first, then the rest, newest first in each.

        Running first, so however many ended shares a session gathers, the ones that
        still let someone read are on the first page.
        """
        running = TraceContentShare.revoked_at.is_(None)
        result = await self.db.execute(
            select(TraceContentShare)
            .where(TraceContentShare.workspace_id == workspace_id, TraceContentShare.trace_id == trace_id)
            .order_by(case((running, 0), else_=1), TraceContentShare.created_at.desc(), TraceContentShare.id)
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def active(
        self, workspace_id: uuid.UUID, trace_id: str, audiences: Collection[str]
    ) -> TraceContentShare | None:
        """The newest share of the session to any of ``audiences`` that has not ended."""
        result = await self.db.execute(
            select(TraceContentShare)
            .where(
                TraceContentShare.workspace_id == workspace_id,
                TraceContentShare.trace_id == trace_id,
                TraceContentShare.audience.in_(list(audiences)),
                TraceContentShare.revoked_at.is_(None),
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

    async def revoke_by(self, workspace_ids: Collection[uuid.UUID], user_id: uuid.UUID, now: datetime) -> None:
        """Stage the end of every running share this user made in any of the workspaces, in one statement."""
        if not workspace_ids:
            return
        await self.db.execute(
            update(TraceContentShare)
            .where(
                TraceContentShare.workspace_id.in_(list(workspace_ids)),
                TraceContentShare.shared_by_user_id == user_id,
                TraceContentShare.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )

    async def revoke_all_by(self, user_id: uuid.UUID, now: datetime) -> None:
        """Stage the end of every running share this user made, in any workspace, in one statement."""
        await self.db.execute(
            update(TraceContentShare)
            .where(TraceContentShare.shared_by_user_id == user_id, TraceContentShare.revoked_at.is_(None))
            .values(revoked_at=now)
        )

    async def record_read(
        self,
        share: TraceContentShare,
        session_user_id: str | None,
        reader_user_id: uuid.UUID,
        reader_email: str,
        *,
        unless_since: datetime,
    ) -> None:
        """Stage the record of one read through a share, unless this reader already has one since ``unless_since``."""
        share_id = share.id
        recent = await self.db.execute(
            select(TraceContentRead.id)
            .where(
                TraceContentRead.share_id == share_id,
                TraceContentRead.reader_user_id == reader_user_id,
                TraceContentRead.read_at >= unless_since,
            )
            .limit(1)
        )
        if recent.scalar_one_or_none() is not None:
            return
        self.db.add(
            TraceContentRead(
                share_id=share_id,
                workspace_id=share.workspace_id,
                trace_id=share.trace_id,
                session_user_id=session_user_id,
                audience=share.audience,
                reader_user_id=reader_user_id,
                reader_email=reader_email,
            )
        )
        await self.db.flush()

    async def latest_reads(self, share_ids: Sequence[uuid.UUID], *, per_share: int) -> Sequence[TraceContentRead]:
        """The newest ``per_share`` reads through each of these shares, newest first."""
        if not share_ids:
            return []
        rank = (
            func.row_number()
            .over(partition_by=TraceContentRead.share_id, order_by=TraceContentRead.read_at.desc())
            .label("rank")
        )
        ranked = select(TraceContentRead.id, rank).where(TraceContentRead.share_id.in_(share_ids)).subquery()
        result = await self.db.execute(
            select(TraceContentRead)
            .join(ranked, ranked.c.id == TraceContentRead.id)
            .where(ranked.c.rank <= per_share)
            .order_by(TraceContentRead.read_at.desc())
        )
        return result.scalars().all()

    async def read_counts(self, share_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, int]:
        """How many reads each of these shares has had."""
        if not share_ids:
            return {}
        result = await self.db.execute(
            select(TraceContentRead.share_id, func.count())
            .where(TraceContentRead.share_id.in_(share_ids))
            .group_by(TraceContentRead.share_id)
        )
        return {share_id: int(count) for share_id, count in result.tuples().all() if share_id is not None}

    async def reads_page(self, share_id: uuid.UUID, *, limit: int, offset: int = 0) -> Sequence[TraceContentRead]:
        """A page of the reads through one share, newest first."""
        result = await self.db.execute(
            select(TraceContentRead)
            .where(TraceContentRead.share_id == share_id)
            .order_by(TraceContentRead.read_at.desc(), TraceContentRead.id)
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def forget_session_user(self, user_id: str) -> None:
        """Clear whose session each read was of, for that user's erasure; the reads themselves stay recorded."""
        await self.db.execute(
            update(TraceContentRead).where(TraceContentRead.session_user_id == user_id).values(session_user_id=None)
        )
