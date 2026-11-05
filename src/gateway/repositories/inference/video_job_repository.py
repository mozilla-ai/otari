"""Data access for the video jobs callers start here."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Never, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.inference import VideoJobRecord
from gateway.repositories.base_repository import BaseRepository

if TYPE_CHECKING:
    import uuid


class VideoJobRepository(BaseRepository[VideoJobRecord, Never, Never]):
    """Add, find and page video job records, and claim one's accounting, in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, VideoJobRecord)

    async def add(
        self,
        *,
        video_id: str,
        provider: str,
        model: str,
        user_id: str,
        api_key_id: str | None,
        workspace_id: uuid.UUID | None,
        seconds: int | None,
        counts_toward_budget: bool,
    ) -> VideoJobRecord:
        """Stage the record of a job the provider accepted."""
        record = VideoJobRecord(
            id=video_id,
            provider=provider,
            model=model,
            user_id=user_id,
            api_key_id=api_key_id,
            workspace_id=workspace_id,
            seconds=seconds,
            counts_toward_budget=counts_toward_budget,
        )
        self.db.add(record)
        await self.db.flush()
        return record

    async def find(self, video_id: str) -> VideoJobRecord | None:
        """The job's record, read fresh rather than from the session's identity map."""
        result = await self.db.execute(
            select(VideoJobRecord).where(VideoJobRecord.id == video_id).execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()

    async def page(self, *, user_id: str | None, workspace_id: uuid.UUID | None, limit: int) -> list[VideoJobRecord]:
        """The newest ``limit`` jobs, those of ``user_id`` and in ``workspace_id`` where either is given."""
        statement = select(VideoJobRecord)
        if user_id is not None:
            statement = statement.where(VideoJobRecord.user_id == user_id)
        if workspace_id is not None:
            statement = statement.where(VideoJobRecord.workspace_id == workspace_id)
        statement = statement.order_by(VideoJobRecord.created_at.desc(), VideoJobRecord.id).limit(limit)
        return list((await self.db.execute(statement)).scalars().all())

    async def remember_completed(
        self,
        video_id: str,
        *,
        seconds: int | None,
        output_url: str | None,
        output_mime_type: str | None,
        completed_at: datetime,
    ) -> None:
        """Keep what the completed job reported on its record."""
        await self.db.execute(
            update(VideoJobRecord)
            .where(VideoJobRecord.id == video_id)
            .values(
                completed_at=completed_at,
                completed_seconds=seconds,
                output_url=output_url,
                output_mime_type=output_mime_type,
            )
            .execution_options(synchronize_session=False)
        )

    async def claim_accounting(self, video_id: str) -> bool:
        """Mark the job billed, returning True to the one caller that moves ``accounted_at`` off NULL.

        The UPDATE takes the row lock, so a concurrent claim on the same job
        waits and then sees ``accounted_at`` set.
        """
        statement = (
            update(VideoJobRecord)
            .where(VideoJobRecord.id == video_id, VideoJobRecord.accounted_at.is_(None))
            .values(accounted_at=datetime.now(UTC))
            .execution_options(synchronize_session=False)
        )
        result = cast("CursorResult[Any]", await self.db.execute(statement))
        return bool(result.rowcount)
