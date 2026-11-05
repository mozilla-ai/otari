"""Video jobs: the record of who started one here, and billing a completed one once."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.repositories.inference import VideoJobRepository

if TYPE_CHECKING:
    import uuid

    from gateway.models.inference import VideoJobRecord


class VideoJobService:
    """The video jobs started here: their records, and the one-time settlement of a completed one."""

    def __init__(self, uow: UnitOfWork, jobs: VideoJobRepository) -> None:
        self._uow = uow
        self._jobs = jobs

    async def record(
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
    ) -> None:
        """Keep the record of a job the provider accepted.

        Best effort: the job already exists upstream, so a record that cannot
        be written is logged rather than failing the request that started it.
        """
        try:
            async with self._uow:
                await self._jobs.add(
                    video_id=video_id,
                    provider=provider,
                    model=model,
                    user_id=user_id,
                    api_key_id=api_key_id,
                    workspace_id=workspace_id,
                    seconds=seconds,
                    counts_toward_budget=counts_toward_budget,
                )
        except DATABASE_ERRORS as exc:
            logger.warning("Failed to persist the record of video job '%s': %s", video_id, describe_database_error(exc))

    async def find(self, video_id: str) -> VideoJobRecord | None:
        """The job's record, or None when no job with that id was started here."""
        async with self._uow:
            return await self._jobs.find(video_id)

    async def page(self, *, user_id: str | None, workspace_id: uuid.UUID | None, limit: int) -> list[VideoJobRecord]:
        """The newest ``limit`` jobs, those of ``user_id`` and in ``workspace_id`` where either is given."""
        async with self._uow:
            return await self._jobs.page(user_id=user_id, workspace_id=workspace_id, limit=limit)

    async def settle_once(self, video_id: str, settle: Callable[[], Awaitable[None]]) -> bool:
        """Run ``settle`` for the one caller that claims the job's accounting, and tell whether this caller did.

        The claim and what ``settle`` writes commit together, so a settlement
        that fails leaves the job unclaimed for a later caller.
        """
        async with self._uow:
            if not await self._jobs.claim_accounting(video_id):
                return False
            await settle()
            return True
