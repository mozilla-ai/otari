"""Video jobs: who may read one, the record of who started it, and billing a completed one once."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING, Protocol

from any_llm.types.video import VideoJob, VideoOutput

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.inference_exceptions import VideoJobNotFoundError
from gateway.log_config import logger
from gateway.repositories.inference import VideoJobRepository
from gateway.services.pricing import price_video

if TYPE_CHECKING:
    import uuid

    from gateway.models.inference import VideoJobRecord
    from gateway.services.pricing import UsageCharge
    from gateway.services.pricing_service import ResolvedPricing


@dataclass(frozen=True)
class CompletedVideoUsage:
    """What the row settling a completed video job carries: who it bills, and what it was priced at."""

    video_id: str
    workspace_id: uuid.UUID
    api_key_id: str | None
    user_id: str
    model: str
    provider: str
    endpoint: str
    counts_toward_budget: bool
    deployment_paid: bool
    charge: UsageCharge | None
    pricing: ResolvedPricing | None

    @property
    def cost(self) -> Decimal | None:
        """What the job cost, or ``None`` when it could not be priced."""
        return None if self.charge is None else self.charge.cost


class VideoUsageRecorder(Protocol):
    """Writes the usage row that settles a completed video job, in the claim's transaction, and tells of it after."""

    async def record(self, usage: CompletedVideoUsage) -> None:
        """Stage the row in the open transaction, raising when it cannot be."""
        ...

    async def recorded(self, usage: CompletedVideoUsage) -> None:
        """Report the row once its transaction has committed. Never raises."""
        ...


class VideoRateLookup(Protocol):
    """Finds the rate a video job settles at: the model's price list, under its organization's overrides."""

    async def __call__(
        self, provider: str, model: str, *, organization_id: uuid.UUID | None, deployment_paid: bool
    ) -> ResolvedPricing | None: ...


class VideoJobService:
    """The video jobs started here: who may read one, their records, and the one-time settlement of a completed one."""

    def __init__(
        self,
        uow: UnitOfWork,
        jobs: VideoJobRepository,
        *,
        usage: VideoUsageRecorder,
        fold_spend: Callable[[str, Decimal], Awaitable[None]],
        rate: VideoRateLookup,
    ) -> None:
        self._uow = uow
        self._jobs = jobs
        self._usage = usage
        self._fold_spend = fold_spend
        self._rate = rate

    @staticmethod
    def authorize(
        record: VideoJobRecord | None,
        video_id: str,
        *,
        user_id: str | None,
        workspace_id: uuid.UUID | None,
        is_master_key: bool,
    ) -> VideoJobRecord:
        """The job's record for its owner or the master key; an unknown or foreign id is not found.

        Both the owning user and the owning workspace have to match, as for a
        batch: the workspace half keeps a key in one workspace from reaching a
        job the same user started in another, and from dispatching that job's
        calls on the other workspace's organization credential. A job whose
        workspace was deleted since has none to match, so only the master key
        reads it.
        """
        if record is not None:
            if is_master_key:
                return record
            if record.user_id == user_id and record.workspace_id is not None and record.workspace_id == workspace_id:
                return record
        raise VideoJobNotFoundError(video_id)

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

        A write that fails is tried once more. One that fails again is logged
        with the job's id, so an operator can reconcile the job upstream, and
        raised: a job the gateway cannot find again must not be handed to the
        caller as started.
        """
        for attempt in (1, 2):
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
                if attempt == 2:
                    logger.error(
                        "The record of video job '%s' on %s could not be written; the job runs upstream unrecorded: %s",
                        video_id,
                        provider,
                        describe_database_error(exc),
                    )
                    raise
            else:
                return

    async def find(self, video_id: str) -> VideoJobRecord | None:
        """The job's record, or None when no job with that id was started here."""
        async with self._uow:
            return await self._jobs.find(video_id)

    async def page(self, *, user_id: str | None, workspace_id: uuid.UUID | None, limit: int) -> list[VideoJobRecord]:
        """The newest ``limit`` jobs, those of ``user_id`` and in ``workspace_id`` where either is given."""
        async with self._uow:
            return await self._jobs.page(user_id=user_id, workspace_id=workspace_id, limit=limit)

    async def remember_completed(self, record: VideoJobRecord, job: VideoJob) -> None:
        """Keep what a completed job reported, so a later read needs no provider that may have forgotten it.

        The first output's address and media type are kept; inline data is not,
        since a provider that returns it also keeps the job. The record then
        answers every later read of the job and serves its content.
        """
        output = next((item for item in job.outputs if item.url is not None), None)
        async with self._uow:
            await self._jobs.remember_completed(
                record.id,
                seconds=job.seconds,
                output_url=None if output is None else output.url,
                output_mime_type=None if output is None else output.mime_type,
                completed_at=datetime.now(UTC),
            )

    @staticmethod
    def remembered(record: VideoJobRecord) -> VideoJob | None:
        """The completed job the record remembers, or None when the provider has to be asked."""
        if record.completed_at is None:
            return None
        outputs = []
        if record.output_url is not None:
            outputs.append(VideoOutput(url=record.output_url, mime_type=record.output_mime_type or "video/mp4"))
        return VideoJob(
            id=record.id,
            model=record.model,
            status="completed",
            created_at=int(record.created_at.timestamp()),
            completed_at=int(record.completed_at.timestamp()),
            seconds=record.completed_seconds,
            outputs=outputs,
        )

    async def settle_completed(
        self,
        record: VideoJobRecord,
        *,
        reported_seconds: int | None,
        endpoint: str,
        workspace_id: uuid.UUID,
        organization_id: uuid.UUID | None,
        deployment_paid: bool,
        charge_deployment: Callable[[Decimal | None], Awaitable[None]] | None = None,
    ) -> bool:
        """Bill a completed job once, its length at the model's per-second rate, and tell whether this call did.

        The length is what the provider reports, else what the request asked
        for. A job with no per-second rate now stays unclaimed, so a later read
        bills it once a rate exists. One whose length nobody knows is claimed
        with no cost, so it is not retried forever. ``charge_deployment`` is
        given the cost before the row is written, where the deployment paid for
        the job.
        """
        if record.accounted_at is not None:
            return False
        seconds = reported_seconds if reported_seconds is not None else record.seconds
        found = await self._rate(
            record.provider, record.model, organization_id=organization_id, deployment_paid=deployment_paid
        )
        pricing = found.pricing if found is not None else None
        charge = price_video(pricing, seconds) if seconds is not None and seconds >= 0 else None
        if seconds is not None and charge is None:
            logger.warning(
                "Video job %s on %s:%s completed with no per-second video price; it stays unbilled until one is set",
                record.id,
                record.provider,
                record.model,
            )
            return False
        usage = CompletedVideoUsage(
            video_id=record.id,
            workspace_id=workspace_id,
            api_key_id=record.api_key_id,
            user_id=record.user_id,
            model=record.model,
            provider=record.provider,
            endpoint=endpoint,
            counts_toward_budget=record.counts_toward_budget,
            deployment_paid=deployment_paid,
            charge=charge,
            pricing=found,
        )
        charge_cost = None if charge_deployment is None else partial(charge_deployment, usage.cost)
        return await self.settle_once(record.id, usage, charge_deployment=charge_cost)

    async def settle_once(
        self,
        video_id: str,
        usage: CompletedVideoUsage,
        *,
        charge_deployment: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        """Settle the job for the one caller that claims its accounting, and tell whether this caller did.

        The claim, the usage row and the spend fold are one transaction, so a
        settlement that fails leaves the job unclaimed for a later caller, and
        no caller sees a claimed job without its row. ``charge_deployment``
        charges a deployment-paid job's cost to its organization first, where
        there is an organization to charge.
        """
        async with self._uow:
            if not await self._jobs.claim_accounting(video_id):
                return False
            if charge_deployment is not None:
                await charge_deployment()
            await self._usage.record(usage)
            cost = usage.cost
            if cost and usage.counts_toward_budget:
                await self._fold_spend(usage.user_id, cost)
        await self._usage.recorded(usage)
        return True
