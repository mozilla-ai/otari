"""The video job records: who started a job here, and billing a completed one once."""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.users import User
from gateway.repositories.inference import InferenceRepositories
from gateway.services.inference import VideoJobService
from gateway.services.workspace_scope import default_workspace_id

_OWNER = "video-owner"
_OTHER = "video-other"


async def _service(async_db: AsyncSession) -> VideoJobService:
    async_db.add_all([User(user_id=_OWNER), User(user_id=_OTHER)])
    await async_db.commit()
    uow = UnitOfWork(async_db)
    return VideoJobService(uow, InferenceRepositories.on(uow).video_jobs)


@pytest.mark.asyncio
async def test_a_recorded_job_is_found_with_what_was_recorded(async_db: AsyncSession) -> None:
    service = await _service(async_db)
    workspace_id = await default_workspace_id(async_db)

    await service.record(
        video_id="video_1",
        provider="gemini",
        model="veo-3.0-generate-001",
        user_id=_OWNER,
        api_key_id=None,
        workspace_id=workspace_id,
        seconds=8,
        counts_toward_budget=False,
    )
    record = await service.find("video_1")

    assert record is not None
    assert (
        record.provider,
        record.model,
        record.user_id,
        record.workspace_id,
        record.seconds,
        record.counts_toward_budget,
    ) == ("gemini", "veo-3.0-generate-001", _OWNER, workspace_id, 8, False)
    assert record.accounted_at is None
    assert await service.find("video_unknown") is None


@pytest.mark.asyncio
async def test_a_record_that_cannot_be_written_is_logged_and_the_request_goes_on(async_db: AsyncSession) -> None:
    service = await _service(async_db)

    await service.record(
        video_id="video_orphan",
        provider="gemini",
        model="veo",
        user_id="nobody",
        api_key_id=None,
        workspace_id=None,
        seconds=None,
        counts_toward_budget=True,
    )

    assert await service.find("video_orphan") is None


@pytest.mark.asyncio
async def test_the_page_holds_the_newest_jobs_of_one_user_in_one_workspace(async_db: AsyncSession) -> None:
    service = await _service(async_db)
    workspace_id = await default_workspace_id(async_db)
    for video_id, user_id, placed in (
        ("video_a", _OWNER, True),
        ("video_b", _OWNER, False),
        ("video_c", _OTHER, True),
        ("video_d", _OWNER, True),
    ):
        await service.record(
            video_id=video_id,
            provider="gemini",
            model="veo",
            user_id=user_id,
            api_key_id=None,
            workspace_id=workspace_id if placed else None,
            seconds=None,
            counts_toward_budget=True,
        )

    owned = await service.page(user_id=_OWNER, workspace_id=None, limit=10)
    in_workspace = await service.page(user_id=_OWNER, workspace_id=workspace_id, limit=10)
    everything = await service.page(user_id=None, workspace_id=None, limit=2)

    assert {record.id for record in owned} == {"video_a", "video_b", "video_d"}
    assert {record.id for record in in_workspace} == {"video_a", "video_d"}
    assert len(everything) == 2
    assert [record.created_at for record in owned] == sorted((r.created_at for r in owned), reverse=True)


@pytest.mark.asyncio
async def test_a_completed_job_is_settled_by_one_caller_and_a_failed_settlement_frees_the_claim(
    async_db: AsyncSession,
) -> None:
    service = await _service(async_db)
    await service.record(
        video_id="video_done",
        provider="gemini",
        model="veo",
        user_id=_OWNER,
        api_key_id=None,
        workspace_id=None,
        seconds=4,
        counts_toward_budget=True,
    )
    failing = AsyncMock(side_effect=RuntimeError("the spend fold failed"))
    settle = AsyncMock()

    with pytest.raises(RuntimeError, match="spend fold"):
        await service.settle_once("video_done", failing)
    unclaimed = await service.find("video_done")
    # Read now: a later find refreshes the same session object in place.
    unclaimed_at = unclaimed.accounted_at if unclaimed is not None else "no record"
    first = await service.settle_once("video_done", settle)
    second = await service.settle_once("video_done", settle)
    claimed = await service.find("video_done")

    assert unclaimed_at is None
    assert (first, second) == (True, False)
    settle.assert_awaited_once()
    assert claimed is not None and claimed.accounted_at is not None
