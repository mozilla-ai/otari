"""The video job records: who started a job here, and billing a completed one once."""

import logging
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from any_llm.types.video import VideoJob, VideoOutput
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger as gateway_logger
from gateway.models.pricing import ModelPricing
from gateway.models.users import User
from gateway.repositories.inference import InferenceRepositories
from gateway.services.inference import CompletedVideoUsage, VideoJobService
from gateway.services.pricing import UsageCharge
from gateway.services.pricing_service import ResolvedPricing
from gateway.services.workspace_scope import default_workspace_id

_OWNER = "video-owner"
_OTHER = "video-other"
_ENDPOINT = "/v1/videos/completed"


class _Recorder:
    def __init__(self, fail: bool = False) -> None:
        self.record = AsyncMock(side_effect=RuntimeError("the row could not be written") if fail else None)
        self.recorded = AsyncMock()


def _per_second(price: float) -> ResolvedPricing:
    pricing = ModelPricing(
        model_key="gemini:veo-3.0-generate-001",
        input_price_per_million=Decimal(0),
        output_price_per_million=Decimal(0),
        unit="second",
        rates={"video": {"output": {"price": price, "unit": "second"}}},
    )
    return ResolvedPricing(pricing, "deployment", "gemini:veo-3.0-generate-001", None)


async def _service(
    async_db: AsyncSession,
    *,
    recorder: _Recorder | None = None,
    fold_spend: AsyncMock | None = None,
    rate: AsyncMock | None = None,
) -> VideoJobService:
    async_db.add_all([User(user_id=_OWNER), User(user_id=_OTHER)])
    await async_db.commit()
    uow = UnitOfWork(async_db)
    return VideoJobService(
        uow,
        InferenceRepositories.on(uow).video_jobs,
        usage=recorder or _Recorder(),
        fold_spend=fold_spend or AsyncMock(),
        rate=rate or AsyncMock(return_value=None),
    )


async def _record(service: VideoJobService, video_id: str, **overrides: Any) -> None:
    fields: dict[str, Any] = {
        "provider": "gemini",
        "model": "veo-3.0-generate-001",
        "user_id": _OWNER,
        "api_key_id": None,
        "workspace_id": None,
        "seconds": None,
        "counts_toward_budget": True,
        "resolution": None,
        "aspect_ratio": None,
        "input_video": None,
        "generate_audio": None,
        "created_at": datetime.now(UTC),
    }
    fields.update(overrides)
    await service.record(video_id=video_id, **fields)


def _usage(workspace_id: uuid.UUID, *, charge: UsageCharge | None, counts: bool = True) -> CompletedVideoUsage:
    return CompletedVideoUsage(
        video_id="video_done",
        workspace_id=workspace_id,
        api_key_id=None,
        user_id=_OWNER,
        model="veo-3.0-generate-001",
        provider="gemini",
        endpoint=_ENDPOINT,
        counts_toward_budget=counts,
        deployment_paid=False,
        charge=charge,
        pricing=None,
    )


@pytest.mark.asyncio
async def test_a_recorded_job_is_found_with_what_was_recorded(async_db: AsyncSession) -> None:
    service = await _service(async_db)
    workspace_id = await default_workspace_id(async_db)

    await _record(service, "video_1", workspace_id=workspace_id, seconds=8, counts_toward_budget=False)
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
async def test_a_record_that_cannot_be_written_is_raised_so_the_job_is_not_handed_out_as_started(
    async_db: AsyncSession,
) -> None:
    service = await _service(async_db)

    with pytest.raises(SQLAlchemyError):
        await _record(service, "video_orphan", user_id="nobody")

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
        await _record(service, video_id, user_id=user_id, workspace_id=workspace_id if placed else None)

    owned = await service.page(user_id=_OWNER, workspace_id=None, limit=10)
    in_workspace = await service.page(user_id=_OWNER, workspace_id=workspace_id, limit=10)
    everything = await service.page(user_id=None, workspace_id=None, limit=2)

    assert {record.id for record in owned} == {"video_a", "video_b", "video_d"}
    assert {record.id for record in in_workspace} == {"video_a", "video_d"}
    assert len(everything) == 2
    assert [record.created_at for record in owned] == sorted((r.created_at for r in owned), reverse=True)


@pytest.mark.asyncio
async def test_a_completed_job_is_settled_by_one_caller_who_records_it_and_folds_its_spend(
    async_db: AsyncSession,
) -> None:
    recorder, fold_spend = _Recorder(), AsyncMock()
    service = await _service(async_db, recorder=recorder, fold_spend=fold_spend)
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done", seconds=4)
    usage = _usage(workspace_id, charge=UsageCharge(Decimal("0.4"), {"seconds": 4}, []))
    charge_deployment = AsyncMock()

    first = await service.settle_once("video_done", usage, charge_deployment=charge_deployment)
    second = await service.settle_once("video_done", usage, charge_deployment=charge_deployment)
    claimed = await service.find("video_done")

    assert (first, second) == (True, False)
    charge_deployment.assert_awaited_once()
    recorder.record.assert_awaited_once_with(usage)
    recorder.recorded.assert_awaited_once_with(usage)
    fold_spend.assert_awaited_once_with(_OWNER, Decimal("0.4"))
    assert claimed is not None and claimed.accounted_at is not None


@pytest.mark.asyncio
async def test_a_settlement_that_fails_leaves_the_job_for_a_later_caller(async_db: AsyncSession) -> None:
    service = await _service(async_db, recorder=_Recorder(fail=True))
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done", seconds=4)
    usage = _usage(workspace_id, charge=UsageCharge(Decimal("0.4"), {"seconds": 4}, []))

    with pytest.raises(RuntimeError, match="could not be written"):
        await service.settle_once("video_done", usage)
    unclaimed = await service.find("video_done")

    assert unclaimed is not None and unclaimed.accounted_at is None
    service._usage.recorded.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("charge", "counts"),
    [(None, True), (UsageCharge(Decimal(0), {}, []), True), (UsageCharge(Decimal("0.4"), {"seconds": 4}, []), False)],
    ids=["unpriced", "free", "budget exempt"],
)
async def test_nothing_is_folded_for_a_job_with_no_cost_or_outside_the_budget(
    async_db: AsyncSession, charge: UsageCharge | None, counts: bool
) -> None:
    recorder, fold_spend = _Recorder(), AsyncMock()
    service = await _service(async_db, recorder=recorder, fold_spend=fold_spend)
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done")

    settled = await service.settle_once("video_done", _usage(workspace_id, charge=charge, counts=counts))

    assert settled is True
    recorder.record.assert_awaited_once()
    fold_spend.assert_not_awaited()


async def _settle(
    service: VideoJobService, workspace_id: uuid.UUID, *, reported: int | None, charged: float | None = None
) -> bool:
    record = await service.find("video_done")
    assert record is not None
    return await service.settle_completed(
        record,
        reported_seconds=reported,
        reported_cost=charged,
        endpoint=_ENDPOINT,
        workspace_id=workspace_id,
        organization_id=None,
        deployment_paid=False,
    )


@pytest.mark.asyncio
async def test_a_completed_job_is_priced_for_the_seconds_reported_at_the_rate_found(async_db: AsyncSession) -> None:
    recorder, fold_spend = _Recorder(), AsyncMock()
    rate = AsyncMock(return_value=_per_second(0.1))
    service = await _service(async_db, recorder=recorder, fold_spend=fold_spend, rate=rate)
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done", seconds=4, api_key_id=None)

    first = await _settle(service, workspace_id, reported=6)
    second = await _settle(service, workspace_id, reported=6)

    assert (first, second) == (True, False)
    rate.assert_awaited_with("gemini", "veo-3.0-generate-001", organization_id=None, deployment_paid=False)
    usage = recorder.record.await_args_list[0].args[0]
    assert (usage.cost, usage.video_id, usage.endpoint, usage.user_id) == (
        Decimal("0.6"),
        "video_done",
        _ENDPOINT,
        _OWNER,
    )
    assert usage.charge is not None and usage.charge.meters == {"seconds": 6}
    fold_spend.assert_awaited_once_with(_OWNER, Decimal("0.6"))


@pytest.mark.asyncio
async def test_a_completed_job_with_no_reported_length_is_priced_for_the_length_asked(async_db: AsyncSession) -> None:
    recorder = _Recorder()
    service = await _service(async_db, recorder=recorder, rate=AsyncMock(return_value=_per_second(0.1)))
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done", seconds=5)

    assert await _settle(service, workspace_id, reported=None) is True
    assert recorder.record.await_args_list[0].args[0].cost == Decimal("0.5")


@pytest.mark.asyncio
async def test_a_completed_job_with_no_rate_stays_unclaimed_for_a_later_read(async_db: AsyncSession) -> None:
    recorder = _Recorder()
    service = await _service(async_db, recorder=recorder)
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done", seconds=5)

    settled = await _settle(service, workspace_id, reported=5)
    record = await service.find("video_done")

    assert settled is False
    recorder.record.assert_not_awaited()
    assert record is not None and record.accounted_at is None


@pytest.mark.asyncio
async def test_a_completed_job_whose_length_nobody_knows_is_billed_what_the_provider_charged(
    async_db: AsyncSession,
) -> None:
    recorder, fold_spend = _Recorder(), AsyncMock()
    service = await _service(async_db, recorder=recorder, fold_spend=fold_spend)
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done")

    settled = await _settle(service, workspace_id, reported=None, charged=0.3899126)

    assert settled is True
    usage = recorder.record.await_args_list[0].args[0]
    assert usage.cost == Decimal("0.389913")
    assert usage.charge is not None and (dict(usage.charge.meters), usage.charge.lines) == ({}, [])
    assert usage.pricing is None
    fold_spend.assert_awaited_once_with(_OWNER, Decimal("0.389913"))


@pytest.mark.asyncio
async def test_a_completed_job_whose_length_nobody_knows_and_with_no_charge_is_billed_for_the_assumed_length(
    async_db: AsyncSession,
) -> None:
    recorder, fold_spend = _Recorder(), AsyncMock()
    service = await _service(
        async_db, recorder=recorder, fold_spend=fold_spend, rate=AsyncMock(return_value=_per_second(0.1))
    )
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done")

    settled = await _settle(service, workspace_id, reported=None)

    assert settled is True
    usage = recorder.record.await_args_list[0].args[0]
    assert usage.cost == Decimal("0.8")
    assert usage.charge is not None and usage.charge.meters == {"seconds": 8}
    fold_spend.assert_awaited_once_with(_OWNER, Decimal("0.8"))


@pytest.mark.asyncio
async def test_the_length_asked_prices_a_job_before_the_providers_charge(async_db: AsyncSession) -> None:
    recorder = _Recorder()
    service = await _service(async_db, recorder=recorder, rate=AsyncMock(return_value=_per_second(0.1)))
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done", seconds=5)

    assert await _settle(service, workspace_id, reported=None, charged=0.39) is True
    assert recorder.record.await_args_list[0].args[0].cost == Decimal("0.5")


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.asyncio
@pytest.mark.parametrize("charged", [0.25, 0.14, 0.0])
async def test_a_rate_far_from_the_providers_charge_is_warned_of_and_still_bills(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture, charged: float
) -> None:
    recorder = _Recorder()
    service = await _service(async_db, recorder=recorder, rate=AsyncMock(return_value=_per_second(0.033)))
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done", seconds=5)
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger=gateway_logger.name)
    try:
        assert await _settle(service, workspace_id, reported=5, charged=charged) is True
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert recorder.record.await_args_list[0].args[0].cost == Decimal("0.165")
    [warning] = _warnings(caplog)
    assert "video_done" in warning and "0.165" in warning and f"{Decimal(str(charged)):.6f}" in warning


@pytest.mark.asyncio
@pytest.mark.parametrize("charged", [0.17, 0.15, 0.18, None, float("nan")])
async def test_a_rate_near_the_providers_charge_or_with_none_to_check_is_not_warned_of(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture, charged: float | None
) -> None:
    service = await _service(async_db, rate=AsyncMock(return_value=_per_second(0.033)))
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done", seconds=5)
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger=gateway_logger.name)
    try:
        assert await _settle(service, workspace_id, reported=5, charged=charged) is True
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_a_completed_job_with_no_length_no_charge_and_no_rate_stays_unclaimed(async_db: AsyncSession) -> None:
    recorder = _Recorder()
    service = await _service(async_db, recorder=recorder)
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done")

    settled = await _settle(service, workspace_id, reported=None)
    record = await service.find("video_done")

    assert settled is False
    recorder.record.assert_not_awaited()
    assert record is not None and record.accounted_at is None


@pytest.mark.asyncio
async def test_the_deployment_is_charged_the_cost_before_the_row_is_written(async_db: AsyncSession) -> None:
    calls: list[object] = []
    recorder = _Recorder()
    recorder.record = AsyncMock(side_effect=lambda usage: calls.append("row"))
    service = await _service(async_db, recorder=recorder, rate=AsyncMock(return_value=_per_second(0.1)))
    workspace_id = await default_workspace_id(async_db)
    await _record(service, "video_done", seconds=4)
    record = await service.find("video_done")
    assert record is not None

    async def charge(cost: Decimal | None) -> None:
        calls.append(("charge", cost))

    settled = await service.settle_completed(
        record,
        reported_seconds=4,
        reported_cost=None,
        endpoint=_ENDPOINT,
        workspace_id=workspace_id,
        organization_id=uuid.uuid4(),
        deployment_paid=True,
        charge_deployment=charge,
    )

    assert settled is True
    assert calls == [("charge", Decimal("0.4")), "row"]
    assert recorder.record.await_args_list[0].args[0].deployment_paid is True


def test_a_job_already_accounted_is_not_settled_again() -> None:
    service = VideoJobService(
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        usage=_Recorder(),
        fold_spend=AsyncMock(),
        rate=AsyncMock(),
    )
    record = SimpleNamespace(accounted_at="earlier")

    import asyncio

    settled = asyncio.run(
        service.settle_completed(
            record,  # type: ignore[arg-type]
            reported_seconds=4,
            reported_cost=None,
            endpoint=_ENDPOINT,
            workspace_id=uuid.uuid4(),
            organization_id=None,
            deployment_paid=False,
        )
    )

    assert settled is False


@pytest.mark.asyncio
async def test_a_completed_job_is_remembered_with_its_length_and_its_video(async_db: AsyncSession) -> None:
    service = await _service(async_db)
    await _record(service, "video_done", seconds=4)
    record = await service.find("video_done")
    assert record is not None
    assert service.remembered(record) is None
    job = VideoJob(
        id="video_done",
        model="veo-3.0-generate-001",
        status="completed",
        seconds=5,
        outputs=[VideoOutput(b64_data="AAAA"), VideoOutput(url="https://cdn.example/v.webm", mime_type="video/webm")],
        cost=0.3899126,
    )

    await service.remember_completed(record, job)
    remembered = service.remembered(await service.find("video_done"))  # type: ignore[arg-type]

    assert remembered is not None
    assert (remembered.id, remembered.status, remembered.seconds) == ("video_done", "completed", 5)
    assert remembered.cost == pytest.approx(0.389913)
    assert [(o.url, o.mime_type) for o in remembered.outputs] == [("https://cdn.example/v.webm", "video/webm")]
    assert remembered.completed_at is not None and remembered.created_at is not None


@pytest.mark.asyncio
async def test_a_providers_charge_past_what_a_cost_holds_is_not_remembered(async_db: AsyncSession) -> None:
    service = await _service(async_db)
    await _record(service, "video_done")
    record = await service.find("video_done")
    assert record is not None

    await service.remember_completed(
        record, VideoJob(id="video_done", model="veo", status="completed", outputs=[], cost=1e13)
    )
    remembered = service.remembered(await service.find("video_done"))  # type: ignore[arg-type]

    assert remembered is not None and remembered.cost is None


@pytest.mark.asyncio
async def test_a_completed_job_with_only_inline_data_is_remembered_without_an_output(async_db: AsyncSession) -> None:
    service = await _service(async_db)
    await _record(service, "video_done")
    record = await service.find("video_done")
    assert record is not None

    await service.remember_completed(record, VideoJob(id="video_done", model="veo", status="completed", outputs=[]))
    remembered = service.remembered(await service.find("video_done"))  # type: ignore[arg-type]

    assert remembered is not None and remembered.outputs == [] and remembered.seconds is None
    assert remembered.cost is None
