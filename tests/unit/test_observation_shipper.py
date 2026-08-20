"""The observation shipper's contract: it may lose records, never a request.

A gateway in hybrid mode has no local database, and observation records are
produced mid-request (the tool loop records a round and immediately calls the
model again). So the transport is a bounded in-memory queue with a periodic
background flush, and every one of its failure modes has to end in a lost
record rather than in a slower or failed request. The tests below pin exactly
that: emitting never awaits network I/O, an overflowing queue drops instead of
applying backpressure, any flush failure is swallowed, and the loss is counted.

HTTP is stubbed with ``httpx.MockTransport``, so nothing here needs a platform.
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable

import httpx
import pytest

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.metrics import REGISTRY
from gateway.services import observation_shipper as shipper_module
from gateway.services.observation_shipper import (
    OBSERVATION_INGEST_PATH,
    NullObservationShipper,
    PlatformObservationShipper,
    create_observation_shipper,
)

_BASE_URL = "http://platform.test/api/v1"
_TOKEN = "gw_test_token"


class _Ingest:
    """A stand-in platform ingest endpoint that records what it was sent."""

    def __init__(self, *, status_code: int = 204, error: Exception | None = None) -> None:
        self.status_code = status_code
        self.error = error
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return httpx.Response(self.status_code)

    @property
    def batches(self) -> list[list[dict[str, object]]]:
        payloads = [json.loads(request.content.decode()) for request in self.requests]
        return [payload["records"] for payload in payloads]

    @property
    def records(self) -> list[dict[str, object]]:
        return [record for batch in self.batches for record in batch]


def _make_shipper(
    ingest: _Ingest,
    *,
    max_queue: int = 1000,
    max_batch: int = 100,
    flush_interval_seconds: float = 0.01,
) -> PlatformObservationShipper:
    """Build a shipper wired to ``ingest``, flushing fast enough for a test."""
    return PlatformObservationShipper(
        base_url=_BASE_URL,
        gateway_token=_TOKEN,
        max_queue=max_queue,
        max_batch=max_batch,
        flush_interval_seconds=flush_interval_seconds,
        transport=httpx.MockTransport(ingest),
    )


def _round_record(index: int = 0) -> dict[str, object]:
    """A loop-round record. Only ``kind`` matters to the transport."""
    return {"kind": "loop_round", "round_index": index, "fingerprint": f"9f2ac{index}"}


def _counter_record() -> dict[str, object]:
    return {"kind": "request_counter", "request_hash": "1bd0f", "entered_loop": True}


def _count(result: str) -> float:
    return REGISTRY.get_sample_value("gateway_observation_records_total", {"result": result}) or 0.0


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition was not met within the timeout")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_emit_awaits_nothing() -> None:
    """The strongest form of "nothing on the request path awaits a flush".

    ``emit`` is a plain function, so a producer physically cannot await network
    I/O through it, whatever the platform is doing.
    """
    assert not inspect.iscoroutinefunction(PlatformObservationShipper.emit)

    ingest = _Ingest(error=AssertionError("emit must not post"))
    shipper = _make_shipper(ingest, flush_interval_seconds=60.0)

    shipper.emit(_round_record())

    assert shipper.queue_depth == 1
    assert ingest.requests == []


@pytest.mark.asyncio
async def test_a_full_queue_drops_the_new_record_and_counts_it() -> None:
    ingest = _Ingest()
    shipper = _make_shipper(ingest, max_queue=2, flush_interval_seconds=60.0)
    before_dropped = _count("dropped_queue_full")
    before_queued = _count("queued")

    shipper.emit(_round_record(0))
    shipper.emit(_round_record(1))
    shipper.emit(_round_record(2))

    assert shipper.queue_depth == 2
    assert _count("dropped_queue_full") - before_dropped == 1.0
    assert _count("queued") - before_queued == 2.0

    # The queue drops the arriving record, keeping the ones already accepted.
    await shipper.stop()
    assert [record["round_index"] for record in ingest.records] == [0, 1]


@pytest.mark.asyncio
async def test_overflow_never_raises_at_the_producer() -> None:
    """Backpressure would reach the tool loop as either a block or a raise.

    A queue this small overflows on all but the first record, so an
    implementation that awaited ``put`` would hang here and one that let
    ``QueueFull`` escape would fail the loop's round instead of losing a record.
    """
    shipper = _make_shipper(_Ingest(), max_queue=1, flush_interval_seconds=60.0)

    async with asyncio.timeout(2):
        for index in range(500):
            shipper.emit(_round_record(index))

    assert shipper.queue_depth == 1


@pytest.mark.asyncio
async def test_records_ship_to_the_ingest_endpoint_with_the_gateway_token() -> None:
    ingest = _Ingest()
    shipper = _make_shipper(ingest)
    before_shipped = _count("shipped")

    await shipper.start()
    try:
        shipper.emit(_round_record())
        await _wait_until(lambda: _count("shipped") - before_shipped == 1.0)
    finally:
        await shipper.stop()

    request = ingest.requests[0]
    assert str(request.url) == f"{_BASE_URL}{OBSERVATION_INGEST_PATH}"
    assert request.method == "POST"
    assert request.headers["X-Gateway-Token"] == _TOKEN
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content.decode()) == {"records": [_round_record()]}


@pytest.mark.asyncio
async def test_a_trailing_slash_on_the_base_url_does_not_double_up() -> None:
    ingest = _Ingest()
    shipper = PlatformObservationShipper(
        base_url=f"{_BASE_URL}/",
        gateway_token=_TOKEN,
        flush_interval_seconds=0.01,
        transport=httpx.MockTransport(ingest),
    )

    await shipper.start()
    try:
        shipper.emit(_round_record())
        await _wait_until(lambda: len(ingest.requests) == 1)
    finally:
        await shipper.stop()

    assert str(ingest.requests[0].url) == f"{_BASE_URL}{OBSERVATION_INGEST_PATH}"


@pytest.mark.asyncio
async def test_both_record_kinds_travel_in_one_batch() -> None:
    """A round record and a request-counter record share the queue and the batch.

    The transport is deliberately blind to the shapes: ``kind`` discriminates
    them for the platform, not here.
    """
    ingest = _Ingest()
    shipper = _make_shipper(ingest)

    await shipper.start()
    try:
        shipper.emit(_round_record(0))
        shipper.emit(_counter_record())
        await _wait_until(lambda: len(ingest.records) == 2)
    finally:
        await shipper.stop()

    assert [record["kind"] for record in ingest.batches[0]] == ["loop_round", "request_counter"]


@pytest.mark.asyncio
async def test_a_batch_is_capped_at_max_batch() -> None:
    ingest = _Ingest()
    shipper = _make_shipper(ingest, max_batch=2)

    await shipper.start()
    try:
        for index in range(5):
            shipper.emit(_round_record(index))
        await _wait_until(lambda: len(ingest.records) == 5)
    finally:
        await shipper.stop()

    assert [len(batch) for batch in ingest.batches] == [2, 2, 1]
    assert [record["round_index"] for record in ingest.records] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_a_steady_trickle_does_not_turn_the_flusher_into_a_post_loop() -> None:
    """The drain stops at a partial batch instead of chasing the queue empty.

    A producer only has to queue one record per round trip to keep the next batch
    non-empty, so a drain that ran until the queue was empty would never return:
    the flusher would stop sleeping altogether and POST tiny batches back to back,
    multiplying the request rate to the platform that batching exists to hold down.
    """
    ingest = _Ingest()

    async def _slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.02)  # a platform that answers, just not instantly
        return ingest(request)

    shipper = PlatformObservationShipper(
        base_url=_BASE_URL,
        gateway_token=_TOKEN,
        max_batch=500,
        flush_interval_seconds=0.1,
        transport=httpx.MockTransport(_slow),
    )

    await shipper.start()
    try:
        # ~10x more records than there are flush intervals, all well under one batch.
        for _ in range(50):
            shipper.emit(_round_record())
            await asyncio.sleep(0.01)
    finally:
        await shipper.stop()

    # 0.5s of production over a 0.1s interval is ~5 ticks; a POST per round trip
    # would instead be ~25. The bound is loose on purpose: the point is that the
    # rate tracks the interval, not the record count.
    assert len(ingest.requests) <= 10, [len(batch) for batch in ingest.batches]
    assert len(ingest.records) == 50


@pytest.mark.parametrize(
    ("status_code", "error"),
    [
        (204, httpx.ReadTimeout("platform too slow")),
        (204, httpx.ConnectError("platform unreachable")),
        (500, None),
        (404, None),
        (401, None),
        (204, RuntimeError("something unforeseen")),
    ],
    ids=["timeout", "unreachable", "http_500", "http_404", "http_401", "unexpected_error"],
)
@pytest.mark.asyncio
async def test_a_failed_flush_loses_records_and_keeps_shipping(status_code: int, error: Exception | None) -> None:
    """Every flush failure is swallowed, counted, and survivable.

    The second half is what makes this more than "no exception escaped": the
    background flusher has to still be running afterwards, or one unreachable
    platform would end observation shipping for the life of the process.
    """
    ingest = _Ingest(status_code=status_code, error=error)
    shipper = _make_shipper(ingest)
    before = _count("dropped_flush_failed")

    await shipper.start()
    try:
        shipper.emit(_round_record(0))
        await _wait_until(lambda: _count("dropped_flush_failed") - before == 1.0)

        # The platform recovers; the flusher is still alive to notice.
        ingest.error = None
        ingest.status_code = 204
        shipper.emit(_round_record(1))
        await _wait_until(lambda: any(record["round_index"] == 1 for record in ingest.records))
    finally:
        await shipper.stop()


@pytest.mark.asyncio
async def test_shutdown_flushes_what_is_queued() -> None:
    """Nothing has flushed yet: the interval is far longer than the test."""
    ingest = _Ingest()
    shipper = _make_shipper(ingest, flush_interval_seconds=60.0)
    before_shipped = _count("shipped")

    await shipper.start()
    for index in range(3):
        shipper.emit(_round_record(index))
    assert ingest.requests == []

    await shipper.stop()

    assert [record["round_index"] for record in ingest.records] == [0, 1, 2]
    assert _count("shipped") - before_shipped == 3.0
    assert shipper.queue_depth == 0


@pytest.mark.asyncio
async def test_shutdown_drains_in_batches() -> None:
    ingest = _Ingest()
    shipper = _make_shipper(ingest, max_batch=2, flush_interval_seconds=60.0)

    await shipper.start()
    for index in range(5):
        shipper.emit(_round_record(index))
    await shipper.stop()

    assert [len(batch) for batch in ingest.batches] == [2, 2, 1]


@pytest.mark.asyncio
async def test_shutdown_is_safe_without_a_start() -> None:
    """Startup can fail between constructing the shipper and starting it."""
    ingest = _Ingest()
    shipper = _make_shipper(ingest, flush_interval_seconds=60.0)
    shipper.emit(_round_record())

    await shipper.stop()

    assert len(ingest.records) == 1


@pytest.mark.asyncio
async def test_shutdown_is_bounded_when_the_backlog_outlasts_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow platform and a deep queue must not hold the process open.

    The budget covers the whole drain, not one batch: without it the drain is
    worst-case (queue / batch) POSTs long, each waiting out its own timeout, which
    outlasts any container stop grace and blocks the lifespan behind an
    observability stream. Every POST here succeeds well inside its own timeout, so
    only the total is what gives up.
    """
    posts = 0

    async def _slow(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        posts += 1
        await asyncio.sleep(0.03)
        return httpx.Response(204)

    monkeypatch.setattr(shipper_module, "_SHUTDOWN_DRAIN_TIMEOUT_SECONDS", 0.1)
    shipper = PlatformObservationShipper(
        base_url=_BASE_URL,
        gateway_token=_TOKEN,
        max_batch=1,  # 20 records is 20 POSTs, ~0.6s of work against a 0.1s budget
        flush_interval_seconds=60.0,
        timeout_seconds=0.05,
        transport=httpx.MockTransport(_slow),
    )
    before_dropped = _count("dropped_shutdown")
    before_shipped = _count("shipped")

    await shipper.start()
    for index in range(20):
        shipper.emit(_round_record(index))

    async with asyncio.timeout(5):
        await shipper.stop()

    shipped = _count("shipped") - before_shipped
    dropped = _count("dropped_shutdown") - before_dropped
    assert posts > 0, "the drain should have shipped what it had time for"
    assert dropped > 0, "and given up on the rest rather than run to completion"
    assert shipped + dropped == 20, "every record is either shipped or counted as lost"
    assert shipper.queue_depth == 0


@pytest.mark.asyncio
async def test_shutdown_is_bounded_when_the_platform_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung platform must not hold the process open.

    The budget is the floor or one batch's timeout, whichever is larger, so a
    per-batch timeout below the floor leaves the floor in charge.
    """
    hang_started = asyncio.Event()

    async def _hang(request: httpx.Request) -> httpx.Response:
        hang_started.set()
        await asyncio.sleep(30)
        return httpx.Response(204)  # pragma: no cover - the drain gives up first

    monkeypatch.setattr(shipper_module, "_SHUTDOWN_DRAIN_TIMEOUT_SECONDS", 0.1)
    shipper = PlatformObservationShipper(
        base_url=_BASE_URL,
        gateway_token=_TOKEN,
        flush_interval_seconds=60.0,
        timeout_seconds=0.05,
        transport=httpx.MockTransport(_hang),
    )
    before = _count("dropped_shutdown")

    await shipper.start()
    shipper.emit(_round_record(0))
    shipper.emit(_round_record(1))

    async with asyncio.timeout(5):
        await shipper.stop()

    assert hang_started.is_set()
    assert _count("dropped_shutdown") - before == 2.0
    assert shipper.queue_depth == 0


def test_the_drain_budget_covers_at_least_one_batch_timeout() -> None:
    """An operator widening the per-batch timeout widens the drain with it.

    Otherwise a legal ``observation_timeout_ms`` above the budget would make the
    final flush structurally unable to complete even one batch, and the whole
    queue would be abandoned on every shutdown.
    """
    generous = PlatformObservationShipper(
        base_url=_BASE_URL, gateway_token=_TOKEN, timeout_seconds=30.0
    )
    assert generous._shutdown_drain_seconds == 30.0

    modest = PlatformObservationShipper(base_url=_BASE_URL, gateway_token=_TOKEN, timeout_seconds=1.0)
    assert modest._shutdown_drain_seconds == shipper_module._SHUTDOWN_DRAIN_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_a_flush_in_flight_at_shutdown_is_delivered_not_cancelled() -> None:
    """Cancelling the flusher first would throw away a batch it was mid-way through.

    The platform is healthy and the drain budget untouched, so those records have
    somewhere to go; a shutdown that discards them loses up to ``max_batch``
    records on every graceful restart of a busy replica.
    """
    in_flight = asyncio.Event()
    release = asyncio.Event()
    ingest = _Ingest()

    async def _slow(request: httpx.Request) -> httpx.Response:
        in_flight.set()
        await release.wait()
        return ingest(request)

    shipper = PlatformObservationShipper(
        base_url=_BASE_URL,
        gateway_token=_TOKEN,
        flush_interval_seconds=0.01,
        transport=httpx.MockTransport(_slow),
    )
    before_shipped = _count("shipped")
    before_dropped = _count("dropped_shutdown")

    await shipper.start()
    for index in range(3):
        shipper.emit(_round_record(index))
    await asyncio.wait_for(in_flight.wait(), timeout=5)

    stop = asyncio.create_task(shipper.stop())
    await asyncio.sleep(0.05)  # stop() lands while the POST is still open
    release.set()
    async with asyncio.timeout(5):
        await stop

    assert _count("shipped") - before_shipped == 3.0
    assert _count("dropped_shutdown") - before_dropped == 0.0
    assert [record["round_index"] for record in ingest.records] == [0, 1, 2]


@pytest.mark.asyncio
async def test_the_shutdown_warning_counts_the_records_lost_in_flight(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The queue's remainder alone under-reports the loss.

    A batch cancelled mid-POST is already off the queue, so a warning counting
    only what is left behind reads ``0 record(s) dropped`` on the common path,
    while the counter records the real loss.
    """

    async def _hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(204)  # pragma: no cover - the drain gives up first

    monkeypatch.setattr(shipper_module, "_SHUTDOWN_DRAIN_TIMEOUT_SECONDS", 0.1)
    shipper = PlatformObservationShipper(
        base_url=_BASE_URL,
        gateway_token=_TOKEN,
        max_batch=4,
        flush_interval_seconds=60.0,
        timeout_seconds=0.1,
        transport=httpx.MockTransport(_hang),
    )
    before = _count("dropped_shutdown")

    await shipper.start()
    for index in range(4):
        shipper.emit(_round_record(index))

    # The ``gateway`` logger does not propagate (log_config sets propagate=False),
    # so caplog's handler goes on it directly rather than on root.
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger=gateway_logger.name)
    try:
        async with asyncio.timeout(5):
            await shipper.stop()
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert _count("dropped_shutdown") - before == 4.0
    gave_up = [record.getMessage() for record in caplog.records if "gave up draining" in record.getMessage()]
    assert gave_up and "4 record(s) dropped" in gave_up[0], gave_up


@pytest.mark.asyncio
async def test_shutdown_stays_bounded_when_records_outlive_the_in_flight_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The budget has to cover both halves of the shutdown, not just the first.

    Waiting on the flusher with a bare ``await task`` hands the budget's
    cancellation to the flusher, which handles CancelledError and so completes
    normally: the timeout is spent without ever raising, and the drain that
    follows runs with a deadline that can no longer fire. That only shows up when
    records are still queued behind the in-flight batch, since an empty queue
    makes the second drain a no-op and hides it.
    """
    hang = asyncio.Event()

    async def _hang_forever(request: httpx.Request) -> httpx.Response:
        hang.set()
        await asyncio.sleep(30)
        return httpx.Response(204)  # pragma: no cover - the drain gives up first

    monkeypatch.setattr(shipper_module, "_SHUTDOWN_DRAIN_TIMEOUT_SECONDS", 0.3)
    shipper = PlatformObservationShipper(
        base_url=_BASE_URL,
        gateway_token=_TOKEN,
        max_batch=5,
        flush_interval_seconds=0.05,
        timeout_seconds=0.1,
        transport=httpx.MockTransport(_hang_forever),
    )
    before = _count("dropped_shutdown")

    await shipper.start()
    for index in range(5):
        shipper.emit(_round_record(index))
    await asyncio.wait_for(hang.wait(), timeout=5)  # that batch is now in flight
    for index in range(5, 15):
        shipper.emit(_round_record(index))  # and these are queued behind it

    loop = asyncio.get_running_loop()
    started = loop.time()
    await shipper.stop()
    elapsed = loop.time() - started

    assert elapsed < 3.0, f"stop() ran {elapsed:.1f}s against a {shipper._shutdown_drain_seconds}s budget"
    assert _count("dropped_shutdown") - before == 15.0
    assert shipper.queue_depth == 0


@pytest.mark.asyncio
async def test_stop_is_idempotent() -> None:
    shipper = _make_shipper(_Ingest())

    await shipper.start()
    await shipper.stop()
    await shipper.stop()


@pytest.mark.asyncio
async def test_the_null_shipper_discards_everything() -> None:
    """Producers emit unconditionally, so standalone mode needs a silent sink."""
    shipper = NullObservationShipper()

    await shipper.start()
    shipper.emit(_round_record())
    await shipper.stop()


def test_create_observation_shipper_is_null_in_standalone_mode() -> None:
    config = GatewayConfig(mode="standalone")

    assert isinstance(create_observation_shipper(config), NullObservationShipper)


def test_create_observation_shipper_is_null_without_a_platform_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive: ``load_config`` fills the base URL in, a hand-built config may not."""
    monkeypatch.setenv("OTARI_AI_TOKEN", _TOKEN)
    config = GatewayConfig(mode="hybrid", platform={})

    assert config.is_hybrid_mode
    assert isinstance(create_observation_shipper(config), NullObservationShipper)


def test_create_observation_shipper_reads_the_platform_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", _TOKEN)
    config = GatewayConfig(
        mode="hybrid",
        platform={
            "base_url": _BASE_URL,
            "observation_max_queue": 42,
            "observation_max_batch": 7,
            "observation_flush_interval_ms": 1500,
            "observation_timeout_ms": 2500,
        },
    )

    shipper = create_observation_shipper(config)

    assert isinstance(shipper, PlatformObservationShipper)
    assert shipper.max_queue == 42
    assert shipper.max_batch == 7
    assert shipper.flush_interval_seconds == 1.5
    assert shipper.timeout_seconds == 2.5


def test_create_observation_shipper_defaults_the_unset_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", _TOKEN)
    config = GatewayConfig(mode="hybrid", platform={"base_url": _BASE_URL})

    shipper = create_observation_shipper(config)

    assert isinstance(shipper, PlatformObservationShipper)
    assert shipper.max_queue == shipper_module.DEFAULT_MAX_QUEUE
    assert shipper.max_batch == shipper_module.DEFAULT_MAX_BATCH
