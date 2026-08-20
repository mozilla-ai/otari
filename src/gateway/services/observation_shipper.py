"""Transport for the observation records the gateway reports to the platform.

A gateway in hybrid mode has no local database by design, so observation
records have nowhere on the box to go, and they are produced *mid-request*: the
tool loop records a round and immediately calls the model again. Emitting one
over HTTP inline would put that network I/O on the request path, competing for
the same event loop and connection pool as the request it is measuring. An
observability stream that can degrade the request path is worse than no stream.

So the shape is a bounded in-memory queue plus a periodic background flush,
batching records to the platform's ingest endpoint. Three properties hold:

- **Bounded, drop on overflow.** ``emit`` is a plain function whose whole body
  is an in-memory put, so a producer physically cannot await a flush. When the
  queue is full the arriving record is discarded and counted; the loop is never
  made to wait on a measurement.
- **Batched, not per request.** One POST per flush interval for the whole
  process, however many requests are in flight, plus extra POSTs only while a
  backlog is clearing in full batches. The per-request counter runs unsampled so
  recurrence counts are trustworthy, which would otherwise double the gateway's
  outbound request rate to the platform at every level of load.
- **Any error loses records, never a request.** A timeout, a 5xx, a peer that
  does not implement the endpoint, an unreachable platform: all of them mean
  records vanish, counted, and the request completes exactly as it does today.

Deliberately not folded into the existing ``/gateway/usage`` report, which
already fires once per request and would carry these happily: usage reports are
billing-relevant and retried, observations are disposable. Coupling them means
either retrying billing data because measurement data failed, or dropping
billing data because the payload grew.

Modeled on ``BatchLogWriter`` in :mod:`gateway.services.log_writer` for the
queue-plus-flush-task shape, with one deliberate difference: that queue is
unbounded, and this one must not be.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from gateway.log_config import logger
from gateway.metrics import (
    OBSERVATION_BATCH_SIZE,
    OBSERVATION_FLUSH_DURATION,
    OBSERVATION_QUEUE_DEPTH,
    OBSERVATION_RECORDS,
)

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

ObservationRecord = dict[str, Any]
"""One record as the producer built it. The transport never inspects it: both
kinds travel in the same batch, discriminated for the platform by ``kind``."""

OBSERVATION_INGEST_PATH = "/gateway/loop-observations"

# Records are fixed-size now that arguments are recorded as shape plus per-value
# hashes, so the queue bound is a capacity decision rather than a source of bias:
# 10k records is a bounded amount of memory and absorbs a large production burst
# between two flushes. It does not buy outage tolerance, since a batch that fails
# to POST is dropped rather than requeued. All four are operator-tunable through
# the platform config block.
DEFAULT_MAX_QUEUE = 10_000
DEFAULT_MAX_BATCH = 500
DEFAULT_FLUSH_INTERVAL_MS = 5_000
DEFAULT_TIMEOUT_MS = 5_000

# Floor for the shutdown drain's total budget, across however many batches it
# takes. Unbounded, the drain is worst-case (queue / batch) POSTs long, each
# waiting out its own timeout, which outlasts any container stop grace: uvicorn's
# lifespan would then block on an observability stream. Whatever is still queued
# when the budget runs out is dropped, and counted as such. The effective budget
# is this floor or one worst-case batch, whichever is larger, so an operator who
# widens ``observation_timeout_ms`` past it does not thereby make the final flush
# unable to complete a single batch.
_SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 5.0

# How long to wait for the flusher to acknowledge a cancellation once the drain
# budget is spent. Short because nothing is left to ship by then.
_TASK_CANCEL_TIMEOUT_SECONDS = 1.0


class ObservationShipper(Protocol):
    """What a producer needs: emit and forget."""

    def emit(self, record: ObservationRecord) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class NullObservationShipper:
    """Discards every record.

    Bound in standalone mode, where there is no platform to ship to, so
    producers can emit unconditionally instead of testing the mode themselves.
    """

    def emit(self, record: ObservationRecord) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class PlatformObservationShipper:
    """Queue observation records and flush them to the platform in batches.

    ``transport`` is httpx's own injection point and exists for tests; leaving
    it unset uses the default network transport.
    """

    def __init__(
        self,
        *,
        base_url: str,
        gateway_token: str,
        max_queue: int = DEFAULT_MAX_QUEUE,
        max_batch: int = DEFAULT_MAX_BATCH,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_MS / 1000,
        timeout_seconds: float = DEFAULT_TIMEOUT_MS / 1000,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}{OBSERVATION_INGEST_PATH}"
        # The same machine identity the usage report and the resolve endpoints
        # carry. A second header for one identity is worse than one shared one.
        self._headers = {"X-Gateway-Token": gateway_token}
        self._queue: asyncio.Queue[ObservationRecord] = asyncio.Queue(maxsize=max_queue)
        self._max_queue = max_queue
        self._max_batch = max_batch
        self._flush_interval_seconds = flush_interval_seconds
        self._timeout_seconds = timeout_seconds
        # A budget below one batch's timeout could not ship even one worst-case
        # batch, so a widened per-batch timeout widens the drain with it.
        self._shutdown_drain_seconds = max(_SHUTDOWN_DRAIN_TIMEOUT_SECONDS, timeout_seconds)
        self._transport = transport
        self._task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        # Set to ask the flusher to finish its current flush and exit, so a POST
        # in flight at shutdown is delivered rather than cancelled.
        self._stopping = asyncio.Event()
        # Records this stop() lost inside a flush that was cancelled mid-POST.
        # They are already off the queue, so the queue's remainder alone would
        # under-report the loss in the warning below.
        self._dropped_in_flight = 0

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def max_queue(self) -> int:
        return self._max_queue

    @property
    def max_batch(self) -> int:
        return self._max_batch

    @property
    def flush_interval_seconds(self) -> float:
        return self._flush_interval_seconds

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    def emit(self, record: ObservationRecord) -> None:
        """Hand a record to the queue, or drop it. Never blocks, never raises.

        Synchronous on purpose: a coroutine here would invite an ``await`` that
        reaches the network, and the whole point is that the producer's next
        statement runs regardless of what the platform is doing.
        """
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            OBSERVATION_RECORDS.labels(result="dropped_queue_full").inc()
            return
        OBSERVATION_RECORDS.labels(result="queued").inc()
        OBSERVATION_QUEUE_DEPTH.set(self._queue.qsize())

    async def start(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop flushing, then ship what is queued within a bounded budget.

        The flusher is asked to stop rather than cancelled outright: cancelling it
        would abort a POST already in flight and discard that whole batch, even
        against a healthy platform with the budget untouched. It is cancelled only
        if the budget runs out, which is the case the accounting below is for.
        """
        task, self._task = self._task, None
        self._dropped_in_flight = 0
        self._stopping.set()
        deadline = time.monotonic() + self._shutdown_drain_seconds
        try:
            if task is not None:
                # ``asyncio.wait`` rather than ``await task``: awaiting a task
                # directly hands a timeout's cancellation to that task, and the
                # flusher handles CancelledError, so it would complete normally,
                # the timeout would be spent without ever raising, and whatever
                # came next would run with a deadline that can no longer fire.
                # ``wait`` leaves the flusher's in-flight POST alone and just
                # reports whether it finished in time.
                done, _ = await asyncio.wait([task], timeout=self._shutdown_drain_seconds)
                if not done:
                    raise TimeoutError
            # Anything a producer queued while that ran. Emptied rather than
            # stopped at the first partial batch: there is no next interval to
            # leave a remainder for. Bounded by what is left of the budget, so
            # the two steps together cannot outlast it.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                await self._drain(until_empty=True)
        except TimeoutError:
            await self._abandon(task)
        finally:
            await self._close_client()

    async def _abandon(self, task: asyncio.Task[None] | None) -> None:
        """Give up on the drain: stop the flusher and account for what is lost."""
        if task is not None and not task.done():
            task.cancel()
            # Bounded, like the refresher shutdown in ``gateway.main``: an
            # httpx timeout is an anyio cancel scope, and one that absorbs this
            # cancellation would leave an unbounded ``await task`` hanging the
            # lifespan on an observability stream.
            await asyncio.wait([task], timeout=_TASK_CANCEL_TIMEOUT_SECONDS)
        abandoned = self._queue.qsize()
        if abandoned:
            self._take(abandoned)
            OBSERVATION_RECORDS.labels(result="dropped_shutdown").inc(abandoned)
        logger.warning(
            "Observation shipper gave up draining at shutdown after %.1fs; %d record(s) dropped",
            self._shutdown_drain_seconds,
            abandoned + self._dropped_in_flight,
        )

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                # Woken early by stop(), so a shutdown does not wait out the
                # interval before the tail is shipped.
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stopping.wait(), self._flush_interval_seconds)
                await self._drain()
            except asyncio.CancelledError:
                break
            except Exception as e:  # pragma: no cover - defensive: the loop outlives any flush
                logger.error("Observation shipper loop error: %s", e)

    async def _drain(self, *, until_empty: bool = False) -> None:
        """Ship what is queued, in batches of at most ``max_batch``.

        Keeps going while batches come back full, so a backlog clears at the
        platform's pace instead of at one batch per interval, and stops at the
        first partial batch. Draining until the queue is empty instead would never
        finish under a steady trickle: a producer only has to queue one record per
        round trip to keep the next batch non-empty, and the flusher would then
        POST tiny batches back to back and never sleep again, which is the
        opposite of batching. ``until_empty`` is for the shutdown drain, which has
        no next interval to leave a remainder for.
        """
        while batch := self._take(self._max_batch):
            await self._flush(batch)
            if len(batch) < self._max_batch and not until_empty:
                break

    def _take(self, limit: int) -> list[ObservationRecord]:
        batch: list[ObservationRecord] = []
        while len(batch) < limit:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if batch:
            OBSERVATION_QUEUE_DEPTH.set(self._queue.qsize())
        return batch

    async def _flush(self, batch: list[ObservationRecord]) -> None:
        """POST one batch. Every outcome except success is a counted loss."""
        started = time.monotonic()
        OBSERVATION_BATCH_SIZE.observe(len(batch))
        try:
            response = await self._post(batch)
        except asyncio.CancelledError:
            # The shutdown budget ran out mid-POST. These records are already off
            # the queue, so they are counted here or they vanish from the ledger,
            # and tallied separately so the warning can report them too.
            OBSERVATION_RECORDS.labels(result="dropped_shutdown").inc(len(batch))
            self._dropped_in_flight += len(batch)
            raise
        except Exception as e:
            self._record_failed_flush(batch, started, f"{type(e).__name__}: {e}")
            return

        if response.is_success:
            OBSERVATION_RECORDS.labels(result="shipped").inc(len(batch))
            OBSERVATION_FLUSH_DURATION.labels(result="ok").observe(time.monotonic() - started)
            return
        # Includes the peer that does not implement the endpoint at all (404):
        # observations are optional, so an older platform costs records, not
        # requests. Nothing is retried, since the records are disposable and a
        # retry would compete with the very traffic being measured.
        self._record_failed_flush(batch, started, f"HTTP {response.status_code}")

    def _record_failed_flush(self, batch: list[ObservationRecord], started: float, reason: str) -> None:
        logger.warning("Observation flush failed, dropping %d record(s): %s", len(batch), reason)
        OBSERVATION_RECORDS.labels(result="dropped_flush_failed").inc(len(batch))
        OBSERVATION_FLUSH_DURATION.labels(result="error").observe(time.monotonic() - started)

    async def _post(self, batch: list[ObservationRecord]) -> httpx.Response:
        client = self._ensure_client()
        return await client.post(
            self._url,
            headers=self._headers,
            json={"records": batch},
            timeout=self._timeout_seconds,
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        """The pooled client flushes share.

        Built on first flush rather than at start, so a gateway that never emits
        an observation never holds one. A client per flush would pay a fresh TCP
        and TLS handshake every interval and pool nothing.

        The limits are sized for what this client actually does: one sequential
        POST per interval to one host. httpx's default ``keepalive_expiry`` of 5s
        equals the default flush interval, so the pooled connection would be
        evicted as idle between every flush and each POST would pay the handshake
        this client exists to avoid; the expiry has to outlast an idle interval.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                transport=self._transport,
                limits=httpx.Limits(
                    max_connections=1,
                    max_keepalive_connections=1,
                    keepalive_expiry=self._flush_interval_seconds * 2,
                ),
            )
        return self._client

    async def _close_client(self) -> None:
        client, self._client = self._client, None
        if client is not None and not client.is_closed:
            await client.aclose()


def create_observation_shipper(config: GatewayConfig) -> ObservationShipper:
    """Build the shipper for this deployment, or a null one if there is nowhere to ship."""
    if not config.is_hybrid_mode:
        return NullObservationShipper()

    base_url = config.platform.get("base_url")
    gateway_token = config.platform_token
    if not base_url or not gateway_token:
        logger.warning("Observations are not being shipped: hybrid mode without a platform base URL and token")
        return NullObservationShipper()

    # Via float, since a setting arrives as whatever YAML parsed it to and a bare
    # ``int()`` rejects the numeric strings the validator accepts. The validator
    # has already established that both are whole and positive, so nothing
    # truncates here.
    return PlatformObservationShipper(
        base_url=str(base_url),
        gateway_token=gateway_token,
        max_queue=int(float(config.platform.get("observation_max_queue", DEFAULT_MAX_QUEUE))),
        max_batch=int(float(config.platform.get("observation_max_batch", DEFAULT_MAX_BATCH))),
        flush_interval_seconds=float(config.platform.get("observation_flush_interval_ms", DEFAULT_FLUSH_INTERVAL_MS))
        / 1000,
        timeout_seconds=float(config.platform.get("observation_timeout_ms", DEFAULT_TIMEOUT_MS)) / 1000,
    )
