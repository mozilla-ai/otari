"""Hands recorded traces to the trace store, off the request path and within fixed limits.

Tracing is best-effort, and best-effort here is a set of limits rather than a
promise to try hard:

- ``submit`` never waits. It is called as a response finishes, so nothing a
  store does (slow, down, full) can hold a response or a connection open.
- The queue is bounded in spans. When a request's spans do not fit, they are
  dropped whole, so a stored trace is never a fragment of what happened.
- One task writes, one batch at a time, so tracing holds at most one
  connection of the request pool, and never one of the metering pool that
  keeps usage rows flowing.
- Each batch has a time limit, and shutdown has one. Past either, the spans
  are dropped rather than retried.

Every drop is counted on ``gateway_trace_spans_dropped`` with its reason, so an
operator can see what tracing lost. Nothing is logged per span.
"""

import asyncio
import contextlib
from collections import deque
from collections.abc import Awaitable, Callable
from time import monotonic

from gateway.core.database import DATA_ERRORS
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.metrics import REGISTRY, Counter, Gauge
from gateway.services.traces._service import TraceService
from gateway.types.traces import TraceWrite, WriteResult

SPANS_DROPPED = Counter(
    "gateway_trace_spans_dropped",
    "Trace spans the gateway recorded and did not store, by reason",
    ["reason"],
    registry=REGISTRY,
)
SPANS_WRITTEN = Counter(
    "gateway_trace_spans_written",
    "Trace spans handed to the trace store, by how the store accounted for them",
    ["result"],
    registry=REGISTRY,
)
QUEUED_SPANS = Gauge(
    "gateway_trace_queue_spans",
    "Trace spans waiting to be written",
    registry=REGISTRY,
)


# Stores one batch and says how its spans were accounted for.
StoreTraces = Callable[[tuple[TraceWrite, ...]], Awaitable[WriteResult]]


def stored_by(build_service: Callable[[UnitOfWork], TraceService]) -> StoreTraces:
    """Store each batch through the trace service, on a worker Unit of Work of its own."""

    async def store(batch: tuple[TraceWrite, ...]) -> WriteResult:
        async with create_unit_of_work() as uow:
            return await build_service(uow).write(batch)

    return store


class TraceWriter:
    """Queue traces and store them in batches, dropping what does not fit."""

    def __init__(
        self,
        store: StoreTraces,
        *,
        max_queued_spans: int,
        batch_spans: int,
        interval_s: float,
        write_timeout_s: float,
        shutdown_s: float,
    ) -> None:
        self._store = store
        self._max_queued = max_queued_spans
        self._batch_spans = batch_spans
        self._interval = interval_s
        self._write_timeout = write_timeout_s
        self._shutdown = shutdown_s
        self._queue: deque[TraceWrite] = deque()
        self._queued_spans = 0
        self._ready = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def discard_user(self, user_id: str) -> int:
        """Drop every queued trace a user owns, before their traces are erased; return the spans dropped.

        Erasure deletes what is stored, so what is still waiting here would otherwise
        be written after it and bring the user's traces back.
        """
        kept = deque(trace for trace in self._queue if trace.user_id != user_id)
        dropped = self._queued_spans - sum(len(trace.spans) for trace in kept)
        self._queue = kept
        self._queued_spans -= dropped
        QUEUED_SPANS.set(self._queued_spans)
        if dropped:
            SPANS_DROPPED.labels(reason="erased").inc(dropped)
        return dropped

    def submit(self, trace: TraceWrite, *, truncated: int = 0) -> None:
        """Queue one request's spans, or drop them all when they do not fit. Never waits."""
        if truncated:
            SPANS_DROPPED.labels(reason="truncated").inc(truncated)
        size = len(trace.spans)
        if not size:
            return
        if self._queued_spans + size > self._max_queued:
            SPANS_DROPPED.labels(reason="queue_full").inc(size)
            return
        self._queue.append(trace)
        self._queued_spans += size
        QUEUED_SPANS.set(self._queued_spans)
        if self._queued_spans >= self._batch_spans:
            self._ready.set()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="trace-writer")

    async def stop(self) -> None:
        """Stop writing, then flush what is queued for at most the shutdown limit and drop the rest."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        deadline = monotonic() + self._shutdown
        while self._queue:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            await self._write(self._take_batch(), timeout=min(self._write_timeout, remaining), on_timeout="shutdown")
        if self._queued_spans:
            SPANS_DROPPED.labels(reason="shutdown").inc(self._queued_spans)
            self._queue.clear()
            self._queued_spans = 0
            QUEUED_SPANS.set(0)

    async def _run(self) -> None:
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._ready.wait(), timeout=self._interval)
            self._ready.clear()
            while self._queue:
                await self._write(self._take_batch(), timeout=self._write_timeout, on_timeout="timeout")

    def _take_batch(self) -> tuple[TraceWrite, ...]:
        batch: list[TraceWrite] = []
        spans = 0
        while self._queue and (not batch or spans + len(self._queue[0].spans) <= self._batch_spans):
            trace = self._queue.popleft()
            batch.append(trace)
            spans += len(trace.spans)
        self._queued_spans -= spans
        QUEUED_SPANS.set(self._queued_spans)
        return tuple(batch)

    async def _write(self, batch: tuple[TraceWrite, ...], *, timeout: float, on_timeout: str) -> None:
        size = sum(len(trace.spans) for trace in batch)
        try:
            result = await asyncio.wait_for(self._store(batch), timeout=timeout)
        except TimeoutError:
            SPANS_DROPPED.labels(reason=on_timeout).inc(size)
            return
        # A batch is one transaction, so one trace the database refuses for what it
        # holds (its API key deleted since the request) would sink every tenant's
        # spans with it: that batch is retried a trace at a time, and only the trace
        # that still fails is dropped. Any other error (the database unreachable,
        # the pool exhausted) would fail each retry the same way, so the batch is
        # dropped at once rather than stalling the writer.
        except DATA_ERRORS as exc:
            if len(batch) > 1:
                for trace in batch:
                    await self._write((trace,), timeout=timeout, on_timeout=on_timeout)
                return
            self._drop(size, exc)
            return
        # Best-effort by contract: whatever storing raises, the writer keeps serving.
        except Exception as exc:  # noqa: BLE001
            self._drop(size, exc)
            return
        SPANS_WRITTEN.labels(result="accepted").inc(result.accepted)
        SPANS_WRITTEN.labels(result="duplicate").inc(result.duplicate)
        SPANS_WRITTEN.labels(result="rejected").inc(result.rejected)

    @staticmethod
    def _drop(size: int, exc: BaseException) -> None:
        # Logged by type only, because a database error's text carries the batch's bound values.
        logger.warning("Trace writer could not store %d spans (%s)", size, type(exc).__name__)
        SPANS_DROPPED.labels(reason="store_error").inc(size)
