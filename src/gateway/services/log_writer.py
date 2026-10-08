"""Usage log writer implementations."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Protocol

from gateway.core.database import DATA_ERRORS, DATABASE_ERRORS, create_log_session
from gateway.log_config import logger
from gateway.metrics import REGISTRY, Counter, Gauge, Histogram
from gateway.models.usage import UsageLog

QUEUE_DEPTH = Gauge(
    "gateway_usage_log_queue_depth",
    "Number of usage log entries waiting to be written",
    registry=REGISTRY,
)

BATCH_SIZE = Histogram(
    "gateway_usage_log_batch_size",
    "Number of rows per flush batch",
    ["writer"],
    registry=REGISTRY,
)

FLUSH_DURATION = Histogram(
    "gateway_usage_log_flush_duration_seconds",
    "Time spent flushing usage log batches",
    ["writer", "result"],
    registry=REGISTRY,
)

ROWS = Counter(
    "gateway_usage_log_rows",
    "Total usage log rows by outcome",
    ["writer", "result"],
    registry=REGISTRY,
)


class LogWriter(Protocol):
    async def put(self, log: UsageLog) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class SingleLogWriter:
    """Write each usage log inline, one transaction per event."""

    async def put(self, log: UsageLog) -> None:
        async with create_log_session() as db:
            try:
                # The writer only logs, because the reservation reconcile path owns spend.
                db.add(log)
                await db.commit()
                ROWS.labels(writer="single", result="written").inc()
            except DATABASE_ERRORS as e:  # pragma: no cover - defensive logging
                await db.rollback()
                logger.error("SingleLogWriter failed: %s", e)
                ROWS.labels(writer="single", result="dropped").inc()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class BatchLogWriter:
    """Queue usage logs and flush them in batches, one insert per batch.

    A failed flush is retried with backoff. When the database refused a row for
    its contents, the batch is then written row by row, so that row does not
    take the rest with it; when the database is unreachable, the batch is
    dropped, as a longer wait would only back the queue up. The queue is bounded
    and a full one drops the row rather than blocking the request writing it,
    which still has its reservation to settle.
    """

    def __init__(
        self,
        max_batch: int = 100,
        flush_interval: float = 1.0,
        max_queue: int = 10_000,
        retries: int = 3,
        retry_backoff: float = 0.5,
        stop_timeout: float = 30.0,
    ) -> None:
        self._queue: asyncio.Queue[UsageLog] = asyncio.Queue(maxsize=max_queue)
        self._max_batch = max_batch
        self._flush_interval = flush_interval
        self._retries = retries
        self._retry_backoff = retry_backoff
        self._stop_timeout = stop_timeout
        self._stopping = asyncio.Event()
        self._flushing = False
        # Rows taken off the queue and neither written nor counted as dropped yet,
        # so a stop that times out mid-flush counts them among what it drops.
        self._taken = 0
        self._task: asyncio.Task[None] | None = None

    async def put(self, log: UsageLog) -> None:
        try:
            self._queue.put_nowait(log)
        except asyncio.QueueFull:
            logger.error("BatchLogWriter queue is full; dropping a usage row")
            ROWS.labels(writer="batch", result="dropped").inc()
        QUEUE_DEPTH.set(self._queue.qsize())

    async def start(self) -> None:
        self._stopping.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Write everything queued, then stop, giving up on what is left after ``stop_timeout``."""
        self._stopping.set()
        # Counted however the drain ended: a timeout that cancels a flush in
        # progress ends the loop quietly, so no ``TimeoutError`` reaches here.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._drain(), timeout=self._stop_timeout)
        lost = self._queue.qsize() + self._taken
        if lost:
            self._taken = 0
            logger.error("BatchLogWriter did not drain within %ss; dropping %d rows", self._stop_timeout, lost)
            ROWS.labels(writer="batch", result="dropped").inc(lost)

    async def _drain(self) -> None:
        """Finish a flush in progress, then write what is still queued."""
        if self._task:
            # Idle, the loop is waiting on the queue and holds no rows, so it is
            # cancelled rather than waited out; mid-flush, it is let finish.
            if not self._flushing:
                self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self._flush_all()

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                batch = await self._collect_batch()
                if batch:
                    self._flushing = True
                    self._taken += len(batch)
                    try:
                        await self._flush(batch)
                        self._taken -= len(batch)
                    finally:
                        self._flushing = False
            except asyncio.CancelledError:  # pragma: no cover - cooperative cancel
                break
            except Exception as e:  # pragma: no cover - defensive logging
                logger.error("BatchLogWriter loop error: %s", e)

    async def _collect_batch(self) -> list[UsageLog]:
        batch: list[UsageLog] = []
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
            batch.append(item)
            self._queue.task_done()
        except TimeoutError:
            return batch

        while len(batch) < self._max_batch:
            try:
                item = self._queue.get_nowait()
                batch.append(item)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        QUEUE_DEPTH.set(self._queue.qsize())
        return batch

    async def _write(self, rows: list[UsageLog]) -> None:
        async with create_log_session() as db:
            # The writer only persists rows; spend is reconciled inline by the
            # budget reservation path (see SingleLogWriter.put).
            db.add_all(rows)
            await db.commit()

    async def _flush(self, batch: list[UsageLog]) -> None:
        start = time.monotonic()
        BATCH_SIZE.labels(writer="batch").observe(len(batch))
        for attempt in range(self._retries + 1):
            try:
                await self._write(batch)
            except DATA_ERRORS as e:
                logger.error("BatchLogWriter flush of %d rows refused; writing them one at a time: %s", len(batch), e)
                FLUSH_DURATION.labels(writer="batch", result="error").observe(time.monotonic() - start)
                await self._salvage(batch)
                return
            except DATABASE_ERRORS as e:
                if attempt < self._retries:
                    logger.warning("BatchLogWriter flush of %d rows failed, retrying: %s", len(batch), e)
                    await asyncio.sleep(self._retry_backoff * 2**attempt)
                    continue
                logger.error("BatchLogWriter flush failed, dropping %d rows: %s", len(batch), e)
                ROWS.labels(writer="batch", result="dropped").inc(len(batch))
                FLUSH_DURATION.labels(writer="batch", result="error").observe(time.monotonic() - start)
                return
            ROWS.labels(writer="batch", result="written").inc(len(batch))
            FLUSH_DURATION.labels(writer="batch", result="ok").observe(time.monotonic() - start)
            return

    async def _salvage(self, batch: list[UsageLog]) -> None:
        """Write a batch that failed as a whole one row at a time, dropping only the rows that fail."""
        for log in batch:
            try:
                await self._write([log])
                ROWS.labels(writer="batch", result="written").inc()
            except DATABASE_ERRORS as e:
                logger.error("BatchLogWriter dropped a usage row: %s", e)
                ROWS.labels(writer="batch", result="dropped").inc()

    async def _flush_all(self) -> None:
        batch: list[UsageLog] = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        self._taken += len(batch)
        for offset in range(0, len(batch), self._max_batch):
            chunk = batch[offset : offset + self._max_batch]
            await self._flush(chunk)
            self._taken -= len(chunk)


def create_log_writer(strategy: str) -> LogWriter:
    if strategy == "batch":
        return BatchLogWriter()
    return SingleLogWriter()


class NoopLogWriter:
    """LogWriter implementation that discards all writes (used when DB is unavailable)."""

    async def put(self, log: UsageLog) -> None:  # noqa: D401,B027 - trivial no-op
        return None

    async def start(self) -> None:  # noqa: D401,B027
        return None

    async def stop(self) -> None:  # noqa: D401,B027
        return None
