"""Usage log writer implementations."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError

from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.metrics import (
    log_writer_batch_size,
    log_writer_flush_duration,
    log_writer_queue_depth,
    log_writer_rows,
)
from gateway.models.entities import UsageLog, User


class LogWriter(Protocol):
    async def put(self, log: UsageLog) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class SingleLogWriter:
    """Write each usage log inline, one transaction per event."""

    async def put(self, log: UsageLog) -> None:
        async with create_session() as db:
            try:
                db.add(log)
                if log.cost and log.user_id:
                    await db.execute(
                        update(User)
                        .where(User.user_id == log.user_id, User.deleted_at.is_(None))
                        .values(spend=User.spend + log.cost)
                    )
                await db.commit()
                log_writer_rows.labels(writer="single", result="written").inc()
            except SQLAlchemyError as e:  # pragma: no cover - defensive logging
                await db.rollback()
                logger.error("SingleLogWriter failed: %s", e)
                log_writer_rows.labels(writer="single", result="dropped").inc()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class BatchLogWriter:
    """Queue usage logs and flush in batches."""

    _STOP_TIMEOUT = 10.0

    def __init__(self, max_batch: int = 100, flush_interval: float = 1.0) -> None:
        self._queue: asyncio.Queue[UsageLog] = asyncio.Queue()
        self._max_batch = max_batch
        self._flush_interval = flush_interval
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def put(self, log: UsageLog) -> None:
        await self._queue.put(log)
        log_writer_queue_depth.set(self._queue.qsize())

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        # Graceful shutdown: signal the loop, let it finish the in-flight flush
        # and drain the queue, then exit cleanly. Cancelling mid-flush would
        # lose the batch (items are task_done()'d before commit).
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, self._STOP_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("BatchLogWriter stop timed out after %.1fs; cancelling", self._STOP_TIMEOUT)
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    batch = await self._collect_batch()
                    if batch:
                        await self._flush(batch)
                except Exception as e:  # pragma: no cover - defensive logging
                    logger.error("BatchLogWriter loop error: %s", e)
        finally:
            try:
                await self._flush_all()
            except Exception as e:  # pragma: no cover - defensive logging
                logger.error("BatchLogWriter final drain failed: %s", e)

    async def _collect_batch(self) -> list[UsageLog]:
        # Wait for first item, stop signal, or flush interval - whichever first.
        get_task = asyncio.ensure_future(self._queue.get())
        stop_task = asyncio.ensure_future(self._stop_event.wait())
        try:
            done, _ = await asyncio.wait(
                {get_task, stop_task},
                timeout=self._flush_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not stop_task.done():
                stop_task.cancel()
            if not get_task.done():
                get_task.cancel()

        batch: list[UsageLog] = []
        if get_task in done:
            batch.append(get_task.result())
            self._queue.task_done()

        while len(batch) < self._max_batch:
            try:
                batch.append(self._queue.get_nowait())
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        return batch

    async def _flush(self, batch: list[UsageLog]) -> None:
        start = time.monotonic()
        log_writer_batch_size.labels(writer="batch").observe(len(batch))
        try:
            async with create_session() as db:
                for log in batch:
                    db.add(log)
                    if log.cost and log.user_id:
                        await db.execute(
                            update(User)
                            .where(User.user_id == log.user_id, User.deleted_at.is_(None))
                            .values(spend=User.spend + log.cost)
                        )
                await db.commit()
                log_writer_rows.labels(writer="batch", result="written").inc(len(batch))
            log_writer_flush_duration.labels(writer="batch", result="ok").observe(time.monotonic() - start)
        except SQLAlchemyError as e:  # pragma: no cover - defensive logging
            logger.error("BatchLogWriter flush failed, dropping %d rows: %s", len(batch), e)
            log_writer_rows.labels(writer="batch", result="dropped").inc(len(batch))
            log_writer_flush_duration.labels(writer="batch", result="error").observe(time.monotonic() - start)

    async def _flush_all(self) -> None:
        batch: list[UsageLog] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
        if batch:
            await self._flush(batch)


def create_log_writer(strategy: str) -> LogWriter:
    if strategy == "batch":
        return BatchLogWriter()
    return SingleLogWriter()  # type: ignore[return-value]


class NoopLogWriter:
    """LogWriter implementation that discards all writes (used when DB is unavailable)."""

    async def put(self, log: UsageLog) -> None:  # noqa: D401,B027 - trivial no-op
        return None

    async def start(self) -> None:  # noqa: D401,B027
        return None

    async def stop(self) -> None:  # noqa: D401,B027
        return None
