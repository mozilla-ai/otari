import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from gateway.models.entities import UsageLog
from gateway.services.log_writer import BatchLogWriter, SingleLogWriter


@pytest.mark.asyncio
async def test_single_log_writer_rolls_back_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    session = AsyncMock()
    session.add = MagicMock()
    session.commit.side_effect = SQLAlchemyError("boom")

    @asynccontextmanager
    async def _session_cm():  # type: ignore[return-annnotation]
        yield session

    monkeypatch.setattr("gateway.services.log_writer.create_session", lambda: _session_cm())
    writer = SingleLogWriter()

    log = UsageLog(id="log", model="test-model", endpoint="/v1/test", status="success")
    await writer.put(log)

    session.rollback.assert_awaited()


def _make_log(i: int) -> UsageLog:
    return UsageLog(id=f"log-{i}", model="m", endpoint="/v1/test", status="success")


class _RecordingFlushBatchWriter(BatchLogWriter):
    """BatchLogWriter that records every flushed id instead of touching the DB.

    Optionally blocks the first flush on an event so we can race it against stop().
    """

    def __init__(self, first_flush_gate: asyncio.Event | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.flushed: list[str] = []
        self._first_flush_gate = first_flush_gate
        self._first_flush_started = asyncio.Event()
        self._first_flush_done = False

    async def _flush(self, batch: list[UsageLog]) -> None:
        if not self._first_flush_done and self._first_flush_gate is not None:
            self._first_flush_started.set()
            await self._first_flush_gate.wait()
            self._first_flush_done = True
        self.flushed.extend(log.id for log in batch)


@pytest.mark.asyncio
async def test_batch_writer_flushes_queued_items_on_stop() -> None:
    writer = _RecordingFlushBatchWriter(max_batch=10, flush_interval=60.0)
    await writer.start()
    for i in range(5):
        await writer.put(_make_log(i))

    await writer.stop()

    assert sorted(writer.flushed, key=lambda s: int(s.split("-")[1])) == [f"log-{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_batch_writer_does_not_drop_in_flight_batch_on_stop() -> None:
    """Regression: stop() used to cancel the task mid-_flush, losing the whole batch.

    The graceful shutdown variant must let the in-flight flush complete.
    """
    gate = asyncio.Event()
    writer = _RecordingFlushBatchWriter(first_flush_gate=gate, max_batch=10, flush_interval=0.01)
    await writer.start()

    for i in range(7):
        await writer.put(_make_log(i))

    await writer._first_flush_started.wait()

    for i in range(7, 12):
        await writer.put(_make_log(i))

    stop_task = asyncio.create_task(writer.stop())
    await asyncio.sleep(0.05)
    assert not stop_task.done()
    gate.set()
    await stop_task

    assert sorted(writer.flushed, key=lambda s: int(s.split("-")[1])) == [f"log-{i}" for i in range(12)]


@pytest.mark.asyncio
async def test_batch_writer_stop_times_out_and_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a flush wedges, stop() should bail out after the timeout rather than hang forever."""
    gate = asyncio.Event()
    writer = _RecordingFlushBatchWriter(first_flush_gate=gate, max_batch=10, flush_interval=0.01)
    monkeypatch.setattr(writer, "_STOP_TIMEOUT", 0.05)
    await writer.start()

    await writer.put(_make_log(0))
    await writer._first_flush_started.wait()

    await writer.stop()

    assert writer._task is not None and writer._task.done()
    gate.set()


@pytest.mark.asyncio
async def test_batch_writer_stop_is_idempotent_when_not_started() -> None:
    writer = BatchLogWriter()
    await writer.stop()
