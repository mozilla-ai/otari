"""The batch writer keeps rows it could not write the first time, and drains on stop."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from gateway.metrics import REGISTRY
from gateway.models.usage import UsageLog
from gateway.services import log_writer as log_writer_module
from gateway.services.log_writer import BatchLogWriter


class _FlakySession:
    """Fails the first ``failures`` commits, and any commit holding a row in ``poison``."""

    def __init__(self, store: "_Store") -> None:
        self._store = store
        self._pending: list[Any] = []

    def add_all(self, rows: Any) -> None:
        self._pending.extend(rows)

    async def commit(self) -> None:
        self._store.commits += 1
        if self._store.failures > 0:
            self._store.failures -= 1
            raise OperationalError("INSERT", None, Exception("connection reset"))
        if any(row in self._store.poison for row in self._pending):
            raise IntegrityError("INSERT", None, Exception("violates a constraint"))
        self._store.written.extend(self._pending)


class _Store:
    def __init__(self, failures: int = 0, poison: tuple[Any, ...] = ()) -> None:
        self.failures = failures
        self.poison = poison
        self.commits = 0
        self.written: list[Any] = []


def _install(monkeypatch: pytest.MonkeyPatch, store: _Store) -> None:
    @asynccontextmanager
    async def _session() -> AsyncIterator[_FlakySession]:
        yield _FlakySession(store)

    monkeypatch.setattr(log_writer_module, "create_log_session", _session)


def _rows(n: int) -> list[UsageLog]:
    return [UsageLog(endpoint="/v1/chat/completions", model="m", provider="p") for _ in range(n)]


@pytest.mark.asyncio
async def test_a_failed_flush_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _Store(failures=2)
    _install(monkeypatch, store)
    rows = _rows(3)

    await BatchLogWriter(retries=3, retry_backoff=0)._flush(rows)

    assert store.written == rows


@pytest.mark.asyncio
async def test_a_row_the_database_refuses_does_not_take_its_batch_with_it(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows(3)
    store = _Store(poison=(rows[1],))
    _install(monkeypatch, store)

    await BatchLogWriter(retries=1, retry_backoff=0)._flush(rows)

    assert store.written == [rows[0], rows[2]]


@pytest.mark.asyncio
async def test_stop_writes_everything_queued(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _Store()
    _install(monkeypatch, store)
    writer = BatchLogWriter(max_batch=2, flush_interval=60)
    rows = _rows(5)

    await writer.start()
    for row in rows:
        await writer.put(row)
    await writer.stop()

    assert sorted(map(id, store.written)) == sorted(map(id, rows))


@pytest.mark.asyncio
async def test_an_unreachable_database_drops_the_batch_after_its_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """No row-by-row attempts against a database that is down: that would only back the queue up."""
    store = _Store(failures=100)
    _install(monkeypatch, store)

    await BatchLogWriter(retries=2, retry_backoff=0)._flush(_rows(10))

    assert store.commits == 3
    assert store.written == []


@pytest.mark.asyncio
async def test_a_full_queue_drops_the_row_rather_than_blocking_the_request() -> None:
    writer = BatchLogWriter(max_queue=1)
    kept, dropped = _rows(2)
    await writer.put(kept)

    await asyncio.wait_for(writer.put(dropped), timeout=1)

    assert writer._queue.qsize() == 1


async def _until(condition: Callable[[], bool]) -> None:
    while not condition():
        await asyncio.sleep(0.001)


def _dropped() -> float:
    return REGISTRY.get_sample_value("gateway_usage_log_rows_total", {"writer": "batch", "result": "dropped"}) or 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_stop_gives_up_after_its_timeout_and_counts_what_it_was_flushing(
    monkeypatch: pytest.MonkeyPatch, started: bool
) -> None:
    """Rows already off the queue when the timeout hits are dropped too, so the metric counts them."""
    store = _Store(failures=100)
    _install(monkeypatch, store)
    writer = BatchLogWriter(flush_interval=0.01, retries=50, retry_backoff=1, stop_timeout=0.2)
    await writer.put(_rows(1)[0])
    if started:
        await writer.start()
        # Until the loop has tried the row and is backing off: a fixed sleep that
        # ran short would leave it on the queue, the unstarted case over again.
        await asyncio.wait_for(_until(lambda: store.commits >= 1), timeout=5)
    before = _dropped()

    await asyncio.wait_for(writer.stop(), timeout=5)

    assert store.written == []
    assert _dropped() - before == 1


@pytest.mark.asyncio
async def test_stop_does_not_wait_out_an_idle_flush_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _Store()
    _install(monkeypatch, store)
    writer = BatchLogWriter(flush_interval=60)
    await writer.start()
    await asyncio.sleep(0)  # let the loop start waiting on the empty queue
    row = _rows(1)[0]
    await writer.put(row)
    await asyncio.sleep(0)

    await asyncio.wait_for(writer.stop(), timeout=5)

    assert store.written == [row]
