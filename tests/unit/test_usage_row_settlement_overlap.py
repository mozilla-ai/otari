"""A success's usage row is written beside its settlement on PostgreSQL, and before it on SQLite."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import gateway.api.routes._pipeline as pipeline


def _session(dialect: str) -> Any:
    return SimpleNamespace(get_bind=lambda: SimpleNamespace(dialect=SimpleNamespace(name=dialect)))


class _HeldWriter:
    """A log writer whose ``put`` holds until released, accepting or dropping the row."""

    def __init__(self, *, accepts: bool = True) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished: list[object] = []
        self._accepts = accepts

    async def put(self, log: object) -> bool:
        self.started.set()
        await self.release.wait()
        self.finished.append(log)
        return self._accepts


@pytest.mark.asyncio
@pytest.mark.parametrize("accepts", [True, False])
async def test_on_postgresql_the_settlement_runs_while_the_row_is_written(accepts: bool) -> None:
    writer = _HeldWriter(accepts=accepts)
    row = object()
    settled: list[bool] = []

    async def settlement() -> None:
        # Starts with the write still held, so the two are not run in turn.
        await writer.started.wait()
        assert writer.finished == []
        settled.append(True)
        writer.release.set()

    await pipeline._put_while_settling(_session("postgresql"), writer, row, settlement())  # type: ignore[arg-type]

    # A row the writer drops still settles, as it did when the two ran in turn.
    assert settled == [True]
    assert writer.finished == [row]


@pytest.mark.asyncio
async def test_on_sqlite_the_row_is_written_before_the_settlement() -> None:
    writer = _HeldWriter()
    writer.release.set()
    row = object()
    order: list[str] = []

    async def settlement() -> None:
        order.append("written" if writer.finished == [row] else "pending")

    await pipeline._put_while_settling(_session("sqlite"), writer, row, settlement())  # type: ignore[arg-type]

    assert order == ["written"]


@pytest.mark.asyncio
async def test_a_cancelled_settlement_leaves_the_write_running() -> None:
    writer = _HeldWriter()
    row = object()
    never = asyncio.Event()

    async def settlement() -> None:
        await never.wait()

    request = asyncio.create_task(
        pipeline._put_while_settling(_session("postgresql"), writer, row, settlement())  # type: ignore[arg-type]
    )
    await writer.started.wait()
    request.cancel()
    await asyncio.sleep(0)
    writer.release.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    await asyncio.gather(*pipeline._USAGE_ROW_WRITES)

    assert writer.finished == [row]
