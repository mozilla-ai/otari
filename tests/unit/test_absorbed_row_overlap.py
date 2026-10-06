"""An absorbed row is written beside the next attempt, and is there when the request answers."""

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

import gateway.api.routes._pipeline as pipeline


@pytest.mark.asyncio
async def test_with_the_workspace_known_the_row_is_written_beside_the_walk() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def write() -> None:
        started.set()
        await release.wait()

    ctx: Any = SimpleNamespace(workspace_id=uuid.uuid4())
    pending: list[asyncio.Task[None]] = []

    await pipeline._write_absorbed_row(ctx, pending, write())

    # Returned before the write finished, so the next attempt is not held up by it.
    assert len(pending) == 1
    await started.wait()
    assert not pending[0].done()
    release.set()
    await pipeline._absorbed_rows_written(pending)
    assert pending[0].done()


@pytest.mark.asyncio
async def test_without_a_workspace_the_row_is_written_inline() -> None:
    written: list[str] = []

    async def write() -> None:
        written.append("row")

    ctx: Any = SimpleNamespace(workspace_id=None)
    pending: list[asyncio.Task[None]] = []

    await pipeline._write_absorbed_row(ctx, pending, write())

    # The write would need the request's session, which only one task may use.
    assert written == ["row"]
    assert pending == []


@pytest.mark.asyncio
async def test_a_cancelled_wait_leaves_the_write_running() -> None:
    release = asyncio.Event()
    finished: list[bool] = []

    async def write() -> None:
        await release.wait()
        finished.append(True)

    ctx: Any = SimpleNamespace(workspace_id=uuid.uuid4())
    pending: list[asyncio.Task[None]] = []
    await pipeline._write_absorbed_row(ctx, pending, write())

    waiter = asyncio.create_task(pipeline._absorbed_rows_written(pending))
    await asyncio.sleep(0)
    waiter.cancel()
    release.set()
    await asyncio.gather(*pending)

    assert finished == [True]
