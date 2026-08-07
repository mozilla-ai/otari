"""Shutdown must not hang on a background refresher that will not stop.

Cancelling a task is a request, not a guarantee. The CancelledError lands at
whatever the task is awaiting, and a nested anyio cancel scope there can consume
it: ``CancelScope.__exit__`` calls ``host_task.uncancel()`` for each pending
uncancellation whenever its own scope was cancelling, then swallows the error it
sees. httpx and the provider SDKs implement their per-operation timeouts as
exactly those scopes, so a shutdown cancel that races one of their timeouts can
be absorbed. The refresher loop then resumes, falls through to its ``sleep``, and
naps out a whole interval (a day, for the models.dev catalog).

An unbounded ``await task`` after a single ``cancel()`` turns that into an
indefinite hang: the lifespan never finishes, so uvicorn's shutdown (and
``TestClient.__exit__``) blocks forever behind a background refresh.

The tests below model the absorption directly rather than trying to provoke it
through a real socket, because the race needs degraded upstream connectivity to
land and is not reproducible on demand.
"""

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI

from gateway.core.config import GatewayConfig
from gateway.main import _REFRESHER_STOP_TIMEOUT_SECONDS, _create_lifespan, _stop_refresher, _stop_refreshers


async def _absorbs_cancellation() -> None:
    """A refresher whose first cancellation is consumed, as a cancel scope would.

    ``uncancel()`` is what makes this faithful: without it the task would still be
    marked cancelling and the next await would re-raise. With it, the task is back
    to a normal state and settles in for a full interval.
    """
    absorbed = False
    while True:
        try:
            await asyncio.sleep(3600)  # stands in for the outbound fetch
        except asyncio.CancelledError:
            if absorbed:
                raise
            task = asyncio.current_task()
            assert task is not None
            task.uncancel()
            absorbed = True
        # The refresher loop's own sleep, reached the same way it is after
        # `except Exception` swallows what looked like a timeout error.
        await asyncio.sleep(86400)


@pytest.mark.asyncio
async def test_stop_refresher_returns_when_the_task_absorbs_its_cancellation() -> None:
    """The regression: shutdown gives up on a refresher instead of hanging."""
    task = asyncio.create_task(_absorbs_cancellation())
    await asyncio.sleep(0)  # let it reach its first await

    started = asyncio.get_running_loop().time()
    await asyncio.wait_for(_stop_refresher(task, "test"), timeout=_REFRESHER_STOP_TIMEOUT_SECONDS + 10)
    elapsed = asyncio.get_running_loop().time() - started

    # It waited out the grace period rather than returning instantly, and it
    # returned rather than blocking on a task that will never finish.
    assert _REFRESHER_STOP_TIMEOUT_SECONDS <= elapsed < _REFRESHER_STOP_TIMEOUT_SECONDS + 10
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_stop_refreshers_bounds_multiple_stuck_tasks_together() -> None:
    """Several cancellation-resistant refreshers share one shutdown bound."""
    tasks = [asyncio.create_task(_absorbs_cancellation()) for _ in range(2)]
    await asyncio.sleep(0)

    started = asyncio.get_running_loop().time()
    await _stop_refreshers([(task, f"test-{index}") for index, task in enumerate(tasks)])
    elapsed = asyncio.get_running_loop().time() - started

    assert _REFRESHER_STOP_TIMEOUT_SECONDS <= elapsed < _REFRESHER_STOP_TIMEOUT_SECONDS + 10
    assert all(not task.done() for task in tasks)
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)


@pytest.mark.asyncio
async def test_stop_refresher_is_prompt_for_a_well_behaved_refresher() -> None:
    """The normal path must stay instant; the bound is only a backstop."""

    async def cooperative() -> None:
        await asyncio.sleep(3600)

    task = asyncio.create_task(cooperative())
    await asyncio.sleep(0)

    started = asyncio.get_running_loop().time()
    await _stop_refresher(task, "test")

    assert asyncio.get_running_loop().time() - started < 1.0
    assert task.cancelled()


@pytest.mark.asyncio
async def test_stop_refresher_logs_an_unexpected_error_instead_of_raising() -> None:
    """A refresher that died must not abort the rest of shutdown.

    The log writer and the pooled search client are closed after the refreshers,
    so an exception escaping here would leak both.
    """

    async def explodes() -> None:
        raise RuntimeError("refresher blew up")

    task = asyncio.create_task(explodes())
    await asyncio.sleep(0)

    await _stop_refresher(task, "test")  # must not raise

    assert task.done()


@pytest.mark.asyncio
async def test_lifespan_shutdown_completes_despite_a_stuck_refresher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the app finishes shutting down even if a refresher hangs.

    This is the shape that failed on CI, where a models.dev fetch absorbed the
    shutdown cancel and `TestClient.__exit__` blocked until the suite's 120s
    timeout killed it.
    """
    monkeypatch.setattr("gateway.main.run_catalog_refresher", lambda *_a, **_k: _absorbs_cancellation())
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'lifespan.db'}",
        master_key="sk-test-master",
    )
    lifespan = _create_lifespan(config)

    # No asyncio.timeout wrapper: if shutdown regresses this hangs, and the
    # suite-wide pytest timeout reports it. A short bound here would be
    # indistinguishable from the fix under test.
    async with lifespan(FastAPI()):
        pass
