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
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from gateway import main as gateway_main
from gateway.container import build_container
from gateway.core.config import GatewayConfig
from gateway.main import (
    _LIFESPAN_WORKERS,
    _REFRESHER_STOP_TIMEOUT_SECONDS,
    _create_lifespan,
    _start_lifespan_workers,
    _stop_refresher,
    _stop_refreshers,
)


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
    await asyncio.wait_for(_stop_refresher(task, "test"), timeout=_REFRESHER_STOP_TIMEOUT_SECONDS + 2)
    elapsed = asyncio.get_running_loop().time() - started

    # It waited out the grace period rather than returning instantly, and it
    # returned rather than blocking on a task that will never finish.
    assert _REFRESHER_STOP_TIMEOUT_SECONDS <= elapsed < _REFRESHER_STOP_TIMEOUT_SECONDS + 2
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

    assert _REFRESHER_STOP_TIMEOUT_SECONDS <= elapsed < _REFRESHER_STOP_TIMEOUT_SECONDS + 2
    assert all(not task.done() for task in tasks)
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)


@pytest.mark.asyncio
async def test_stop_refreshers_allows_no_tasks() -> None:
    """Hybrid mode has no local refreshers to stop."""
    await _stop_refreshers([])


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
    lifespan = _create_lifespan()
    app = FastAPI()
    app.state.config = config
    app.state.enabled_features = ()
    # The first-run key is minted through the bound key format, which create_app
    # would have put here.
    app.state.container = build_container()

    # No asyncio.timeout wrapper: if shutdown regresses this hangs, and the
    # suite-wide pytest timeout reports it. A short bound here would be
    # indistinguishable from the fix under test.
    async with lifespan(app):
        pass


def _recording_refresher(name: str, started: list[str]) -> Callable[..., Coroutine[Any, Any, None]]:
    """A stand-in refresher that records when it is called, not when it is awaited.

    Recording at call time lets a caller cancel the task before the loop runs it,
    so a real refresher that slipped through fails an assertion rather than
    hanging on its own sleep.
    """

    async def _noop() -> None:
        return None

    def _start(*_args: Any, **_kwargs: Any) -> Coroutine[Any, Any, None]:
        started.append(name)
        return _noop()

    return _start


async def _started_worker_names(config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
    """Start the registry against stand-in refreshers, then stop it.

    Returns the worker names that started, and the refreshers they called.
    """
    called: list[str] = []
    for attribute in dir(gateway_main):
        if attribute.startswith("run_"):
            monkeypatch.setattr(gateway_main, attribute, _recording_refresher(attribute, called))

    workers = _start_lifespan_workers(config)
    for task, _worker in workers:
        task.cancel()
    await asyncio.gather(*(task for task, _worker in workers), return_exceptions=True)
    return [worker.name for _task, worker in workers], called


@pytest.mark.asyncio
async def test_every_worker_looks_its_refresher_up_when_it_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registry entry must resolve its refresher in ``gateway.main``, not hold it.

    The root conftest substitutes refreshers by that name to keep the unit suite
    off the network. An entry holding the function object would keep calling the
    original, and the substitution would silently do nothing.
    """
    names, called = await _started_worker_names(GatewayConfig(master_key="sk-test-master"), monkeypatch)

    assert names == [worker.name for worker in _LIFESPAN_WORKERS]
    assert len(called) == len(_LIFESPAN_WORKERS)


@pytest.mark.asyncio
async def test_the_reservation_sweeper_is_the_one_worker_a_setting_turns_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sweeping is opt-out, and opting out must not drop any other worker."""
    config = GatewayConfig(master_key="sk-test-master", budget_reservation_sweep_interval_sec=0)
    names, _called = await _started_worker_names(config, monkeypatch)

    assert "budget reservation sweep" not in names
    assert names == [worker.name for worker in _LIFESPAN_WORKERS if worker.name != "budget reservation sweep"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        GatewayConfig(master_key="sk-test-master", files_sweep_interval_sec=0),
        GatewayConfig(master_key="sk-test-master", files_enabled=False),
    ],
)
async def test_the_file_sweeper_stops_with_files_or_its_interval(
    monkeypatch: pytest.MonkeyPatch, config: GatewayConfig
) -> None:
    """Disabling files, or the sweep alone, drops that one worker and no other."""
    names, _called = await _started_worker_names(config, monkeypatch)

    assert "file retention sweep" not in names
    assert names == [worker.name for worker in _LIFESPAN_WORKERS if worker.name != "file retention sweep"]
