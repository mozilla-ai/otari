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
import logging
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from gateway import main as gateway_main
from gateway.container import build_container
from gateway.core.config import GatewayConfig
from gateway.core.database import dispose_db
from gateway.log_config import logger as gateway_logger
from gateway.main import (
    _LIFESPAN_WORKERS,
    _REFRESHER_STOP_TIMEOUT_SECONDS,
    _create_lifespan,
    _start_lifespan_workers,
    _stop_refreshers,
)
from gateway.ports.file_storage_port import FileStoragePort
from gateway.ports.provider_file_port import ProviderFilePort
from gateway.services.organizations import ExpiredInvitationCleanup, InvitationDelivery
from gateway.services.tools import AttachedMcpServerIndex


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
    await asyncio.wait_for(_stop_refreshers([(task, "test")]), timeout=_REFRESHER_STOP_TIMEOUT_SECONDS + 2)
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
    """A gateway with no refreshers running has nothing to stop."""
    await _stop_refreshers([])


@pytest.mark.asyncio
async def test_stop_refresher_is_prompt_for_a_well_behaved_refresher() -> None:
    """The normal path must stay instant; the bound is only a backstop."""

    async def cooperative() -> None:
        await asyncio.sleep(3600)

    task = asyncio.create_task(cooperative())
    await asyncio.sleep(0)

    started = asyncio.get_running_loop().time()
    await _stop_refreshers([(task, "test")])

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

    await _stop_refreshers([(task, "test")])  # must not raise

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
    # create_app would have put the container here. It is built with the config
    # because the file store binding reads it, and the first-run key is minted
    # through its bound key format.
    app.state.container = build_container(config=config)

    # No asyncio.timeout wrapper: if shutdown regresses this hangs, and the
    # suite-wide pytest timeout reports it. A short bound here would be
    # indistinguishable from the fix under test.
    async with lifespan(app):
        pass


@pytest.mark.asyncio
async def test_a_cache_reset_that_raises_does_not_stop_the_rest_of_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The next reset still runs, the engine is still disposed, and the failure is logged."""
    reset: list[str] = []
    disposed: list[bool] = []

    def explodes() -> None:
        raise RuntimeError("reset blew up")

    first, second, *rest = _LIFESPAN_WORKERS
    workers = (
        replace(first, reset=explodes),
        replace(second, reset=lambda: reset.append(second.name)),
        *rest,
    )
    monkeypatch.setattr(gateway_main, "_LIFESPAN_WORKERS", workers)

    async def recording_dispose_db() -> None:
        disposed.append(True)
        await dispose_db()

    monkeypatch.setattr(gateway_main, "dispose_db", recording_dispose_db)
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'lifespan.db'}",
        master_key="sk-test-master",
    )
    app = FastAPI()
    app.state.config = config
    app.state.enabled_features = ()
    app.state.container = build_container(config=config)

    # The gateway logger does not propagate, so caplog sees it only with its handler attached.
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        async with _create_lifespan()(app):
            pass
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert reset == [second.name]
    assert disposed == [True]
    failures = [r for r in caplog.records if r.getMessage() == f"{first.name} cache reset failed at shutdown"]
    assert len(failures) == 1
    assert failures[0].exc_info is not None


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
    monkeypatch.setattr(InvitationDelivery, "poll", _recording_refresher("InvitationDelivery.poll", called))
    monkeypatch.setattr(ExpiredInvitationCleanup, "poll", _recording_refresher("ExpiredInvitationCleanup.poll", called))
    monkeypatch.setattr(
        AttachedMcpServerIndex, "run_refresher", _recording_refresher("AttachedMcpServerIndex.run_refresher", called)
    )

    workers = _start_lifespan_workers(config, build_container(config=config))
    for task, _worker in workers:
        task.cancel()
    await asyncio.gather(*(task for task, _worker in workers), return_exceptions=True)
    return [worker.name for _task, worker in workers], called


def _full_config(**overrides: Any) -> GatewayConfig:
    """A config every conditional worker starts under, so one that does not is the finding.

    Each ``and no other`` assertion below compares against the whole registry,
    which only means anything from a baseline that turns nothing off: a worker
    gated on a setting this config leaves unset would look like a worker the
    change under test dropped.
    """
    return GatewayConfig(master_key="sk-test-master", sandbox_url="http://sandbox:8080", **overrides)


@pytest.mark.asyncio
async def test_every_worker_looks_its_refresher_up_when_it_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registry entry must resolve its refresher in ``gateway.main``, not hold it.

    The root conftest substitutes refreshers by that name to keep the unit suite
    off the network. An entry holding the function object would keep calling the
    original, and the substitution would silently do nothing.
    """
    names, called = await _started_worker_names(_full_config(), monkeypatch)

    assert names == [worker.name for worker in _LIFESPAN_WORKERS]
    assert len(called) == len(_LIFESPAN_WORKERS)


@pytest.mark.asyncio
async def test_the_reservation_sweeper_is_the_one_worker_a_setting_turns_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sweeping is opt-out, and opting out must not drop any other worker."""
    config = _full_config(budget_reservation_sweep_interval_sec=0)
    names, _called = await _started_worker_names(config, monkeypatch)

    assert "budget reservation sweep" not in names
    assert names == [worker.name for worker in _LIFESPAN_WORKERS if worker.name != "budget reservation sweep"]


@pytest.mark.asyncio
async def test_the_file_sweeper_stops_with_its_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interval of 0 drops that one worker and no other."""
    names, _called = await _started_worker_names(_full_config(files_sweep_interval_sec=0), monkeypatch)

    assert "file retention sweep" not in names
    assert names == [worker.name for worker in _LIFESPAN_WORKERS if worker.name != "file retention sweep"]


@pytest.mark.asyncio
async def test_the_file_sweeper_keeps_running_with_the_files_api_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turning the API off leaves the sweep, since every stored file still owes the deletion its expiry promised."""
    names, _called = await _started_worker_names(_full_config(files_api_enabled=False), monkeypatch)

    assert names == [worker.name for worker in _LIFESPAN_WORKERS]


@pytest.mark.asyncio
async def test_the_file_sweeper_skips_a_store_it_cannot_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the API off a broken backend does not stop the boot, so the sweep that needs it stays off."""
    config = _full_config(files_api_enabled=False, files_backend="s3", files_s3_bucket=None)

    names, _called = await _started_worker_names(config, monkeypatch)

    assert names == [worker.name for worker in _LIFESPAN_WORKERS if worker.name != "file retention sweep"]


class _WaitingFiles:
    """Stands in for the file service the boot check asks."""

    def __init__(self, *, waiting: bool, fails: Exception | None = None) -> None:
        self.waiting = waiting
        self.fails = fails
        self.asked = False

    async def has_reclaimable(self) -> bool:
        self.asked = True
        if self.fails is not None:
            raise self.fails
        return self.waiting


async def _unreclaimed_file_warnings(
    config: GatewayConfig,
    files: _WaitingFiles,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    *,
    storage: FileStoragePort | None,
) -> list[str]:
    """Run the boot check against ``files`` and return the warnings it logged about reclaiming."""

    @asynccontextmanager
    async def _unit_of_work() -> AsyncIterator[object]:
        yield object()

    monkeypatch.setattr(gateway_main, "create_unit_of_work", _unit_of_work)
    monkeypatch.setattr(gateway_main, "build_file_service", lambda _uow, _backends, _config: files)
    gateway_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=gateway_logger.name):
            await gateway_main._warn_about_unreclaimed_files(config, storage, cast(ProviderFilePort, object()))
    finally:
        gateway_logger.removeHandler(caplog.handler)
    return [record.getMessage() for record in caplog.records if "reclaim" in record.getMessage()]


@pytest.mark.asyncio
async def test_a_stopped_sweep_with_files_waiting_says_so_at_boot(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    files = _WaitingFiles(waiting=True)

    warnings = await _unreclaimed_file_warnings(
        _full_config(files_sweep_interval_sec=0),
        files,
        monkeypatch,
        caplog,
        storage=cast(FileStoragePort, object()),
    )

    assert len(warnings) == 1
    assert "files_sweep_interval_sec" in warnings[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interval", "waiting", "asked"),
    [
        (0, False, True),
        (3600, True, False),
    ],
)
async def test_the_boot_check_stays_quiet_when_the_sweep_runs_or_nothing_waits(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, interval: int, waiting: bool, asked: bool
) -> None:
    files = _WaitingFiles(waiting=waiting)

    warnings = await _unreclaimed_file_warnings(
        _full_config(files_sweep_interval_sec=interval),
        files,
        monkeypatch,
        caplog,
        storage=cast(FileStoragePort, object()),
    )

    assert warnings == []
    assert files.asked is asked


@pytest.mark.asyncio
async def test_the_boot_check_leaves_a_missing_store_to_its_own_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Without a store the sweep cannot run at all, which the store's own warning already says."""
    files = _WaitingFiles(waiting=True)

    warnings = await _unreclaimed_file_warnings(
        _full_config(files_sweep_interval_sec=0), files, monkeypatch, caplog, storage=None
    )

    assert warnings == []
    assert files.asked is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        GatewayConfig(master_key="sk-test-master"),
        _full_config(sandbox_container_idle_ttl_sec=0),
    ],
)
async def test_the_container_sweeper_stops_without_a_sandbox_or_without_reuse(
    monkeypatch: pytest.MonkeyPatch, config: GatewayConfig
) -> None:
    """Nothing holds a sandbox past its request, so there are no rows to sweep."""
    names, _called = await _started_worker_names(config, monkeypatch)

    assert "sandbox container sweep" not in names
    assert names == [worker.name for worker in _LIFESPAN_WORKERS if worker.name != "sandbox container sweep"]


@pytest.mark.asyncio
async def test_the_idempotency_sweep_runs_while_the_header_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Records stored before the header was turned off still hold generated content, so they must still expire."""
    names, _called = await _started_worker_names(_full_config(idempotency_retention_sec=0), monkeypatch)

    assert "idempotency sweep" in names


def test_the_idempotency_sweep_cannot_be_turned_off() -> None:
    with pytest.raises(ValidationError):
        GatewayConfig(idempotency_sweep_interval_sec=0)


@pytest.mark.asyncio
async def test_the_boot_check_lets_the_boot_go_on_when_the_database_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    files = _WaitingFiles(waiting=True, fails=SQLAlchemyError("connection refused"))

    warnings = await _unreclaimed_file_warnings(
        _full_config(files_sweep_interval_sec=0),
        files,
        monkeypatch,
        caplog,
        storage=cast(FileStoragePort, object()),
    )

    assert len(warnings) == 1
    assert warnings[0].startswith("Could not check for files waiting to be reclaimed")


@pytest.mark.asyncio
async def test_the_boot_check_does_not_hide_an_error_that_is_not_the_database(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    files = _WaitingFiles(waiting=True, fails=RuntimeError("a bug in the check"))

    with pytest.raises(RuntimeError, match="a bug in the check"):
        await _unreclaimed_file_warnings(
            _full_config(files_sweep_interval_sec=0),
            files,
            monkeypatch,
            caplog,
            storage=cast(FileStoragePort, object()),
        )
