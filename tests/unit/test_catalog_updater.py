"""The scheduled catalog update loop: when it runs, and that a failure never stops it."""

import asyncio
import logging

import pytest
from sqlalchemy.exc import OperationalError

from gateway.core.config import GatewayConfig
from gateway.services.providers import run_catalog_updater


def _ticks(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """Let the loop sleep ``count`` times, then cancel it the way shutdown does."""
    remaining = [count]

    async def _sleep(_seconds: float) -> None:
        if remaining[0] == 0:
            raise asyncio.CancelledError
        remaining[0] -= 1

    monkeypatch.setattr("gateway.services.providers._hosted_catalog_update.asyncio.sleep", _sleep)


@pytest.mark.asyncio
async def test_a_manual_policy_never_runs_an_update(monkeypatch: pytest.MonkeyPatch) -> None:
    _ticks(monkeypatch, 3)
    runs: list[str] = []

    async def run(policy: str, interval: float) -> str | None:
        runs.append(policy)
        return "ran"

    with pytest.raises(asyncio.CancelledError):
        await run_catalog_updater(GatewayConfig(pricing_refresh="manual"), run)

    assert runs == []


@pytest.mark.asyncio
async def test_each_tick_runs_the_update_under_the_policy_in_force(monkeypatch: pytest.MonkeyPatch) -> None:
    _ticks(monkeypatch, 2)
    config = GatewayConfig(pricing_refresh="review")
    runs: list[tuple[str, float]] = []

    async def run(policy: str, interval: float) -> str | None:
        runs.append((policy, interval))
        config.pricing_refresh = "auto"
        return None

    with pytest.raises(asyncio.CancelledError):
        await run_catalog_updater(config, run)

    interval = float(config.pricing_refresh_interval_seconds)
    assert runs == [("review", interval), ("auto", interval)]


@pytest.mark.asyncio
async def test_a_failed_update_is_logged_without_its_database_detail_and_the_loop_goes_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _ticks(monkeypatch, 3)
    calls = 0

    async def run(policy: str, interval: float) -> str | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OperationalError("UPDATE secret", {"token": "sk-secret"}, Exception("boom"))
        if calls == 2:
            raise RuntimeError("upstream")
        return "prices applied; 0 models added; 0 providers not reached"

    with caplog.at_level(logging.INFO, logger="gateway"), pytest.raises(asyncio.CancelledError):
        await run_catalog_updater(GatewayConfig(pricing_refresh="auto"), run)

    assert calls == 3
    assert "sk-secret" not in caplog.text
    assert "Scheduled catalog update: prices applied" in caplog.text
