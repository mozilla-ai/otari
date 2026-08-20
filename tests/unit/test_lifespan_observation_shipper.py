"""The lifespan owns the observation shipper's background flush task.

Producers reach the shipper through ``app.state``, so nothing observes anything
unless the lifespan builds one, starts its flusher, and publishes it. Shutdown
matters as much: the shipper is the only thing holding queued records, and its
final flush is the difference between a clean stop and a lost tail.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI

from gateway.core.config import GatewayConfig
from gateway.main import _create_lifespan
from gateway.services.observation_shipper import (
    NullObservationShipper,
    ObservationRecord,
    PlatformObservationShipper,
)


class _RecordingShipper:
    """An ObservationShipper that logs its own lifecycle."""

    def __init__(self, *, fail_to_start: bool = False) -> None:
        self.fail_to_start = fail_to_start
        self.starts = 0
        self.stops = 0

    def emit(self, record: ObservationRecord) -> None:
        return None

    async def start(self) -> None:
        self.starts += 1
        if self.fail_to_start:
            raise RuntimeError("flusher would not start")

    async def stop(self) -> None:
        self.stops += 1


def _standalone_config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'lifespan.db'}",
        master_key="sk-test-master",
    )


@pytest.mark.asyncio
async def test_lifespan_starts_publishes_and_stops_the_shipper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shipper = _RecordingShipper()
    monkeypatch.setattr("gateway.main.create_observation_shipper", lambda _config: shipper)
    app = FastAPI()

    async with _create_lifespan(_standalone_config(tmp_path))(app):
        assert shipper.starts == 1
        assert shipper.stops == 0
        assert app.state.observation_shipper is shipper

    assert shipper.stops == 1


@pytest.mark.asyncio
async def test_a_shipper_that_fails_to_start_is_not_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors the log writer's guard: there is nothing to stop, and the rest of
    the shutdown still has to run."""
    shipper = _RecordingShipper(fail_to_start=True)
    monkeypatch.setattr("gateway.main.create_observation_shipper", lambda _config: shipper)

    with pytest.raises(RuntimeError, match="flusher would not start"):
        async with _create_lifespan(_standalone_config(tmp_path))(FastAPI()):
            pytest.fail("the lifespan should not have yielded")  # pragma: no cover

    assert shipper.starts == 1
    assert shipper.stops == 0


@pytest.mark.asyncio
async def test_standalone_mode_gets_the_null_shipper(tmp_path: Path) -> None:
    """No platform to ship to, but producers still emit unconditionally."""
    app = FastAPI()

    async with _create_lifespan(_standalone_config(tmp_path))(app):
        assert isinstance(app.state.observation_shipper, NullObservationShipper)


@pytest.mark.asyncio
async def test_hybrid_mode_gets_the_platform_shipper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    config = GatewayConfig(mode="hybrid", platform={"base_url": "http://platform.test/api/v1"})
    app = FastAPI()

    # Nothing is emitted, so the flusher's first tick finds an empty queue and
    # the shutdown drain posts nothing: no platform is contacted here.
    async with _create_lifespan(config)(app):
        shipper = app.state.observation_shipper
        assert isinstance(shipper, PlatformObservationShipper)
        assert shipper.queue_depth == 0
