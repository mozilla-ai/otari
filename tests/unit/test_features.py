"""The core feature registry and the three wiring points that share one reading of it.

A listed feature mounts its routers as core routes, hosts its surface, and runs
its worker under the lifespan's supervisor, each only when its own ``enabled``
says so. The registry is empty today, so a probe feature stands in for one.
"""

import ast
import asyncio
import logging
import threading
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from gateway import features
from gateway.api.deps import reset_config
from gateway.api.routes.bootstrap import HOSTED_SURFACES, STANDALONE_SURFACES, published_surfaces
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.core.feature import CoreFeature, Worker
from gateway.core.surface import Surface
from gateway.main import _create_lifespan, create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "src" / "gateway" / "features.py"
PLATFORM_TOKEN = "gw_test_token"


@pytest.fixture(autouse=True)
def _reset_process_state() -> Generator[None, None, None]:
    """Put back the process-wide config and engine ``create_app`` and the lifespan install."""
    yield
    reset_config()
    reset_db()


def _standalone(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(database_url=f"sqlite:///{tmp_path / 'features.db'}", master_key="sk-test-master")


def _hosted(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        mode="hosted", database_url=f"sqlite:///{tmp_path / 'features.db'}", master_key="sk-test-master"
    )


def _hybrid() -> GatewayConfig:
    return GatewayConfig(mode="hybrid", platform={"base_url": "http://localhost:8100/api/v1"})


def _probe(*, enabled: bool, surface: Surface | None = Surface("probe"), worker: Worker | None = None) -> CoreFeature:
    router = APIRouter(prefix="/probe")

    @router.get("")
    async def read_probe() -> dict[str, str]:
        return {"ok": "yes"}

    return CoreFeature(
        name="probe",
        surface=surface,
        enabled=lambda _config: enabled,
        routers=lambda _config: (router,),
        worker=worker,
    )


def test_the_registry_is_a_literal_tuple() -> None:
    """Nothing scans and nothing computes: the line in the repo is the whole answer."""
    tree = ast.parse(REGISTRY_PATH.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "CORE_FEATURES"
    ]
    assert len(assignments) == 1, "CORE_FEATURES is assigned once, with its annotation"
    value = assignments[0].value
    assert isinstance(value, ast.Tuple), "CORE_FEATURES is a tuple literal, not a computed value"
    assert all(isinstance(element, ast.Attribute | ast.Name) for element in value.elts), (
        "each entry names the CoreFeature its own module declares; none is constructed in the registry"
    )


def test_registry_features_have_distinct_names_and_surfaces() -> None:
    """A collision is a mistake to catch here, not one for the published set to merge away."""
    names = [feature.name for feature in features.CORE_FEATURES]
    surfaces = [feature.surface.name for feature in features.CORE_FEATURES if feature.surface is not None]
    assert len(set(names)) == len(names), "two features share a name"
    assert len(set(surfaces)) == len(surfaces), "two features share a surface"
    assert not set(surfaces) & {*STANDALONE_SURFACES, *HOSTED_SURFACES}, "a feature surface repeats an edition's own"


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_a_feature_mounts_its_routers_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=enabled),))
    app = create_app(_standalone(tmp_path))
    mounted = {route.path for route in app.routes if isinstance(route, APIRoute)}
    assert (f"{API_ROOT}/probe" in mounted) is enabled


def test_a_feature_route_is_a_core_route_with_no_capability_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listed feature is part of the build, so its route answers as any core route does.

    A contributed router would sit behind ``require_capability``; a registry
    router does not, which is the difference between the two seams.
    """
    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=True),))
    app = create_app(_standalone(tmp_path))

    with TestClient(app) as client:
        response = client.get(f"{API_ROOT}/probe")

    assert response.status_code == 200
    assert response.json() == {"ok": "yes"}


def test_a_hybrid_gateway_mounts_no_feature_routers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Management plane only: hybrid returns before the management routers, and a feature sits with them."""
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=True),))
    app = create_app(_hybrid())
    mounted = {route.path for route in app.routes if isinstance(route, APIRoute)}
    assert f"{API_ROOT}/probe" not in mounted


def test_an_enabled_feature_hosts_its_surface_beside_the_fixed_set(tmp_path: Path) -> None:
    enabled = (_probe(enabled=True),)
    assert published_surfaces(_standalone(tmp_path), enabled) == sorted((*STANDALONE_SURFACES, "probe"))
    assert published_surfaces(_hosted(tmp_path), enabled) == sorted((*HOSTED_SURFACES, "probe"))


def test_a_feature_surface_is_published_only_by_the_deployments_it_names(tmp_path: Path) -> None:
    enabled = (_probe(enabled=True, surface=Surface("probe", standalone=False)),)
    assert "probe" not in published_surfaces(_standalone(tmp_path), enabled)
    assert "probe" in published_surfaces(_hosted(tmp_path), enabled)


def test_a_surface_the_edition_already_hosts_is_published_once(tmp_path: Path) -> None:
    """The published surfaces are a set, whatever the registry repeats."""
    enabled = (_probe(enabled=True, surface=Surface(STANDALONE_SURFACES[0])),)
    assert published_surfaces(_standalone(tmp_path), enabled) == sorted(STANDALONE_SURFACES)


def test_a_disabled_feature_hosts_no_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=False),))
    app = create_app(_standalone(tmp_path))

    with TestClient(app) as client:
        surfaces = client.get(f"{API_ROOT}/bootstrap").json()["surfaces"]

    assert surfaces == sorted(STANDALONE_SURFACES)


def test_a_feature_with_no_page_hosts_no_surface(tmp_path: Path) -> None:
    enabled = (_probe(enabled=True, surface=None),)
    assert published_surfaces(_standalone(tmp_path), enabled) == sorted(STANDALONE_SURFACES)


def test_a_hybrid_gateway_publishes_no_surface_whatever_the_registry_says(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=True),))
    app = create_app(_hybrid())

    with TestClient(app) as client:
        surfaces = client.get(f"{API_ROOT}/bootstrap").json()["surfaces"]

    assert surfaces == []


def test_the_bootstrap_publishes_an_enabled_feature_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The end the helper exists for: the dashboard learns about the page from here."""
    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=True),))
    app = create_app(_standalone(tmp_path))

    with TestClient(app) as client:
        surfaces = client.get(f"{API_ROOT}/bootstrap").json()["surfaces"]

    assert "probe" in surfaces
    assert surfaces == sorted(surfaces)


class _Probe:
    """A worker that records the task it ran on and what it was handed."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.config: GatewayConfig | None = None

    async def run(self, config: GatewayConfig) -> None:
        self.task = asyncio.current_task()
        self.config = config
        self.started.set()
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_the_lifespan_starts_an_enabled_worker_and_stops_it_at_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _Probe()
    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=True, worker=probe.run),))
    app = create_app(_standalone(tmp_path))

    async with _create_lifespan()(app):
        await asyncio.wait_for(probe.started.wait(), timeout=5)
        assert probe.config is app.state.config

    assert probe.task is not None
    assert probe.task.cancelled()


@pytest.mark.asyncio
async def test_the_lifespan_leaves_a_disabled_worker_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _Probe()
    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=False, worker=probe.run),))
    app = create_app(_standalone(tmp_path))

    async with _create_lifespan()(app):
        await asyncio.sleep(0)

    assert not probe.started.is_set()


@pytest.mark.asyncio
async def test_a_worker_that_dies_is_reported_when_it_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Not only at shutdown, which is the one place the supervisor otherwise looks."""
    died = asyncio.Event()

    async def worker(_config: GatewayConfig) -> None:
        died.set()
        raise RuntimeError("probe worker blew up")

    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=True, worker=worker),))
    app = create_app(_standalone(tmp_path))
    gateway_logger = logging.getLogger("gateway")
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.ERROR, logger="gateway")
    try:
        async with _create_lifespan()(app):
            await asyncio.wait_for(died.wait(), timeout=5)
            await asyncio.sleep(0)
            assert "probe worker stopped with an unexpected error" in caplog.text
    finally:
        gateway_logger.removeHandler(caplog.handler)


@pytest.mark.asyncio
async def test_a_worker_that_will_not_stop_is_abandoned_and_named_as_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Shutdown still finishes under the shared bound, and its warning says what it gave up on."""
    started = asyncio.Event()
    tasks: list[asyncio.Task[Any]] = []

    async def stubborn(_config: GatewayConfig) -> None:
        task = asyncio.current_task()
        assert task is not None
        tasks.append(task)
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # Absorbed once, as a nested cancel scope around an outbound call can.
            task.uncancel()
        await asyncio.sleep(3600)

    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=True, worker=stubborn),))
    app = create_app(_standalone(tmp_path))
    gateway_logger = logging.getLogger("gateway")
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        async with _create_lifespan()(app):
            await asyncio.wait_for(started.wait(), timeout=5)
    finally:
        gateway_logger.removeHandler(caplog.handler)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert "probe worker did not stop within" in caplog.text


@pytest.mark.asyncio
async def test_a_hybrid_gateway_starts_no_feature_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone only, like the refreshers: a hybrid gateway has no local database for a worker to work on."""
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    probe = _Probe()
    monkeypatch.setattr(features, "CORE_FEATURES", (_probe(enabled=True, worker=probe.run),))
    app = create_app(_hybrid())

    async with _create_lifespan()(app):
        await asyncio.sleep(0)

    assert not probe.started.is_set()


@pytest.mark.parametrize("built_enabled", [True, False], ids=["switched-off-later", "switched-on-later"])
def test_a_switch_changed_after_the_app_is_built_moves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, built_enabled: bool
) -> None:
    """The routers, the surface and the worker all keep the answer ``enabled`` gave when the app was built.

    A setting that changes later, as a stored dashboard override does at startup, must not move one of them
    without the others: a feature switched off there would otherwise keep answering and working with no page.
    """
    switch = {"on": built_enabled}
    started = threading.Event()

    async def worker(_config: GatewayConfig) -> None:
        started.set()
        await asyncio.sleep(3600)

    feature = replace(_probe(enabled=True, worker=worker), enabled=lambda _config: switch["on"])
    monkeypatch.setattr(features, "CORE_FEATURES", (feature,))
    app = create_app(_standalone(tmp_path))
    switch["on"] = not built_enabled

    with TestClient(app) as client:
        probe_status = client.get(f"{API_ROOT}/probe").status_code
        surfaces = client.get(f"{API_ROOT}/bootstrap").json()["surfaces"]
        worker_started = started.is_set()

    assert probe_status == (200 if built_enabled else 404)
    assert ("probe" in surfaces) is built_enabled
    assert worker_started is built_enabled
