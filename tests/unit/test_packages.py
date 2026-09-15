"""The core package registry and the three wiring points that read it.

A listed package mounts its routers as core routes, hosts its surface, and runs
its worker under the lifespan's supervisor, each only when its own ``enabled``
says so. The registry is empty today, so a probe package stands in for one.
"""

import ast
import asyncio
import importlib.util
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from gateway import packages
from gateway.api.deps import reset_config
from gateway.api.routes.bootstrap import HOSTED_SURFACES, STANDALONE_SURFACES, hosted_surfaces
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.core.package import CorePackage, Worker
from gateway.main import _create_lifespan, create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "src" / "gateway" / "packages.py"
CHECK_SCRIPT_PATH = REPO_ROOT / "scripts" / "check_architecture.py"
PLATFORM_TOKEN = "gw_test_token"


@pytest.fixture(autouse=True)
def _reset_process_state() -> Generator[None, None, None]:
    """Put back the process-wide config and engine ``create_app`` and the lifespan install."""
    yield
    reset_config()
    reset_db()


def _standalone(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(database_url=f"sqlite:///{tmp_path / 'packages.db'}", master_key="sk-test-master")


def _hosted(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        mode="hosted", database_url=f"sqlite:///{tmp_path / 'packages.db'}", master_key="sk-test-master"
    )


def _hybrid() -> GatewayConfig:
    return GatewayConfig(mode="hybrid", platform={"base_url": "http://localhost:8100/api/v1"})


def _probe(*, enabled: bool, surface: str | None = "probe", worker: Worker | None = None) -> CorePackage:
    router = APIRouter(prefix="/probe")

    @router.get("")
    async def read_probe() -> dict[str, str]:
        return {"ok": "yes"}

    return CorePackage(
        name="probe",
        surface=surface,
        enabled=lambda _config: enabled,
        routers=lambda _config: (router,),
        worker=worker,
    )


def _load_check_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_architecture", CHECK_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_registry_is_a_literal_tuple() -> None:
    """Nothing scans and nothing computes: the line in the repo is the whole answer."""
    tree = ast.parse(REGISTRY_PATH.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "CORE_PACKAGES"
    ]
    assert len(assignments) == 1, "CORE_PACKAGES is assigned once, with its annotation"
    value = assignments[0].value
    assert isinstance(value, ast.Tuple)
    assert all(isinstance(element, ast.Attribute | ast.Name) for element in value.elts)


def test_the_boundary_check_lists_the_same_packages_as_the_registry() -> None:
    """The check script keys its package rules by name, so the two lists must agree."""
    check = _load_check_script()
    assert set(check.FEATURE_PACKAGES) == {package.name for package in packages.CORE_PACKAGES}


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_a_package_mounts_its_routers_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=enabled),))
    app = create_app(_standalone(tmp_path))
    mounted = {route.path for route in app.routes if isinstance(route, APIRoute)}
    assert (f"{API_ROOT}/probe" in mounted) is enabled


def test_a_package_route_is_a_core_route_with_no_capability_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listed package is part of the build, so its route answers as any core route does.

    A contributed router would sit behind ``require_capability``; a registry
    router does not, which is the difference between the two seams.
    """
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=True),))
    app = create_app(_standalone(tmp_path))

    with TestClient(app) as client:
        response = client.get(f"{API_ROOT}/probe")

    assert response.status_code == 200
    assert response.json() == {"ok": "yes"}


def test_a_hybrid_gateway_mounts_no_package_routers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Management plane only: hybrid returns before the management routers, and a package sits with them."""
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=True),))
    app = create_app(_hybrid())
    mounted = {route.path for route in app.routes if isinstance(route, APIRoute)}
    assert f"{API_ROOT}/probe" not in mounted


def test_an_enabled_package_hosts_its_surface_beside_the_fixed_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=True),))
    assert hosted_surfaces(_standalone(tmp_path)) == sorted((*STANDALONE_SURFACES, "probe"))
    assert hosted_surfaces(_hosted(tmp_path)) == sorted((*HOSTED_SURFACES, "probe"))


def test_a_disabled_package_hosts_no_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=False),))
    assert hosted_surfaces(_standalone(tmp_path)) == sorted(STANDALONE_SURFACES)


def test_a_package_with_no_page_hosts_no_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=True, surface=None),))
    assert hosted_surfaces(_standalone(tmp_path)) == sorted(STANDALONE_SURFACES)


def test_a_hybrid_gateway_hosts_no_surface_whatever_the_registry_says(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=True),))
    assert hosted_surfaces(_hybrid()) == []


def test_the_bootstrap_publishes_an_enabled_package_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The end the helper exists for: the dashboard learns about the page from here."""
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=True),))
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
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=True, worker=probe.run),))
    app = FastAPI()
    app.state.config = _standalone(tmp_path)

    async with _create_lifespan()(app):
        await asyncio.wait_for(probe.started.wait(), timeout=5)
        assert probe.config is app.state.config

    assert probe.task is not None
    assert probe.task.cancelled()


@pytest.mark.asyncio
async def test_the_lifespan_leaves_a_disabled_worker_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _Probe()
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=False, worker=probe.run),))
    app = FastAPI()
    app.state.config = _standalone(tmp_path)

    async with _create_lifespan()(app):
        await asyncio.sleep(0)

    assert not probe.started.is_set()


@pytest.mark.asyncio
async def test_a_hybrid_gateway_starts_no_package_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone only, like the refreshers: a hybrid gateway has no local database for a worker to work on."""
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    probe = _Probe()
    monkeypatch.setattr(packages, "CORE_PACKAGES", (_probe(enabled=True, worker=probe.run),))
    app = FastAPI()
    app.state.config = _hybrid()

    async with _create_lifespan()(app):
        await asyncio.sleep(0)

    assert not probe.started.is_set()
