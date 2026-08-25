"""The bootstrap hook, end to end on a running app.

``OTARI_BOOTSTRAP`` is the whole of how a build layers itself onto Otari
without editing an Otari source file: it names a callable that receives the
composition-root container after the core adapters are bound, and may rebind a
port and contribute routers. What that buys is only real on a booted app, so
this exercises both halves against one: a rebound ``EntitlementPort`` decides
which of two contributed routers answers, and the plain build (no selector set)
mounts neither.
"""

import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.container import Container
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app

from .conftest import build_test_client

# An overlay in miniature: it rebinds one port and contributes two routers, one
# gated on a capability it grants and one on a capability it does not. Written
# to a file and imported by dotted path, because the import is the mechanism
# under test.
OVERLAY_MODULE = '''
from fastapi import APIRouter

from gateway.container import Container, RouterContribution

from gateway.ports.entitlement_port import EntitlementPort

granted_router = APIRouter()
withheld_router = APIRouter()


@granted_router.get("/v1/overlay-probe")
async def overlay_probe() -> dict[str, str]:
    return {"source": "overlay"}


@withheld_router.get("/v1/overlay-withheld")
async def overlay_withheld() -> dict[str, str]:
    return {"source": "overlay"}


# Under a prefix the hybrid stub router claims with a ``{path:path}`` catch-all,
# so mounting order decides which one answers.
@granted_router.get("/v1/organizations/overlay-probe")
async def overlay_probe_under_a_stub_prefix() -> dict[str, str]:
    return {"source": "overlay"}


class ProbeEntitlementAdapter:
    """Grants one capability, so the other contributed router stays refused."""

    def __init__(self, session):
        self.session = session

    async def entitlements(self):
        return {"probe"}


def register(container: Container) -> None:
    container.bind(EntitlementPort, ProbeEntitlementAdapter)
    container.contribute_router(RouterContribution(capability="probe", router=granted_router))
    container.contribute_router(RouterContribution(capability="unlicensed", router=withheld_router))
'''


@pytest.fixture
def overlay_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[str]:
    """Write the overlay module somewhere importable and return its dotted path.

    Dropped from ``sys.modules`` on the way in, so this test's file is what
    loads, and again on the way out, so the module object does not outlive the
    ``tmp_path`` it was imported from. ``monkeypatch.delitem(..., raising=False)``
    does not manage the second: on a name that is absent it records no undo
    entry, so teardown would restore the first such test's module rather than
    clearing it.
    """
    (tmp_path / "otari_test_overlay.py").write_text(OVERLAY_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("otari_test_overlay", None)
    yield "otari_test_overlay"
    sys.modules.pop("otari_test_overlay", None)


@pytest.fixture
def bootstrap_config(postgres_url: str, overlay_on_path: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        bootstrap=f"{overlay_on_path}:register",
    )


@pytest.fixture
def bootstrap_client(bootstrap_config: GatewayConfig) -> Generator[TestClient]:
    yield from build_test_client(bootstrap_config)


def test_a_contributed_route_is_served_when_the_capability_is_entitled(
    bootstrap_client: TestClient,
) -> None:
    response = bootstrap_client.get("/v1/overlay-probe")

    assert response.status_code == 200
    assert response.json() == {"source": "overlay"}


def test_a_contributed_route_is_refused_when_the_capability_is_not_entitled(
    bootstrap_client: TestClient,
) -> None:
    # Mounted, but gated: the refusal is indistinguishable from a path nothing
    # serves, so the response does not disclose that the surface exists.
    response = bootstrap_client.get("/v1/overlay-withheld")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_the_rebound_port_is_what_the_gateway_resolves(bootstrap_client: TestClient) -> None:
    container: Container = bootstrap_client.app.state.container  # type: ignore[attr-defined]

    assert "otari_test_overlay:register rebound EntitlementPort" in container.summary
    assert "contributed routers for probe, unlicensed" in container.summary


def test_nothing_is_mounted_or_rebound_without_a_selector(client: TestClient) -> None:
    # The acceptance case for every deployment that runs no overlay: the same
    # app, built with OTARI_BOOTSTRAP unset, serves neither contributed path and
    # imports nothing.
    assert client.get("/v1/overlay-probe").status_code == 404
    assert client.get("/v1/overlay-withheld").status_code == 404
    container: Container = client.app.state.container  # type: ignore[attr-defined]
    assert container.router_contributions() == ()
    assert container.summary.startswith("no bootstrap, core defaults for ")


def test_a_contributed_route_wins_over_the_hybrid_stub_catch_all(
    overlay_on_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The hybrid stubs are ``{path:path}`` catch-alls over whole management
    # prefixes, and FastAPI serves the first matching route, so a contributed
    # route under one of them is only reachable if the stubs are mounted last.
    # Mounted earlier it would silently answer "manage this via the platform
    # UI" and the overlay's handler would never run.
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    config = GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://localhost:8100/api/v1"},
        bootstrap=f"{overlay_on_path}:register",
    )
    app = create_app(config)

    try:
        with TestClient(app) as client:
            response = client.get("/v1/organizations/overlay-probe")
            # The stub still answers a path the overlay does not serve.
            stub_response = client.get("/v1/organizations/something-else")
    finally:
        reset_config()
        reset_db()

    assert response.status_code == 200
    assert response.json() == {"source": "overlay"}
    assert stub_response.status_code == 404
    assert stub_response.json()["detail"].startswith("This endpoint is not available in hybrid mode")
