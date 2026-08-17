"""Unit tests for the deployment bootstrap the dashboard shell renders from.

The shell fetches this before it renders anything, so it decides whether a
sign-in screen, a management dashboard, or a data-plane landing page is the
right thing to show. Unit rather than integration because the route reads
configuration and nothing else: no database, so no PostgreSQL to stand up.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.api.routes.bootstrap import STANDALONE_CAPABILITIES
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app

PLATFORM_TOKEN = "gw_test_token"
MASTER_KEY = "sk-master-not-in-the-bootstrap"


def _standalone(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'bootstrap.db'}",
        master_key=MASTER_KEY,
    )


def _hybrid(**platform: str) -> GatewayConfig:
    return GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://localhost:8100/api/v1", **platform},
    )


def test_standalone_reports_a_local_operator_and_the_full_capability_set(tmp_path: Path) -> None:
    app = create_app(_standalone(tmp_path))

    with TestClient(app) as client:
        response = client.get("/v1/bootstrap")

    assert response.status_code == 200
    assert response.json() == {
        "deployment_type": "standalone",
        "session_type": "local_operator",
        "capabilities": sorted(STANDALONE_CAPABILITIES),
        "management_url": None,
    }


def test_bootstrap_needs_no_credential(tmp_path: Path) -> None:
    """A browser cannot sign in before it knows whether signing in is possible.

    The request above already omits every credential; this asserts the master key
    the config sets is genuinely not being required, rather than the route
    happening to be unauthenticated because nothing protects it.
    """
    app = create_app(_standalone(tmp_path))

    with TestClient(app) as client:
        anonymous = client.get("/v1/bootstrap")
        # Named to contrast with /v1/settings, which the same client cannot read.
        settings = client.get("/v1/settings")

    assert anonymous.status_code == 200
    assert settings.status_code == 401
    # Unauthenticated but not cacheable: the answer describes this gateway's
    # configuration, and a shared cache must not serve one gateway's to another.
    assert anonymous.headers["Cache-Control"] == "private, no-store, no-cache"


def test_every_capability_names_a_route_the_gateway_mounts(tmp_path: Path) -> None:
    """A capability that outlives its API would gate a surface onto a 404."""
    app = create_app(_standalone(tmp_path))
    mounted = {getattr(route, "path", "") for route in app.routes}

    for capability in STANDALONE_CAPABILITIES:
        assert any(path.startswith(f"/v1/{capability}") for path in mounted), (
            f"capability {capability!r} names no mounted /v1/ route"
        )


def test_hybrid_reports_no_session_no_capabilities_and_the_hosted_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    app = create_app(_hybrid())

    with TestClient(app) as client:
        response = client.get("/v1/bootstrap")

    assert response.status_code == 200
    assert response.json() == {
        "deployment_type": "hybrid",
        "session_type": "none",
        "capabilities": [],
        "management_url": "https://otari.ai",
    }

    reset_config()
    reset_db()


def test_hybrid_bootstrap_leaks_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one route a hybrid gateway serves to an unauthenticated browser."""
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    app = create_app(_hybrid())

    with TestClient(app) as client:
        body = client.get("/v1/bootstrap").text

    assert PLATFORM_TOKEN not in body

    reset_config()
    reset_db()


def test_management_url_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator on a staging platform links at that platform, not otari.ai."""
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    app = create_app(_hybrid(management_url="https://staging.otari.example/"))

    with TestClient(app) as client:
        response = client.get("/v1/bootstrap")

    assert response.json()["management_url"] == "https://staging.otari.example/"

    reset_config()
    reset_db()


@pytest.mark.parametrize("configured", ["javascript:alert(1)", "otari.ai", ""])
def test_a_management_url_that_is_not_an_http_link_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch, configured: str
) -> None:
    """The browser turns this into an anchor, so a bad scheme is a startup error.

    The empty string falls back to the default rather than failing, so it is here
    to prove that: a blank override is an unset override, not a broken one.
    """
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)

    if configured:
        with pytest.raises(ValueError, match="management_url"):
            create_app(_hybrid(management_url=configured))
    else:
        app = create_app(_hybrid(management_url=configured))
        with TestClient(app) as client:
            assert client.get("/v1/bootstrap").json()["management_url"] == "https://otari.ai"

    reset_config()
    reset_db()
