"""Unit tests for the deployment bootstrap the dashboard shell renders from.

The shell fetches this before it renders anything, so it decides whether a
sign-in screen, a management dashboard, or a data-plane landing page is the
right thing to show, and which credential that sign-in screen should ask for.
Unit rather than integration because the one database read the route makes runs
on the SQLite file each test stands up, so there is no PostgreSQL to wait for.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from gateway.api.deps import reset_config
from gateway.api.routes import bootstrap as bootstrap_route
from gateway.api.routes.bootstrap import STANDALONE_SURFACES
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


def test_standalone_reports_a_local_operator_and_the_full_surface_set(tmp_path: Path) -> None:
    app = create_app(_standalone(tmp_path))

    with TestClient(app) as client:
        response = client.get("/v1/bootstrap")

    assert response.status_code == 200
    assert response.json() == {
        "deployment_type": "standalone",
        "session_type": "local_operator",
        "surfaces": sorted(STANDALONE_SURFACES),
        "sign_in_methods": ["master_key"],
        "management_url": None,
        "maintenance_mode": False,
        "passkeys_ready": False,
        "mail_ready": False,
    }


def test_a_database_outage_reports_no_sign_in_rather_than_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shell fetches this first, so a 500 here is a blank page, not a login screen.

    "No sign-in method" is also the truth while the database is unreachable:
    minting a session writes a row, so neither credential could get anyone in.
    """

    async def _unavailable(_db: object) -> bool:
        raise SQLAlchemyError("database is down")

    monkeypatch.setattr(bootstrap_route, "operator_has_password", _unavailable)
    app = create_app(_standalone(tmp_path))

    with TestClient(app) as client:
        response = client.get("/v1/bootstrap")

    assert response.status_code == 200
    assert response.json()["deployment_type"] == "standalone"
    assert response.json()["sign_in_methods"] == []


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


def test_passkeys_ready_turns_on_with_an_address_alone(tmp_path: Path) -> None:
    """What the account page gates its "add a passkey" form on.

    Distinct from ``passkey`` in ``sign_in_methods``, which is narrower: that
    one also requires a passkey to exist, so gating the card on it would hide
    the form from the operator about to register the first one. This answers the
    prior question, whether a ceremony could run here at all.
    """
    unconfigured = _standalone(tmp_path)

    with TestClient(create_app(unconfigured)) as client:
        answered = client.get("/v1/bootstrap").json()
        assert answered["passkeys_ready"] is False
        assert "passkey" not in answered["sign_in_methods"]

    reset_config()
    reset_db()

    # An address is the whole requirement: the relying-party ID derives from it.
    ready = _standalone(tmp_path)
    ready.public_base_url = "https://otari.example.com"

    with TestClient(create_app(ready)) as client:
        answered = client.get("/v1/bootstrap").json()
        assert answered["passkeys_ready"] is True
        # Still no registered passkey, so it is not offered as a sign-in yet.
        assert "passkey" not in answered["sign_in_methods"]

    reset_config()
    reset_db()


def test_mail_ready_turns_on_only_with_a_transport_and_a_public_url(tmp_path: Path) -> None:
    """What the dashboard gates a mail-dependent affordance on.

    A transport alone is not enough: every message the control plane sends
    carries a link back into this deployment, and it needs its own address to
    build one.
    """
    transport_only = _standalone(tmp_path)
    transport_only.smtp_host = "smtp.example.com"
    transport_only.mail_from_email = "otari@example.com"

    with TestClient(create_app(transport_only)) as client:
        assert client.get("/v1/bootstrap").json()["mail_ready"] is False

    reset_config()
    reset_db()

    ready = _standalone(tmp_path)
    ready.smtp_host = "smtp.example.com"
    ready.mail_from_email = "otari@example.com"
    ready.public_base_url = "https://otari.example.com"

    with TestClient(create_app(ready)) as client:
        assert client.get("/v1/bootstrap").json()["mail_ready"] is True

    reset_config()
    reset_db()


def test_every_surface_names_a_route_the_gateway_mounts(tmp_path: Path) -> None:
    """A surface that outlives its API would gate a nav item onto a 404."""
    app = create_app(_standalone(tmp_path))
    mounted = {getattr(route, "path", "") for route in app.routes}

    for surface in STANDALONE_SURFACES:
        assert any(path.startswith(f"/v1/{surface}") for path in mounted), (
            f"surface {surface!r} names no mounted /v1/ route"
        )


def test_hybrid_reports_no_session_no_surfaces_and_the_hosted_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    app = create_app(_hybrid())

    with TestClient(app) as client:
        response = client.get("/v1/bootstrap")

    assert response.status_code == 200
    assert response.json() == {
        "deployment_type": "hybrid",
        "session_type": "none",
        "surfaces": [],
        "sign_in_methods": [],
        "management_url": "https://otari.ai",
        "maintenance_mode": False,
        "passkeys_ready": False,
        "mail_ready": False,
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
