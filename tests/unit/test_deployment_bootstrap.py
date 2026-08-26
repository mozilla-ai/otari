"""Unit tests for the deployment bootstrap the dashboard shell renders from.

The shell fetches this before it renders anything, so it decides whether a
sign-in screen, a management dashboard, or a data-plane landing page is the
right thing to show, and which credential that sign-in screen should ask for.
Unit rather than integration because the one database read the route makes runs
on the SQLite file each test stands up, so there is no PostgreSQL to wait for.
"""

from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from gateway.api.deps import reset_config
from gateway.api.routes import bootstrap as bootstrap_route
from gateway.api.routes.bootstrap import HOSTED_SURFACES, STANDALONE_SURFACES
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app

PLATFORM_TOKEN = "gw_test_token"
MASTER_KEY = "sk-master-not-in-the-bootstrap"


@pytest.fixture(autouse=True)
def _reset_process_state() -> Generator[None, None, None]:
    """Return the process-wide config and engine to their defaults after each test.

    ``create_app`` installs both, and nothing in the root ``conftest`` puts them
    back. Every test here used to end with the pair by hand, which is the same
    guarantee right up until an assertion fails: a trailing statement is skipped
    when the test does not reach it, and a finalizer is not.

    Hardening rather than a fix for an observed failure, and worth saying so: no
    leak in this file is currently reachable that way, because every test builds
    its own app and ``create_app`` installs a config over whatever the last one
    left. That is a property of these tests, not a guarantee, and it is the kind
    that stops holding quietly.

    The pairs left inline below are the ones that are not cleanup: a test
    comparing two deployments has to put the first away before it builds the
    second, and that has to happen mid-test.
    """
    yield
    reset_config()
    reset_db()


def _standalone(tmp_path: Path, docs_url: str | None = None) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'bootstrap.db'}",
        master_key=MASTER_KEY,
        docs_url=docs_url,
    )


def _hosted(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        mode="hosted",
        database_url=f"sqlite:///{tmp_path / 'bootstrap.db'}",
        master_key=MASTER_KEY,
    )


def _hybrid(*, docs_url: str | None = None, **platform: str) -> GatewayConfig:
    return GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://localhost:8100/api/v1", **platform},
        docs_url=docs_url,
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
        "docs_url": None,
        "maintenance_mode": False,
        "passkeys_ready": False,
        "oauth_providers": [],
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


# A surface names its router's ``/v1/`` prefix, so the prefix is derived from the
# name. One is nested rather than top-level and cannot be: the organization's own
# provider keys hang off ``/v1/organizations``, and naming them ``organizations``
# would collapse them into the roster surface, which is a different page with
# different access. Listed here rather than in the surface tuple itself so the
# tuple stays the plain list of names the dashboard gates on.
SURFACE_ROUTE_PREFIXES = {"organization_providers": "/v1/organizations/me/provider-keys"}


@pytest.mark.parametrize(
    ("build", "surfaces"),
    [(_standalone, STANDALONE_SURFACES), (_hosted, HOSTED_SURFACES)],
    ids=["standalone", "hosted"],
)
def test_every_surface_names_a_route_the_gateway_mounts(
    tmp_path: Path,
    build: Callable[[Path], GatewayConfig],
    surfaces: tuple[str, ...],
) -> None:
    """A surface that outlives its API would gate a nav item onto a 404.

    Each edition is asked about its own app rather than both about standalone's.
    They mount the identical router set today, so the two runs are the same
    assertion twice; the point is that an edition which stopped mounting one
    would be caught here rather than in a browser.
    """
    app = create_app(build(tmp_path))
    mounted = {getattr(route, "path", "") for route in app.routes}

    for surface in surfaces:
        prefix = SURFACE_ROUTE_PREFIXES.get(surface, f"/v1/{surface}")
        assert any(path.startswith(prefix) for path in mounted), f"surface {surface!r} names no mounted /v1/ route"



def test_hosted_swaps_the_process_wide_provider_page_for_the_per_organization_one(tmp_path: Path) -> None:
    """The whole point of the hosted surface set, in the two rows that differ.

    ``provider_credentials`` is keyed on the instance name alone, so a credential
    added through ``/providers`` is served to every organization on the
    deployment and shadows that organization's own BYO key. The page is right for
    the single-tenant product and wrong for a control plane, and the
    organization-scoped one is the other way around.
    """
    app = create_app(_hosted(tmp_path))

    with TestClient(app) as client:
        answered = client.get("/v1/bootstrap").json()

    assert answered["deployment_type"] == "hosted"
    # Still this deployment's own sign-in: "hosted_user" is a session minted by
    # somebody else's account system, which no build here does.
    assert answered["session_type"] == "local_operator"
    assert "organization_providers" in answered["surfaces"]
    assert "providers" not in answered["surfaces"]
    # Everything else is standalone's set, so a surface added there is not
    # silently withheld from a control plane.
    assert set(answered["surfaces"]) ^ set(STANDALONE_SURFACES) == {"organization_providers", "providers"}



def test_hosted_answers_everything_below_the_edition_the_way_standalone_does(tmp_path: Path) -> None:
    """Hosted mode is standalone's multi-tenant sibling, not a third data plane.

    It owns its own database and mints its own sessions, so a change to hosted
    mode that also changed how somebody signs in, or whether the deployment is
    frozen, would be a change nobody asked for.
    """
    standalone_app = create_app(_standalone(tmp_path))
    with TestClient(standalone_app) as client:
        standalone = client.get("/v1/bootstrap").json()

    reset_config()
    reset_db()

    hosted_app = create_app(_hosted(tmp_path))
    with TestClient(hosted_app) as client:
        hosted = client.get("/v1/bootstrap").json()

    differ = {key for key in standalone if standalone[key] != hosted[key]}
    assert differ == {"deployment_type", "surfaces"}



def test_hosted_mode_refuses_a_platform_token(tmp_path: Path) -> None:
    """The same conflict standalone already refuses, for the same reason.

    A deployment that holds its own management API is not also a data plane
    reporting to somebody else's control plane, so the two settings cannot both
    be meant.
    """
    config = _hosted(tmp_path)
    config._platform_token = PLATFORM_TOKEN
    config._platform_token_resolved = True

    with pytest.raises(ValueError, match="conflicts with OTARI_AI_TOKEN"):
        config.validate_mode_selection()


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
        "docs_url": None,
        "maintenance_mode": False,
        "passkeys_ready": False,
        "oauth_providers": [],
        "mail_ready": False,
    }



def test_hybrid_bootstrap_leaks_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one route a hybrid gateway serves to an unauthenticated browser."""
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    app = create_app(_hybrid())

    with TestClient(app) as client:
        body = client.get("/v1/bootstrap").text

    assert PLATFORM_TOKEN not in body



def test_management_url_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator on a staging platform links at that platform, not otari.ai."""
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    app = create_app(_hybrid(management_url="https://staging.otari.example/"))

    with TestClient(app) as client:
        response = client.get("/v1/bootstrap")

    assert response.json()["management_url"] == "https://staging.otari.example/"



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



def test_a_deployment_with_no_docs_url_points_at_the_bundled_guide(tmp_path: Path) -> None:
    """Null is the answer the dashboard reads as "use the bundled guide", not a missing field."""
    app = create_app(_standalone(tmp_path))

    with TestClient(app) as client:
        response = client.get("/v1/bootstrap")

    assert response.json()["docs_url"] is None



def test_docs_url_is_published_to_a_standalone_dashboard(tmp_path: Path) -> None:
    """The retargeted Documentation link, published exactly as configured.

    No trailing slash is trimmed and no path is appended: unlike ``management_url``,
    which the dashboard suffixes to reach ``/terms`` and ``/playground``, this is
    the whole destination, and a docs site can need its trailing slash to resolve.
    """
    app = create_app(_standalone(tmp_path, docs_url="https://docs.otari.ai/en/"))

    with TestClient(app) as client:
        response = client.get("/v1/bootstrap")

    assert response.json()["docs_url"] == "https://docs.otari.ai/en/"



def test_a_hybrid_gateway_carries_the_hosted_docs_link_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The setting is deployment-wide, not standalone-only.

    A gateway attached to otari.ai serves the same shell, and its operator reads
    the hosted documentation rather than the guide bundled for a self-hosted one.
    """
    monkeypatch.setenv("OTARI_AI_TOKEN", PLATFORM_TOKEN)
    app = create_app(_hybrid(docs_url="https://docs.otari.ai/en/"))

    with TestClient(app) as client:
        response = client.get("/v1/bootstrap")

    assert response.json()["docs_url"] == "https://docs.otari.ai/en/"



def test_docs_url_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OTARI_DOCS_URL is the whole configuration surface a container needs."""
    monkeypatch.setenv("OTARI_DOCS_URL", "https://docs.otari.ai/en/")

    assert GatewayConfig(database_url=f"sqlite:///{tmp_path / 'env.db'}").docs_url == "https://docs.otari.ai/en/"


@pytest.mark.parametrize("configured", ["javascript:alert(1)", "docs.otari.ai", "/docs"])
def test_a_docs_url_that_is_not_an_http_link_is_refused_at_load(configured: str) -> None:
    """The browser turns this into an anchor, so a bad scheme is a config error.

    Refused where the config is built rather than where the app is, which is the
    one difference from ``platform.management_url``: that one is a key inside a
    free-form dict, while this is a field of its own and pydantic can validate it.
    """
    with pytest.raises(ValidationError, match="docs_url"):
        GatewayConfig(docs_url=configured)


@pytest.mark.parametrize("configured", ["", "   "])
def test_a_blank_docs_url_is_an_unset_one(configured: str) -> None:
    """A container templating an empty value has not configured a docs site."""
    assert GatewayConfig(docs_url=configured).docs_url is None
