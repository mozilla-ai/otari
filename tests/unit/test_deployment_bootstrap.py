"""Unit tests for the deployment bootstrap the dashboard shell renders from.

The shell fetches this before it renders anything, so it decides whether a
sign-in screen or a management dashboard is the
right thing to show, and which credential that sign-in screen should ask for.
Unit rather than integration because the one database read the route makes runs
on the SQLite file each test stands up, so there is no PostgreSQL to wait for.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.routing import iter_route_contexts
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from gateway.api.deps import reset_config
from gateway.api.routes import bootstrap as bootstrap_route
from gateway.api.routes.bootstrap import SURFACES, published_surfaces
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.services.secret_box import generate_secret_key

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


def _gateway_config(
    tmp_path: Path,
    docs_url: str | None = None,
    *,
    terms_url: str | None = None,
    privacy_url: str | None = None,
    site_url: str | None = None,
) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'bootstrap.db'}",
        master_key=MASTER_KEY,
        docs_url=docs_url,
        site_url=site_url,
        terms_url=terms_url,
        privacy_url=privacy_url,
    )


def test_the_bootstrap_reports_the_full_surface_set(tmp_path: Path) -> None:
    app = create_app(_gateway_config(tmp_path))

    with TestClient(app) as client:
        response = client.get(f"{API_ROOT}/bootstrap")

    assert response.status_code == 200
    assert response.json() == {
        # The trace explorer and the Health page are core features, published on top of the declared surfaces
        # while they are on.
        "surfaces": sorted({*SURFACES, "health_checks", "traces"}),
        "sign_in_methods": ["master_key"],
        "docs_url": None,
        "terms_url": None,
        "privacy_url": None,
        "site_url": None,
        "maintenance_mode": False,
        "passkeys_enabled": True,
        "passkeys_ready": False,
        "oauth_providers": [],
        "feedback_enabled": True,
        "federation_member_rules": False,
        "mail_ready": False,
        "public_catalog": False,
        "open_signup": False,
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
    app = create_app(_gateway_config(tmp_path))

    with TestClient(app) as client:
        response = client.get(f"{API_ROOT}/bootstrap")

    assert response.status_code == 200
    assert response.json()["sign_in_methods"] == []


def test_bootstrap_needs_no_credential(tmp_path: Path) -> None:
    """A browser cannot sign in before it knows whether signing in is possible.

    The request above already omits every credential; this asserts the master key
    the config sets is genuinely not being required, rather than the route
    happening to be unauthenticated because nothing protects it.
    """
    app = create_app(_gateway_config(tmp_path))

    with TestClient(app) as client:
        anonymous = client.get(f"{API_ROOT}/bootstrap")
        # Named to contrast with /api/v1/settings, which the same client cannot read.
        settings = client.get(f"{API_ROOT}/settings")

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
    unconfigured = _gateway_config(tmp_path)

    with TestClient(create_app(unconfigured)) as client:
        answered = client.get(f"{API_ROOT}/bootstrap").json()
        assert answered["passkeys_ready"] is False
        assert "passkey" not in answered["sign_in_methods"]

    reset_config()
    reset_db()

    # An address is the whole requirement: the relying-party ID derives from it.
    ready = _gateway_config(tmp_path)
    ready.public_base_url = "https://otari.example.com"

    with TestClient(create_app(ready)) as client:
        answered = client.get(f"{API_ROOT}/bootstrap").json()
        assert answered["passkeys_ready"] is True
        # Still no registered passkey, so it is not offered as a sign-in yet.
        assert "passkey" not in answered["sign_in_methods"]


def test_a_deployment_can_turn_passkeys_off_entirely(tmp_path: Path) -> None:
    """The switch is the setting, not the absence of an address.

    An address is also what OAuth redirects and mail links are built from, so a
    deployment that serves those cannot switch passkeys off by leaving one out.
    With the setting off the deployment says so, publishes no passkey sign-in,
    and takes the ceremony and management routes out of the API altogether.
    """
    off = _gateway_config(tmp_path)
    off.public_base_url = "https://otari.example.com"
    off.passkeys_enabled = False

    with TestClient(create_app(off)) as client:
        answered = client.get(f"{API_ROOT}/bootstrap").json()
        assert answered["passkeys_enabled"] is False
        assert answered["passkeys_ready"] is False
        assert "passkey" not in answered["sign_in_methods"]
        # Not mounted, so a 404 rather than the 503 a missing relying party gives.
        for method, path in (
            ("post", "/auth/webauthn/authenticate/options"),
            ("post", "/auth/webauthn/authenticate"),
            ("post", "/auth/webauthn/register/options"),
            ("get", "/auth/webauthn/credentials"),
            ("delete", "/auth/webauthn/credentials/00000000-0000-0000-0000-000000000000"),
        ):
            response = getattr(client, method)(f"{API_ROOT}{path}")
            assert response.status_code == 404, (method, path, response.status_code)


def test_passkeys_stay_on_by_default_and_mounted(tmp_path: Path) -> None:
    on = _gateway_config(tmp_path)
    on.public_base_url = "https://otari.example.com"

    with TestClient(create_app(on)) as client:
        answered = client.get(f"{API_ROOT}/bootstrap").json()
        assert answered["passkeys_enabled"] is True
        assert answered["passkeys_ready"] is True
        # Mounted: refused for want of a session, not for want of a route.
        assert client.get(f"{API_ROOT}/auth/webauthn/credentials").status_code in {401, 403}


def test_member_federation_rules_are_published_when_the_setting_is_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    on = _gateway_config(tmp_path)
    on.federation_member_rules = True

    with TestClient(create_app(on)) as client:
        assert client.get(f"{API_ROOT}/bootstrap").json()["federation_member_rules"] is True


def test_mail_ready_turns_on_only_with_a_transport_and_a_public_url(tmp_path: Path) -> None:
    """What the dashboard gates a mail-dependent affordance on.

    A transport alone is not enough: every message the gateway sends
    carries a link back into this deployment, and it needs its own address to
    build one.
    """
    transport_only = _gateway_config(tmp_path)
    transport_only.smtp_host = "smtp.example.com"
    transport_only.mail_from_email = "otari@example.com"

    with TestClient(create_app(transport_only)) as client:
        assert client.get(f"{API_ROOT}/bootstrap").json()["mail_ready"] is False

    reset_config()
    reset_db()

    ready = _gateway_config(tmp_path)
    ready.smtp_host = "smtp.example.com"
    ready.mail_from_email = "otari@example.com"
    ready.public_base_url = "https://otari.example.com"

    with TestClient(create_app(ready)) as client:
        assert client.get(f"{API_ROOT}/bootstrap").json()["mail_ready"] is True


def test_open_signup_is_published_only_with_mail_to_carry_the_verification(tmp_path: Path) -> None:
    """Both halves, because the page reads this to decide whether to invite registrations.

    The route refuses without a transport anyway, so publishing the setting
    alone would advertise a form whose every submit answers 503.
    """
    no_mail = _gateway_config(tmp_path)
    no_mail.open_signup = True

    with TestClient(create_app(no_mail)) as client:
        assert client.get(f"{API_ROOT}/bootstrap").json()["open_signup"] is False

    reset_config()
    reset_db()

    ready = _gateway_config(tmp_path)
    ready.open_signup = True
    ready.smtp_host = "smtp.example.com"
    ready.mail_from_email = "otari@example.com"
    ready.public_base_url = "https://otari.example.com"

    with TestClient(create_app(ready)) as client:
        assert client.get(f"{API_ROOT}/bootstrap").json()["open_signup"] is True


# A surface names its router's ``/api/v1/`` prefix, so the prefix is derived from the
# name. Three are nested rather than top-level and cannot be: they hang off
# ``/api/v1/organizations``, and naming them ``organizations`` would collapse them
# into the roster surface, which is a different page with different access. Listed
# here rather than in the surface tuple itself so the tuple stays the plain list of
# names the dashboard gates on.
#
# ``organization_guardrails`` covers two routers, the mandates and the definitions,
# because one page shows both. A surface maps to one prefix here, so the mandates
# are what it names and the definitions prefix is deliberately unmapped: what this
# check is for is a surface whose API went away, and a name that reaches one
# mounted route has not.
SURFACE_ROUTE_PREFIXES = {
    "hosted_providers": f"{API_ROOT}/hosted-providers",
    "health_checks": f"{API_ROOT}/health-checks",
    "organization_guardrails": f"{API_ROOT}/organizations/me/guardrails",
    "organization_providers": f"{API_ROOT}/organizations/me/provider-keys",
    "organization_usage": f"{API_ROOT}/organizations/me/usage",
}


def test_the_organization_guardrail_surface_is_published(tmp_path: Path) -> None:
    """The row that lets an organization see its own guardrails at all.

    Both routers are asserted, because one name covers them and only one of the
    two can be in the prefix map above.
    """
    assert "organization_guardrails" in SURFACES

    mounted = {route.path or "" for route in iter_route_contexts(create_app(_gateway_config(tmp_path)).routes)}

    assert any(path.startswith(f"{API_ROOT}/organizations/me/guardrails") for path in mounted)
    assert any(path.startswith(f"{API_ROOT}/organizations/me/guardrail-definitions") for path in mounted)


def test_the_organization_usage_surface_is_published(tmp_path: Path) -> None:
    """The organization-wide Usage page, which no other page answers.

    ``/usage`` reads the workspace the sidebar selected, so a deployment serving
    several organizations needs this page too.
    """
    with TestClient(create_app(_gateway_config(tmp_path))) as client:
        answered = client.get(f"{API_ROOT}/bootstrap").json()

    assert "organization_usage" in answered["surfaces"]


def test_every_surface_names_a_route_the_gateway_mounts(tmp_path: Path) -> None:
    """A surface that outlives its API would gate a nav item onto a 404."""
    app = create_app(_gateway_config(tmp_path))
    mounted = {route.path or "" for route in iter_route_contexts(app.routes)}

    # The fixed tuple and what the endpoint publishes, which adds each enabled
    # registry feature's surface: a feature whose route is not mounted fails here.
    for surface in {*SURFACES, *published_surfaces(app.state.enabled_features)}:
        prefix = SURFACE_ROUTE_PREFIXES.get(surface, f"{API_ROOT}/{surface}")
        assert any(path.startswith(prefix) for path in mounted), f"surface {surface!r} names no mounted /api/v1/ route"


def test_a_deployment_with_no_docs_url_points_at_the_bundled_guide(tmp_path: Path) -> None:
    """Null is the answer the dashboard reads as "use the bundled guide", not a missing field."""
    app = create_app(_gateway_config(tmp_path))

    with TestClient(app) as client:
        response = client.get(f"{API_ROOT}/bootstrap")

    assert response.json()["docs_url"] is None


def test_docs_url_is_published_to_the_dashboard(tmp_path: Path) -> None:
    """The retargeted Documentation link, published exactly as configured.

    No trailing slash is trimmed and no path is appended: this is the whole
    destination, and a docs site can need its trailing slash to resolve.
    """
    app = create_app(_gateway_config(tmp_path, docs_url="https://docs.otari.ai/en/"))

    with TestClient(app) as client:
        response = client.get(f"{API_ROOT}/bootstrap")

    assert response.json()["docs_url"] == "https://docs.otari.ai/en/"


def test_docs_url_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OTARI_DOCS_URL is the whole configuration surface a container needs."""
    monkeypatch.setenv("OTARI_DOCS_URL", "https://docs.otari.ai/en/")

    assert GatewayConfig(database_url=f"sqlite:///{tmp_path / 'env.db'}").docs_url == "https://docs.otari.ai/en/"


@pytest.mark.parametrize("configured", ["javascript:alert(1)", "docs.otari.ai", "/docs"])
def test_a_docs_url_that_is_not_an_http_link_is_refused_at_load(configured: str) -> None:
    """The browser turns this into an anchor, so a bad scheme is a config error.

    Refused where the config is built rather than where the app is: it is a
    field of its own, so pydantic can validate it.
    """
    with pytest.raises(ValidationError, match="docs_url"):
        GatewayConfig(docs_url=configured)


@pytest.mark.parametrize("configured", ["", "   "])
def test_a_blank_docs_url_is_an_unset_one(configured: str) -> None:
    """A container templating an empty value has not configured a docs site."""
    assert GatewayConfig(docs_url=configured).docs_url is None


def test_a_deployment_with_no_legal_urls_leaves_the_account_menu_as_it_was(tmp_path: Path) -> None:
    """Null both, which is what a self-hosted gateway publishes.

    The dashboard reads that as no Terms of service row and a Data & Privacy row
    that stays disabled with the reason it has always carried, rather than as a
    missing field.
    """
    app = create_app(_gateway_config(tmp_path))

    with TestClient(app) as client:
        body = client.get(f"{API_ROOT}/bootstrap").json()

    assert body["terms_url"] is None
    assert body["privacy_url"] is None


def test_a_deployment_publishes_the_legal_pages_on_its_own_site(tmp_path: Path) -> None:
    """otari-ai#1945: the privacy notice needs a home the composed dashboard can reach.

    A deployment that serves its dashboard beside a site that owns the notice and
    the terms publishes both addresses so the account menu can name them.
    """
    app = create_app(
        _gateway_config(
            tmp_path,
            terms_url="https://otari.ai/terms",
            privacy_url="https://otari.ai/privacy",
        )
    )

    with TestClient(app) as client:
        body = client.get(f"{API_ROOT}/bootstrap").json()

    assert "management_url" not in body
    assert body["terms_url"] == "https://otari.ai/terms"
    assert body["privacy_url"] == "https://otari.ai/privacy"


def test_site_url_is_published_for_the_public_catalogs_logo(tmp_path: Path) -> None:
    """Where the logo on the pages ahead of a session goes, for a deployment with a site of its own."""
    app = create_app(_gateway_config(tmp_path, site_url="https://otari.ai/"))

    with TestClient(app) as client:
        body = client.get(f"{API_ROOT}/bootstrap").json()

    assert body["site_url"] == "https://otari.ai/"


def test_site_url_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OTARI_SITE_URL", "https://otari.ai/")
    assert GatewayConfig(database_url=f"sqlite:///{tmp_path / 'env.db'}").site_url == "https://otari.ai/"


@pytest.mark.parametrize("configured", ["javascript:alert(1)", "otari.ai", "/"])
def test_a_site_url_that_is_not_an_http_link_is_refused_at_load(configured: str) -> None:
    with pytest.raises(ValidationError, match="site_url"):
        GatewayConfig(site_url=configured)


@pytest.mark.parametrize("field", ["terms_url", "privacy_url"])
def test_a_legal_url_is_read_from_the_environment(field: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OTARI_TERMS_URL and OTARI_PRIVACY_URL are the whole configuration surface a container needs."""
    monkeypatch.setenv(f"OTARI_{field.upper()}", "https://otari.ai/legal")

    config = GatewayConfig(database_url=f"sqlite:///{tmp_path / 'env.db'}")

    assert getattr(config, field) == "https://otari.ai/legal"


@pytest.mark.parametrize("field", ["terms_url", "privacy_url"])
@pytest.mark.parametrize("configured", ["javascript:alert(1)", "otari.ai/terms", "/terms"])
def test_a_legal_url_that_is_not_an_http_link_is_refused_at_load(field: str, configured: str) -> None:
    """Held to ``docs_url``'s bar, and by the same validator: the browser turns each into an anchor."""
    with pytest.raises(ValidationError, match=field):
        # Built through model_validate so the field name can be a parameter; the
        # constructor's keywords are typed one field at a time.
        GatewayConfig.model_validate({field: configured})


@pytest.mark.parametrize("field", ["terms_url", "privacy_url"])
def test_a_blank_legal_url_is_an_unset_one(field: str) -> None:
    """A container templating an empty value has published no legal page."""
    assert getattr(GatewayConfig.model_validate({field: "   "}), field) is None


@pytest.mark.parametrize("field", ["docs_url", "terms_url", "privacy_url", "site_url"])
@pytest.mark.parametrize(
    "configured",
    ["https://token@otari.ai/terms", "https://user:secret@otari.ai/terms"],
)
def test_a_link_url_carrying_a_credential_is_refused_at_load(field: str, configured: str) -> None:
    """``GET /api/v1/bootstrap`` is unauthenticated, so a credential written into any
    of these link fields would reach every browser that asked for it, which no
    redaction in the operator-gated config viewer would cover. ``docs_url`` is
    covered too: it rides the same validator and the same response.
    """
    with pytest.raises(ValidationError, match="no username or password"):
        GatewayConfig.model_validate({field: configured})


def test_the_unauthenticated_bootstrap_cannot_publish_a_legal_page_credential(tmp_path: Path) -> None:
    """The end the refusal above exists to protect, asserted through the route.

    The validator test says the value cannot be built; this says the published
    payload is what a browser gets, so the two cannot drift apart if somebody
    later relaxes one of them.
    """
    app = create_app(
        _gateway_config(
            tmp_path,
            terms_url="https://otari.ai/terms",
            privacy_url="https://otari.ai/privacy",
        )
    )

    with TestClient(app) as client:
        body = client.get(f"{API_ROOT}/bootstrap").text

    for field in ('"terms_url"', '"privacy_url"'):
        assert "@" not in body.split(field)[1].split(",")[0]
