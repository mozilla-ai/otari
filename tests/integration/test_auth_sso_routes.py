"""The public sign-in routes for an organization's own OIDC connection, end to end.

Everything on this side of the exchange is real: the routes, the connection and domain-claim
rows, and the session cookie a sign-in mints. What is replaced is apron-auth's outbound half
(``auth_sso._exchange_sso_code``), the way ``tests/integration/test_oauth_api.py`` replaces
``auth_oauth.exchange_code``, because a test cannot complete a consent screen. A test about a
refusal that happens before any exchange (an unknown slug, a missing flow cookie, an issuer
mismatch) leaves it unstubbed, so the real check is what is proven.
"""

import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

import pytest
from apron_auth import ServerMetadata
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.identity import EmailAssertion, OIDCConnection, ProviderIdentity
from gateway.models.tenancy import Organization, OrganizationDomain
from gateway.rate_limit import RateLimiter
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME
from gateway.services.secret_box import encrypt_secret, generate_secret_key

from .conftest import build_test_client

ORIGIN = "https://testserver"
ISSUER = "https://sso.acme.example"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decrypting a connection's stored client secret needs a key configured."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


@pytest.fixture
def client(test_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    """A client whose base URL is ``ORIGIN``, matching ``public_base_url`` below."""
    test_config.public_base_url = ORIGIN
    yield from build_test_client(test_config, base_url=ORIGIN)


def _discovery_document(issuer: str = ISSUER, *, iss_parameter_supported: bool = True) -> dict[str, Any]:
    return ServerMetadata(
        authorize_url=f"{issuer}/authorize",
        token_url=f"{issuer}/token",
        jwks_url=f"{issuer}/jwks",
        userinfo_url=f"{issuer}/userinfo",
        issuer=issuer,
        iss_parameter_supported=iss_parameter_supported,
        code_challenge_methods=["S256"],
        token_endpoint_auth_methods=["client_secret_basic"],
    ).model_dump(mode="json")


def _add_organization(db_session: Session, *, slug: str) -> Organization:
    organization = Organization(name=slug.title(), slug=slug)
    db_session.add(organization)
    db_session.commit()
    db_session.refresh(organization)
    return organization


def _add_connection(
    db_session: Session,
    organization_id: uuid.UUID,
    *,
    issuer: str = ISSUER,
    enabled: bool = True,
) -> OIDCConnection:
    connection = OIDCConnection(
        organization_id=organization_id,
        issuer=issuer,
        resolved_issuer=issuer,
        client_id="a-client-id",
        encrypted_client_secret=encrypt_secret("a-client-secret"),
        client_secret_last4="cret",
        display_name="Acme SSO",
        discovery_document=_discovery_document(issuer),
        discovered_at=datetime.now(UTC),
        enabled=enabled,
    )
    db_session.add(connection)
    db_session.commit()
    db_session.refresh(connection)
    return connection


def _add_claim(db_session: Session, organization_id: uuid.UUID, *, domain: str) -> None:
    claim = OrganizationDomain(
        organization_id=organization_id,
        domain=domain,
        verification_token="tok" + uuid.uuid4().hex[:16],
        verified_at=datetime.now(UTC),
    )
    db_session.add(claim)
    db_session.commit()


def _acme(db_session: Session, *, slug: str = "acme") -> Organization:
    """An organization with an enabled connection and a proven claim on ``acme.example``."""
    organization = _add_organization(db_session, slug=slug)
    _add_connection(db_session, organization.id)
    _add_claim(db_session, organization.id, domain="acme.example")
    return organization


def _authorize(client: TestClient, slug: str) -> Any:
    response = client.get(f"{API_ROOT}/auth/sso/{slug}/authorize")
    assert response.status_code == 200, response.text
    return response.json()


def stub_exchange(
    monkeypatch: pytest.MonkeyPatch,
    *,
    subject: str = "ada-subject",
    email: str | None = "ada@acme.example",
    full_name: str | None = "Ada Lovelace",
    # Defaults to what a connection reports in practice: trust_email_verified is off for every
    # connection, so apron-auth sends no claim at all and the assertion is unasserted.
    email_verified: EmailAssertion = EmailAssertion.UNASSERTED,
) -> list[str]:
    """Replace the provider round trip, and record every code it was handed."""
    spent: list[str] = []

    async def _exchange(_client: Any, *, code: str, **_kwargs: Any) -> ProviderIdentity:
        spent.append(code)
        return ProviderIdentity(
            provider=f"oidc:{ISSUER}",
            subject=subject,
            email=email,
            full_name=full_name,
            email_verified=email_verified,
        )

    monkeypatch.setattr("gateway.api.routes.auth_sso._exchange_sso_code", _exchange)
    return spent


# ---------- lookup ----------


def test_lookup_routes_a_claimed_domain(client: TestClient, db_session: Session) -> None:
    organization = _acme(db_session)

    response = client.post(f"{API_ROOT}/auth/sso/lookup", json={"email": "ada@acme.example"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["found"] is True
    assert body["organization_name"] == organization.name
    assert body["slug"] == organization.slug


def test_lookup_answers_the_same_for_an_unclaimed_domain_and_an_unknown_one(
    client: TestClient, db_session: Session
) -> None:
    _acme(db_session)

    unclaimed = client.post(f"{API_ROOT}/auth/sso/lookup", json={"email": "ada@unclaimed.example"})
    unknown = client.post(f"{API_ROOT}/auth/sso/lookup", json={"email": "ada@never-heard-of-it.example"})

    assert unclaimed.status_code == unknown.status_code == 200
    assert unclaimed.json() == unknown.json() == {"found": False, "organization_name": None, "slug": None}


def test_a_mixed_case_address_with_a_trailing_dot_routes(client: TestClient, db_session: Session) -> None:
    _acme(db_session)

    response = client.post(f"{API_ROOT}/auth/sso/lookup", json={"email": "Ada@Acme.Example."})

    assert response.status_code == 200, response.text
    assert response.json()["found"] is True


def test_lookup_is_throttled(client: TestClient) -> None:
    client.app.state.public_auth_rate_limiter = RateLimiter(1)  # type: ignore[attr-defined]

    first = client.post(f"{API_ROOT}/auth/sso/lookup", json={"email": "ada@example.com"})
    second = client.post(f"{API_ROOT}/auth/sso/lookup", json={"email": "ada@example.com"})

    assert first.status_code == 200
    assert second.status_code == 429


# ---------- authorize ----------


def test_authorize_for_an_unknown_slug_and_a_slug_with_no_connection_answer_alike(
    client: TestClient, db_session: Session
) -> None:
    """An unknown slug and a slug naming an organization with no connection answer alike."""
    bare = _add_organization(db_session, slug="bare")

    unknown = client.get(f"{API_ROOT}/auth/sso/never-heard-of-it/authorize")
    no_connection = client.get(f"{API_ROOT}/auth/sso/{bare.slug}/authorize")

    assert unknown.status_code == no_connection.status_code == 404
    assert unknown.json() == no_connection.json()


def test_authorize_hands_back_a_consent_url_and_sets_the_flow_cookie(client: TestClient, db_session: Session) -> None:
    organization = _acme(db_session)

    bound = _authorize(client, organization.slug)

    assert bound["authorization_url"].startswith(f"{ISSUER}/authorize")
    assert bound["state"]
    jar = {cookie.name for cookie in client.cookies.jar}
    assert "otari_sso_flow" in jar


def test_a_disabled_connection_answers_the_same_as_no_connection(client: TestClient, db_session: Session) -> None:
    organization = _add_organization(db_session, slug="disabled-org")
    _add_connection(db_session, organization.id, enabled=False)
    _add_claim(db_session, organization.id, domain="disabled-org.example")
    bare = _add_organization(db_session, slug="bare-two")

    response = client.get(f"{API_ROOT}/auth/sso/{organization.slug}/authorize")
    unknown = client.get(f"{API_ROOT}/auth/sso/{bare.slug}/authorize")

    assert response.status_code == unknown.status_code == 404
    assert response.json() == unknown.json()


# ---------- callback ----------


def test_the_callback_refuses_without_the_flow_cookie(client: TestClient, db_session: Session) -> None:
    organization = _acme(db_session)
    bound = _authorize(client, organization.slug)
    # The browser lost the cookie: the request names a real, still-pending state with no token to
    # bind it to, the way a tab that lost its HttpOnly cookie between redirect and return would.
    client.cookies.delete("otari_sso_flow")

    response = client.post(
        f"{API_ROOT}/auth/sso/callback", json={"code": "a-code", "state": bound["state"], "iss": ISSUER}
    )

    assert response.status_code == 400, response.text
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_google_state_returned_to_the_sso_callback_is_refused(
    client: TestClient, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(test_config, "oauth_google_client_id", "google-id")
    monkeypatch.setattr(test_config, "oauth_google_client_secret", "google-secret")
    google = client.get(f"{API_ROOT}/auth/oauth/google/authorize")
    assert google.status_code == 200, google.text

    response = client.post(f"{API_ROOT}/auth/sso/callback", json={"code": "a-code", "state": google.json()["state"]})

    assert response.status_code == 400, response.text
    assert SESSION_COOKIE_NAME not in response.cookies


def test_an_sso_state_returned_to_the_oauth_callback_is_refused(
    client: TestClient, db_session: Session, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reverse of the check above: a state minted here is not spendable at the built-in callback."""
    monkeypatch.setattr(test_config, "oauth_google_client_id", "google-id")
    monkeypatch.setattr(test_config, "oauth_google_client_secret", "google-secret")
    organization = _acme(db_session)
    bound = _authorize(client, organization.slug)

    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "a-code", "state": bound["state"]})

    assert response.status_code == 400, response.text
    assert SESSION_COOKIE_NAME not in response.cookies


def test_the_callback_is_refused_in_maintenance_mode(
    client: TestClient, db_session: Session, master_key_header: dict[str, str]
) -> None:
    organization = _acme(db_session)
    bound = _authorize(client, organization.slug)
    frozen = client.patch(f"{API_ROOT}/settings/maintenance-mode", json={"enabled": True}, headers=master_key_header)
    assert frozen.status_code == 200, frozen.text

    response = client.post(
        f"{API_ROOT}/auth/sso/callback", json={"code": "a-code", "state": bound["state"], "iss": ISSUER}
    )

    assert response.status_code == 503, response.text
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_callback_whose_iss_names_another_issuer_is_refused(client: TestClient, db_session: Session) -> None:
    organization = _acme(db_session)
    bound = _authorize(client, organization.slug)

    response = client.post(
        f"{API_ROOT}/auth/sso/callback",
        json={"code": "a-code", "state": bound["state"], "iss": "https://attacker.example"},
    )

    assert response.status_code == 400, response.text
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_successful_callback_sets_the_session_cookie(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = _acme(db_session)
    bound = _authorize(client, organization.slug)
    spent = stub_exchange(monkeypatch)

    response = client.post(f"{API_ROOT}/auth/sso/callback", json={"code": "the-code", "state": bound["state"]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["active_organization_id"] == str(organization.id)
    assert SESSION_COOKIE_NAME in response.cookies
    assert spent == ["the-code"]


def test_a_second_sign_in_through_the_same_connection_reaches_the_same_account(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = _acme(db_session)
    stub_exchange(monkeypatch)

    first_bound = _authorize(client, organization.slug)
    first = client.post(f"{API_ROOT}/auth/sso/callback", json={"code": "code-one", "state": first_bound["state"]})
    assert first.status_code == 200, first.text

    second_bound = _authorize(client, organization.slug)
    second = client.post(f"{API_ROOT}/auth/sso/callback", json={"code": "code-two", "state": second_bound["state"]})

    assert second.status_code == 200, second.text
    assert second.json()["user_id"] == first.json()["user_id"]


def test_a_valid_flow_cookie_reaches_the_provider_rather_than_a_state_refusal(
    client: TestClient, db_session: Session
) -> None:
    """A state and ``iss`` that both match proceed to the provider, which the fake issuer answers with 502.

    A 400 here would mean the browser's own matching token never reached the exchange, so the
    cookie's real check never ran.
    """
    organization = _acme(db_session)
    bound = _authorize(client, organization.slug)

    response = client.post(
        f"{API_ROOT}/auth/sso/callback", json={"code": "a-code", "state": bound["state"], "iss": ISSUER}
    )

    assert response.status_code == 502, response.text
