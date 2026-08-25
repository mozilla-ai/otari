"""OAuth sign-in end to end, with only the provider's own endpoints stubbed.

Everything on this side of the exchange is real: the routes, the container, the
``IdentityProviderPort`` adapter the base build binds, the roster it resolves
against, and the session cookie a sign-in mints. What is replaced is
apron-auth's outbound half, because a test cannot complete a consent screen.

That replacement is the reason
``tests/integration/test_oauth_live_provider.py`` exists: it is the check that
the request shape apron-auth actually sends is one Google and GitHub accept, and
nothing here can stand in for it.
"""

from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlmodel import col, select

from gateway.core.config import GatewayConfig
from gateway.models.tenancy import User
from gateway.services import oauth_service
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME
from gateway.services.oauth_service import OAuthIdentity

ORIGIN = "http://testserver"
PASSWORD = "a-real-password"  # pragma: allowlist secret


@pytest.fixture
def oauth_configured(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Register both providers on the deployment TestClient serves."""
    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)
    monkeypatch.setattr(test_config, "oauth_google_client_id", "google-id")
    monkeypatch.setattr(test_config, "oauth_google_client_secret", "google-secret")
    monkeypatch.setattr(test_config, "oauth_github_client_id", "github-id")
    monkeypatch.setattr(test_config, "oauth_github_client_secret", "github-secret")


def stub_exchange(
    monkeypatch: pytest.MonkeyPatch,
    *,
    email: str | None = "ada@example.com",
    full_name: str | None = "Ada Lovelace",
    email_verified: bool = True,
) -> list[str]:
    """Replace the provider round trip, and record every code it was handed.

    Patched on the route module's own reference, so the substitution is visible
    from the handler rather than depending on how it imported the service.
    """
    spent: list[str] = []

    async def _exchange(_config: GatewayConfig, provider: str, *, code: str) -> OAuthIdentity:
        spent.append(code)
        return OAuthIdentity(
            provider=provider,
            email=email,
            full_name=full_name,
            email_verified=email_verified,
        )

    monkeypatch.setattr("gateway.api.routes.auth_oauth.exchange_code", _exchange)
    return spent


def _identity(db_session: Session, email: str) -> User:
    """The tenancy identity holding ``email``, read outside the test client."""
    identity = db_session.execute(select(User).where(col(User.email) == email)).scalar_one()
    return identity


def add_member(client: TestClient, master_key_header: dict[str, str], *, email: str) -> str:
    """Put an address on the roster, the way an operator does, and return its id."""
    response = client.post(
        "/v1/organizations/me/members",
        json={"email": email, "role": "member"},
        headers=master_key_header,
    )
    assert response.status_code == 201, response.text
    member: dict[str, Any] = response.json()
    return str(member["user_id"])


# ---------- what the deployment publishes ----------


def test_the_bootstrap_offers_no_provider_until_one_is_configured(client: TestClient) -> None:
    # The default. This is what makes the sign-in screen carry no OAuth
    # affordance out of the box rather than a pair of dead buttons.
    bootstrap = client.get("/v1/bootstrap")

    assert bootstrap.status_code == 200, bootstrap.text
    assert bootstrap.json()["oauth_providers"] == []


def test_the_bootstrap_names_the_providers_an_operator_configured(
    client: TestClient, oauth_configured: None
) -> None:
    assert client.get("/v1/bootstrap").json()["oauth_providers"] == ["github", "google"]


def test_a_provider_missing_its_secret_is_not_published(
    client: TestClient, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)
    monkeypatch.setattr(test_config, "oauth_google_client_id", "google-id")

    assert client.get("/v1/bootstrap").json()["oauth_providers"] == []


# ---------- starting the flow ----------


@pytest.mark.parametrize("provider", ["google", "github"])
def test_authorize_hands_back_a_consent_url_and_a_fresh_state(
    client: TestClient, oauth_configured: None, provider: str
) -> None:
    first = client.get(f"/v1/auth/oauth/{provider}/authorize")
    second = client.get(f"/v1/auth/oauth/{provider}/authorize")

    assert first.status_code == 200, first.text
    query = parse_qs(urlsplit(first.json()["authorization_url"]).query)
    assert query["redirect_uri"] == [f"{ORIGIN}/auth/{provider}/callback"]
    assert query["state"] == [first.json()["state"]]
    # A fresh value per request: only the one the browser kept is the one it
    # will compare against.
    assert first.json()["state"] != second.json()["state"]


def test_authorize_needs_no_credential(client: TestClient, oauth_configured: None) -> None:
    # It is how somebody who holds nothing starts signing in, so requiring a
    # credential would be circular.
    assert client.get("/v1/auth/oauth/google/authorize").status_code == 200


def test_authorize_refuses_an_unconfigured_provider_and_names_the_settings(
    client: TestClient,
) -> None:
    response = client.get("/v1/auth/oauth/google/authorize")

    assert response.status_code == 503
    assert "oauth_google_client_id" in response.json()["detail"]


def test_a_provider_this_deployment_could_never_configure_is_not_a_route(
    client: TestClient, oauth_configured: None
) -> None:
    # The path parameter is bounded by the config vocabulary, so an unknown
    # segment is refused by the framework rather than by a handler.
    assert client.get("/v1/auth/oauth/not-a-provider/authorize").status_code == 422


# ---------- finishing the flow ----------


def test_a_rostered_member_signs_in_and_gets_the_same_session_a_password_would(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = add_member(client, master_key_header, email="ada@example.com")
    spent = stub_exchange(monkeypatch)

    response = client.post("/v1/auth/oauth/google/callback", json={"code": "the-code"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["user_id"] == user_id
    assert body["active_organization_id"]
    # The token travels only in the cookie, exactly as the password and passkey
    # sign-ins do.
    assert SESSION_COOKIE_NAME in response.cookies
    assert "token" not in body
    assert spent == ["the-code"]

    # And the cookie authenticates the management API on its own, with no header
    # credential: the provider minted the same session a password would have.
    assert client.cookies.get(SESSION_COOKIE_NAME)
    membership = client.get("/v1/organizations/me")
    assert membership.status_code == 200, membership.text
    assert membership.json()["organization"]["id"] == body["active_organization_id"]


def test_the_provider_is_recorded_on_the_identity_it_signed_in(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    # Read from the column rather than an endpoint: `user.oauth_provider` is
    # carried for schema parity with the platform and no route publishes it, so
    # the link is only observable here.
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    signed_in = client.post("/v1/auth/oauth/github/callback", json={"code": "c"})
    assert signed_in.status_code == 200, signed_in.text

    identity = _identity(db_session, "ada@example.com")
    assert identity.oauth_provider == "github"
    # And the provider's assertion lifted the local verification gate.
    assert identity.email_verified_at is not None


def test_a_verified_provider_address_lifts_the_local_verification_gate(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A member an operator added has never verified their address here, and the
    # password sign-in hard-blocks that. The provider's assertion is a stronger
    # proof of the same fact, so this is how a deployment that cannot send mail
    # still lets a member in.
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    assert client.post("/v1/auth/oauth/google/callback", json={"code": "c"}).status_code == 200


def test_an_address_nobody_put_on_the_roster_is_refused_rather_than_provisioned(
    client: TestClient, oauth_configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The base build's roster policy, and the whole reason the decision sits
    # behind IdentityProviderPort: social sign-in widens how a member
    # authenticates, never who may. Provisioning here would let any holder of a
    # Google account into a self-hosted gateway.
    stub_exchange(monkeypatch, email="stranger@example.com")

    response = client.post("/v1/auth/oauth/google/callback", json={"code": "c"})

    assert response.status_code == 401
    assert "not registered on this gateway" in response.json()["detail"]
    assert SESSION_COOKIE_NAME not in response.cookies


def test_an_unverified_provider_address_is_refused_even_when_it_is_on_the_roster(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch, email_verified=False)

    response = client.post("/v1/auth/oauth/google/callback", json={"code": "c"})

    assert response.status_code == 401
    assert "did not confirm that address is yours" in response.json()["detail"]
    assert SESSION_COOKIE_NAME not in response.cookies


def test_a_provider_that_returns_no_address_at_all_is_refused(
    client: TestClient, oauth_configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_exchange(monkeypatch, email=None)

    assert client.post("/v1/auth/oauth/google/callback", json={"code": "c"}).status_code == 401


def test_a_deactivated_identity_cannot_sign_in_with_a_provider_either(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    # Deactivating somebody has to close every road in, or OAuth becomes the
    # door left open behind them. Flipped in the database because this edition
    # exposes no route that deactivates a tenancy identity; `/v1/users` is the
    # request-plane spend identity, which is a different table.
    add_member(client, master_key_header, email="ada@example.com")
    identity = _identity(db_session, "ada@example.com")
    identity.is_active = False
    db_session.add(identity)
    db_session.commit()
    stub_exchange(monkeypatch)

    response = client.post("/v1/auth/oauth/google/callback", json={"code": "c"})

    assert response.status_code == 401
    # Collapsed into the unknown-identity refusal rather than saying "switched
    # off", so somebody an operator shut out cannot keep confirming their
    # account is still on file.
    assert "not registered on this gateway" in response.json()["detail"]


def test_a_differently_cased_provider_address_still_finds_its_roster_row(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch, email="Ada@Example.COM")

    assert client.post("/v1/auth/oauth/google/callback", json={"code": "c"}).status_code == 200


def test_the_callback_refuses_an_unconfigured_provider_before_spending_anything(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gate is a dependency ahead of the handler, so it holds even with the
    # exchange stubbed out. It was a check inside the exchange first, and this
    # test is what showed that a caller could reach the identity resolution of a
    # provider this deployment never configured.
    spent = stub_exchange(monkeypatch)

    response = client.post("/v1/auth/oauth/google/callback", json={"code": "c"})

    assert response.status_code == 503
    assert "oauth_google_client_id" in response.json()["detail"]
    assert spent == []


def test_maintenance_mode_freezes_an_oauth_sign_in_before_the_exchange(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The freeze is on starting a session, not on a credential, so an OAuth
    # sign-in has to answer to it or the switch is bypassable by anybody holding
    # a Google account. Refused before the exchange, so a frozen deployment
    # spends nobody's single-use authorization code.
    add_member(client, master_key_header, email="ada@example.com")
    frozen = client.patch(
        "/v1/settings/maintenance-mode", json={"enabled": True}, headers=master_key_header
    )
    assert frozen.status_code == 200, frozen.text
    spent = stub_exchange(monkeypatch)

    response = client.post("/v1/auth/oauth/google/callback", json={"code": "c"})

    assert response.status_code == 503
    assert spent == []
    assert SESSION_COOKIE_NAME not in response.cookies


def test_the_callback_body_carries_the_code_and_nothing_else_the_server_trusts(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No redirect_uri and no state: the URI is derived from public_base_url so a
    # browser cannot choose what this server sends to a provider, and the state
    # is checked in the browser against the value it stored.
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    response = client.post(
        "/v1/auth/oauth/google/callback",
        json={
            "code": "c",
            "redirect_uri": "https://attacker.example.com/callback",
            "state": "anything",
        },
    )

    assert response.status_code == 200, response.text
    # The extra fields were ignored rather than honored: the URI this deployment
    # would send is still its own, whatever the body asked for.
    assert oauth_service.redirect_uri(test_config, "google") == f"{ORIGIN}/auth/google/callback"


def test_an_oversized_code_is_refused_before_any_outbound_call(
    client: TestClient, oauth_configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    spent = stub_exchange(monkeypatch)

    response = client.post("/v1/auth/oauth/google/callback", json={"code": "x" * 4096})

    assert response.status_code == 422
    assert spent == []


def test_a_database_failure_while_staging_the_session_rolls_back_and_says_nothing_more(
    client: TestClient,
    master_key_header: dict[str, str],
    oauth_configured: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staging the session row is inside the route's error handling, not beside it.

    ``create_dashboard_session`` prunes expired rows with a DELETE before it
    stages anything, so it can fail on its own rather than only at commit time.
    Outside the handled block that failure skipped the rollback and surfaced as
    a bare 500 from the generic handler.
    """
    add_member(client, master_key_header, email="ada@example.com")
    stub_exchange(monkeypatch)

    async def _explode(*_args: Any, **_kwargs: Any) -> tuple[str, Any]:
        raise SQLAlchemyError("pruning failed")

    monkeypatch.setattr("gateway.api.routes.auth_oauth.create_dashboard_session", _explode)

    response = client.post("/v1/auth/oauth/google/callback", json={"code": "c"})

    assert response.status_code == 500
    # The generic wording, not the exception: the error-detail boundary holds on
    # this path the way it does on the commit path beside it.
    assert response.json()["detail"] == "Database error"
    assert "pruning failed" not in response.text
    assert SESSION_COOKIE_NAME not in response.cookies


# ---------- the redirect a provider actually lands on ----------


@pytest.mark.parametrize("provider", ["google", "github"])
def test_the_provider_redirect_path_bounces_into_the_dashboard_hash_route(
    client: TestClient, provider: str
) -> None:
    # A redirect URI may not carry a fragment, so the provider is pointed at an
    # ordinary path and this is what turns it into the hash route the dashboard
    # renders ahead of its auth gate.
    response = client.get(
        f"/auth/{provider}/callback?code=the-code&state=the-state", follow_redirects=False
    )

    assert response.status_code == 303
    assert response.headers["location"] == (
        f"/#/auth/{provider}/callback?code=the-code&state=the-state"
    )


def test_the_provider_redirect_path_carries_an_error_query_through_too(
    client: TestClient,
) -> None:
    response = client.get(
        "/auth/google/callback?error=access_denied&state=s", follow_redirects=False
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/#/auth/google/callback?error=access_denied&state=s"


def test_the_provider_redirect_path_works_with_no_query_at_all(client: TestClient) -> None:
    response = client.get("/auth/google/callback", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/#/auth/google/callback"


def test_the_provider_redirect_path_needs_no_credential(client: TestClient) -> None:
    # A person is here because a provider sent them, holding nothing.
    assert client.get("/auth/google/callback", follow_redirects=False).status_code == 303
