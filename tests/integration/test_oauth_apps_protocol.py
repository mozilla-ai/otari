"""Public OAuth protocol validation and standard form interoperability."""

from urllib.parse import parse_qs, urlencode

import pytest
from apron_auth.pkce import generate_code_challenge as pkce_challenge
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.settings.identity import OAuthAppConfig

from .test_oauth_apps import REDIRECT, VERIFIER, approve, code_from, redeem
from .test_oauth_apps import test_config as test_config
from .test_organization_member_keys import _World
from .test_organization_member_keys import world as world


def _authorization(**changes: str) -> dict[str, str]:
    return {
        "client_id": "thunderbolt",
        "redirect_uri": REDIRECT,
        "state": "s" * 43,
        "response_type": "code",
        "code_challenge": pkce_challenge(VERIFIER),
        "code_challenge_method": "S256",
        **changes,
    }


@pytest.mark.parametrize("changes", [{"client_id": "unknown"}, {"redirect_uri": "https://attacker.invalid/cb"}])
def test_unknown_client_or_callback_gets_uncached_oauth_error(client: TestClient, changes: dict[str, str]) -> None:
    response = client.get(f"{API_ROOT}/oauth/apps/authorize", params=_authorization(**changes), follow_redirects=False)
    assert response.status_code == 400
    assert response.json() == {"error": "invalid_request", "error_description": "Unknown app or redirect URI"}
    assert "location" not in response.headers
    assert "no-store" in response.headers["cache-control"]
    assert response.headers["pragma"] == "no-cache"


@pytest.mark.parametrize(
    "media_type", ["Application/X-WWW-Form-Urlencoded", "APPLICATION/X-WWW-FORM-URLENCODED; charset=UTF-8"]
)
def test_token_exchange_accepts_caseless_form_media_types(client: TestClient, world: _World, media_type: str) -> None:
    code = code_from(approve(client, world))
    client.cookies.clear()
    response = client.post(
        f"{API_ROOT}/oauth/apps/token",
        content=urlencode(
            {
                "grant_type": "authorization_code",
                "client_id": "thunderbolt",
                "redirect_uri": REDIRECT,
                "code": code,
                "code_verifier": VERIFIER,
            }
        ),
        headers={"Content-Type": media_type},
    )
    assert response.status_code == 200, response.text
    assert response.json()["token_type"] == "Bearer"


def test_client_metadata_accepts_registered_ids_with_slashes(client: TestClient, test_config: GatewayConfig) -> None:
    client_id = "team/app + extra"
    test_config.oauth_apps[client_id] = OAuthAppConfig(name="Team app", redirect_uris=[REDIRECT])
    response = client.get(f"{API_ROOT}/oauth/apps/clients", params={"client_id": client_id, "redirect_uri": REDIRECT})
    assert response.status_code == 200, response.text
    assert response.json() == {
        "name": "Team app",
        "redirect_uri": REDIRECT,
        "issuer": "http://testserver",
        "scopes": ["inference:run"],
    }


def test_base64_state_survives_both_authorization_steps(client: TestClient, world: _World) -> None:
    state = "aB+/aB+/aB+/aB+/=="
    response = client.get(
        f"{API_ROOT}/oauth/apps/authorize", params=_authorization(state=state), follow_redirects=False
    )
    assert response.status_code == 302, response.text
    assert parse_qs(response.headers["location"].split("?", 1)[1])["state"] == [state]
    approved = approve(client, world, state=state)
    assert approved.status_code == 200, approved.text
    assert parse_qs(approved.json()["redirect_uri"].split("?", 1)[1])["state"] == [state]


def test_authorization_ignores_unrecognized_parameters(client: TestClient) -> None:
    """RFC 6749 section 3.1: a client library's ``scope`` or ``nonce`` must not refuse the request."""
    response = client.get(
        f"{API_ROOT}/oauth/apps/authorize",
        params=_authorization(scope="openid", nonce="n" * 16),
        follow_redirects=False,
    )
    assert response.status_code == 302, response.text
    forwarded = parse_qs(response.headers["location"].split("?", 1)[1])
    assert "scope" not in forwarded
    assert "nonce" not in forwarded


def test_token_exchange_ignores_unrecognized_parameters(client: TestClient, world: _World) -> None:
    response = redeem(client, code_from(approve(client, world)), client_secret="", scope="openid")
    assert response.status_code == 200, response.text


def test_consent_still_refuses_unrecognized_fields(client: TestClient, world: _World) -> None:
    response = approve(client, world, unexpected="value")
    assert response.status_code == 422, response.text
