"""An MCP client discovers Otari's sign-in, registers itself, and signs in as its person, read-only."""

from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from apron_auth.pkce import generate_code_challenge as pkce_challenge
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

from .conftest import build_test_client
from .test_organization_member_keys import _World
from .test_organization_member_keys import world as world

_REDIRECT = "http://localhost:33418/callback"
_VERIFIER = "w" * 43
_MCP_HEADERS = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}


def _register(client: TestClient, **metadata: Any) -> Any:
    body = {"client_name": "Claude Code", "redirect_uris": [_REDIRECT], **metadata}
    return client.post(f"{API_ROOT}/oauth/apps/register", json=body)


def _approve(client: TestClient, world: _World, client_id: str, **changes: Any) -> Any:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    body = {
        "client_id": client_id,
        "redirect_uri": _REDIRECT,
        "state": "s" * 43,
        "code_challenge": pkce_challenge(_VERIFIER),
        "code_challenge_method": "S256",
        "scopes": [],
        "expiration": "30-days",
        "organization_id": str(world.alpha),
        **changes,
    }
    response = client.post(f"{API_ROOT}/oauth/apps/authorize", json=body, headers={"Sec-Fetch-Site": "same-origin"})
    client.cookies.clear()
    return response


def _renew(client: TestClient, client_id: str, refresh_token: str) -> Any:
    return client.post(
        f"{API_ROOT}/oauth/apps/token",
        data={"grant_type": "refresh_token", "client_id": client_id, "refresh_token": refresh_token},
    )


def _get_context(client: TestClient, token: str) -> Any:
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_context", "arguments": {}}},
        headers={**_MCP_HEADERS, "Authorization": f"Bearer {token}"},
    )


def _sign_in(client: TestClient, world: _World) -> tuple[str, dict[str, Any]]:
    client_id = _register(client).json()["client_id"]
    redeemed = _redeem(client, client_id, _code(_approve(client, world, client_id)))
    assert redeemed.status_code == 200, redeemed.text
    return client_id, redeemed.json()


def _redeem(client: TestClient, client_id: str, code: str) -> Any:
    return client.post(
        f"{API_ROOT}/oauth/apps/token",
        data={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "redirect_uri": _REDIRECT,
            "code": code,
            "code_verifier": _VERIFIER,
            "resource": "http://testserver/mcp",
        },
    )


def _code(response: Any) -> str:
    assert response.status_code == 200, response.text
    return parse_qs(urlsplit(str(response.json()["redirect_uri"])).query)["code"][0]


def test_an_unauthenticated_call_names_where_to_sign_in(client: TestClient) -> None:
    refused = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=_MCP_HEADERS)
    assert refused.status_code == 401
    metadata_url = "http://testserver/.well-known/oauth-protected-resource/mcp"
    assert refused.headers["www-authenticate"] == f'Bearer resource_metadata="{metadata_url}"'

    resource = client.get(metadata_url).json()
    assert resource["resource"] == "http://testserver/mcp"
    assert resource["authorization_servers"] == ["http://testserver"]

    server = client.get("/.well-known/oauth-authorization-server").json()
    assert server["issuer"] == "http://testserver"
    assert server["registration_endpoint"] == f"http://testserver{API_ROOT}/oauth/apps/register"
    assert server["token_endpoint"] == f"http://testserver{API_ROOT}/oauth/apps/token"
    assert server["code_challenge_methods_supported"] == ["S256"]


def test_a_rejected_token_is_named_invalid(client: TestClient) -> None:
    refused = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={**_MCP_HEADERS, "Authorization": "Bearer otari-pat-revoked"},
    )
    assert refused.status_code == 401
    assert 'error="invalid_token"' in refused.headers["www-authenticate"]


def test_a_client_registers_itself_as_an_unverified_public_client(client: TestClient) -> None:
    registered = _register(client, token_endpoint_auth_method="client_secret_post", scope="anything")
    assert registered.status_code == 201, registered.text
    body = registered.json()
    assert body["client_id"].startswith("mcp-")
    assert body["token_endpoint_auth_method"] == "none"
    assert body["redirect_uris"] == [_REDIRECT]

    # Any free loopback port, as RFC 8252 lets a native client choose.
    details = client.get(
        f"{API_ROOT}/oauth/apps/clients",
        params={"client_id": body["client_id"], "redirect_uri": "http://localhost:40000/callback"},
    )
    assert details.status_code == 200, details.text
    assert details.json() | {"redirect_uri": None} == {
        "name": "Claude Code",
        "redirect_uri": None,
        "issuer": "http://testserver",
        "scopes": [],
        "verified": False,
        "spend_limits": False,
        "credential": "access_token",
    }


def test_registration_refuses_a_callback_off_https_and_loopback(client: TestClient) -> None:
    refused = _register(client, redirect_uris=["http://evil.example/callback"])
    assert refused.status_code == 400
    assert refused.json()["error"] == "invalid_redirect_uri"


def test_a_registered_client_signs_in_for_an_hour_and_renews(client: TestClient, world: _World) -> None:
    client_id = _register(client).json()["client_id"]
    code = _code(_approve(client, world, client_id))
    redeemed = _redeem(client, client_id, code)
    assert redeemed.status_code == 200, redeemed.text
    token = redeemed.json()
    assert token["access_token"].startswith("otari-mcp-at-")
    assert token["refresh_token"].startswith("otari-mcp-rt-")
    assert token["expires_in"] == 3600
    assert "scope" not in token
    assert token["manage_url"].endswith("/account")

    context = _get_context(client, token["access_token"])
    assert context.status_code == 200, context.text
    assert str(world.alpha) in context.json()["result"]["content"][0]["text"]
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    listed = client.get(f"{API_ROOT}/auth/access-tokens")
    client.cookies.clear()
    assert [row["name"] for row in listed.json()["data"]] == ["Claude Code (MCP)"]
    assert listed.json()["data"][0]["organization_id"] == str(world.alpha)

    renewed = _renew(client, client_id, token["refresh_token"])
    assert renewed.status_code == 200, renewed.text
    assert renewed.json()["refresh_token"] != token["refresh_token"]
    assert _get_context(client, token["access_token"]).status_code == 401
    assert _get_context(client, renewed.json()["access_token"]).status_code == 200

    replayed = _redeem(client, client_id, code)
    assert replayed.status_code == 400
    assert replayed.json()["error"] == "invalid_grant"


def test_an_mcp_access_token_reads_only_through_mcp(client: TestClient, world: _World) -> None:
    _client_id, token = _sign_in(client, world)
    direct = client.get(f"{API_ROOT}/organizations/me", headers={"Authorization": f"Bearer {token['access_token']}"})
    assert direct.status_code == 401
    assert direct.json()["detail"] == "This access token works only through the MCP server"


def test_a_refresh_token_used_twice_ends_the_sign_in(client: TestClient, world: _World) -> None:
    client_id, token = _sign_in(client, world)
    renewed = _renew(client, client_id, token["refresh_token"]).json()
    stolen = _renew(client, client_id, token["refresh_token"])
    assert stolen.status_code == 400
    assert stolen.json()["error"] == "invalid_grant"
    # The rotation that already happened is ended too, since either side may hold the stolen copy.
    assert _renew(client, client_id, renewed["refresh_token"]).status_code == 400
    assert _get_context(client, renewed["access_token"]).status_code == 401


def test_a_refresh_token_renews_only_its_own_client(client: TestClient, world: _World) -> None:
    _client_id, token = _sign_in(client, world)
    other = _register(client).json()["client_id"]
    assert _renew(client, other, token["refresh_token"]).status_code == 400


@pytest.mark.parametrize("kind", ["access_token", "refresh_token"])
def test_the_revocation_endpoint_ends_a_sign_in(client: TestClient, world: _World, kind: str) -> None:
    client_id, token = _sign_in(client, world)
    revoked = client.post(f"{API_ROOT}/oauth/apps/revoke", data={"token": token[kind], "client_id": client_id})
    assert revoked.status_code == 200, revoked.text
    assert _get_context(client, token["access_token"]).status_code == 401
    assert _renew(client, client_id, token["refresh_token"]).status_code == 400


@pytest.mark.parametrize("expiration", ["never", "1-year"])
def test_a_sign_in_lasts_at_most_ninety_days(client: TestClient, world: _World, expiration: str) -> None:
    client_id = _register(client).json()["client_id"]
    refused = _approve(client, world, client_id, expiration=expiration)
    assert refused.status_code == 400, refused.text


def test_consent_names_the_organization_the_person_is_in(client: TestClient, world: _World) -> None:
    client_id = _register(client).json()["client_id"]
    assert _approve(client, world, client_id, organization_id=str(world.beta)).status_code == 400
    assert _approve(client, world, client_id, organization_id=None).status_code == 400


def test_a_registered_client_is_never_given_an_api_key(client: TestClient, world: _World) -> None:
    client_id = _register(client).json()["client_id"]
    approved = _approve(
        client, world, client_id, workspace_id=str(world.workspaces["alpha_one"]), scopes=["inference:run"]
    )
    token = _redeem(client, client_id, _code(approved)).json()["access_token"]
    assert token.startswith("otari-mcp-at-")


def test_an_unknown_client_cannot_redeem(client: TestClient) -> None:
    refused = _redeem(client, "mcp-unregistered", "c" * 43)
    assert refused.status_code == 400
    assert refused.json()["error"] == "invalid_client"


def test_turning_the_server_off_takes_the_surface_and_its_registration(
    test_config: GatewayConfig,
) -> None:
    """``mcp_server_enabled: false`` is the switch the docs offer, so the surface and sign-in go with it."""
    for off in build_test_client(test_config.model_copy(update={"mcp_server_enabled": False})):
        call = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        assert off.post("/mcp", json=call, headers=_MCP_HEADERS).status_code == 404
        for discovery in ("/.well-known/oauth-authorization-server", "/.well-known/oauth-protected-resource/mcp"):
            assert off.get(discovery).status_code == 404, discovery
        assert _register(off).status_code == 404
