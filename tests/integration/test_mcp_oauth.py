"""An MCP client discovers Otari's sign-in, registers itself, and is given its person's access token."""

from typing import Any
from urllib.parse import parse_qs, urlsplit

from apron_auth.pkce import generate_code_challenge as pkce_challenge
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

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
        **changes,
    }
    response = client.post(f"{API_ROOT}/oauth/apps/authorize", json=body, headers={"Sec-Fetch-Site": "same-origin"})
    client.cookies.clear()
    return response


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


def test_a_registered_client_is_given_the_persons_access_token(client: TestClient, world: _World) -> None:
    client_id = _register(client).json()["client_id"]
    code = _code(_approve(client, world, client_id))
    redeemed = _redeem(client, client_id, code)
    assert redeemed.status_code == 200, redeemed.text
    token = redeemed.json()
    assert token["access_token"].startswith("otari-pat-")
    assert token["expires_in"] == 30 * 24 * 3600
    assert "scope" not in token
    assert token["manage_url"].endswith("/account")

    listed = client.get(f"{API_ROOT}/auth/access-tokens", headers={"Authorization": f"Bearer {token['access_token']}"})
    assert [row["name"] for row in listed.json()["data"]] == ["Claude Code (MCP)"]
    context = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_context", "arguments": {}}},
        headers={**_MCP_HEADERS, "Authorization": f"Bearer {token['access_token']}"},
    )
    assert context.status_code == 200, context.text
    assert str(world.alpha) in context.json()["result"]["content"][0]["text"]

    replayed = _redeem(client, client_id, code)
    assert replayed.status_code == 400
    assert replayed.json()["error"] == "invalid_grant"


def test_a_registered_client_is_never_given_an_api_key(client: TestClient, world: _World) -> None:
    client_id = _register(client).json()["client_id"]
    approved = _approve(
        client, world, client_id, workspace_id=str(world.workspaces["alpha_one"]), scopes=["inference:run"]
    )
    token = _redeem(client, client_id, _code(approved)).json()["access_token"]
    assert token.startswith("otari-pat-")


def test_an_unknown_client_cannot_redeem(client: TestClient) -> None:
    refused = _redeem(client, "mcp-unregistered", "c" * 43)
    assert refused.status_code == 400
    assert refused.json()["error"] == "invalid_client"
