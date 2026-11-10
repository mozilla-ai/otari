"""Per-user MCP connections over HTTP: the sign-in routes, and what an API key is told before and after.

`test_mcp_user_connections.py` covers the rules at the service layer. This file covers what only a
route shows: the callback's redirects, the status and code a key gets, and that no body carries a token.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from mcp.types import ListToolsResult
from mcp.types import Tool as MCPTool
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.error_codes import ERROR_CODE_HEADER, MCP_SIGN_IN_REQUIRED
from gateway.models.api_keys import APIKey
from gateway.models.mcp import ResolvedMcpServer
from gateway.models.tools import McpOAuthFlow, McpUserConnection
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import ServerCheck
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tools import _mcp_oauth
from gateway.services.tools._mcp_oauth import AUTH_METHOD_NONE, McpOAuthClient, McpOAuthMetadata, McpOAuthTokens

PUBLIC_URL = "https://93.184.216.34/mcp"
ISSUER = "https://93.184.216.36"
ORIGIN = "https://otari.example.com"
SECRETS = ("user-access-token", "user-refresh-token")


@pytest.fixture(autouse=True)
def _deployment(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)
    yield


@pytest.fixture(autouse=True)
def _wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stands in for the MCP server's authorization server and for the connection check."""

    async def discover(server_url: str) -> McpOAuthMetadata:
        return McpOAuthMetadata(
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            registration_endpoint=f"{ISSUER}/register",
        )

    async def register_client(registration_endpoint: str, redirect_uri: str) -> McpOAuthClient:
        return McpOAuthClient(client_id="dynamic", auth_method=AUTH_METHOD_NONE)

    async def exchange_code(*args: Any, **kwargs: Any) -> McpOAuthTokens:
        return McpOAuthTokens(access_token=SECRETS[0], refresh_token=SECRETS[1], expires_in=3600)

    async def check(server: ResolvedMcpServer) -> ServerCheck:
        return ServerCheck(tool_names=[])

    monkeypatch.setattr(_mcp_oauth, "discover", discover)
    monkeypatch.setattr(_mcp_oauth, "register_client", register_client)
    monkeypatch.setattr(_mcp_oauth, "exchange_code", exchange_code)
    monkeypatch.setattr(mcp_stateless, "check_stored_server", check)


@pytest.fixture
def user_key(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    """An API key owned by a user."""
    created = client.post(f"{API_ROOT}/users", json={"user_id": "alice"}, headers=master_key_header)
    assert created.status_code == 200, created.text
    response = client.post(
        f"{API_ROOT}/keys", json={"key_name": "alice", "user_id": "alice"}, headers=master_key_header
    )
    assert response.status_code == 200, response.text
    key: dict[str, Any] = response.json()
    return key


def _bearer(key: dict[str, Any]) -> dict[str, str]:
    return {API_KEY_HEADER: f"Bearer {key['key']}"}


def _workspace(test_db: Session, key: dict[str, Any]) -> uuid.UUID:
    row = test_db.get(APIKey, key["id"])
    assert row is not None
    return row.workspace_id


def _per_user_server(client: TestClient, master: dict[str, str], workspace: uuid.UUID) -> str:
    response = client.post(
        f"{API_ROOT}/workspaces/{workspace}/mcp-servers",
        json={"name": "drive", "url": PUBLIC_URL, "credential_mode": "per_user"},
        headers=master,
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert (body["credential_mode"], body["has_token"], body["oauth_client_id"]) == ("per_user", False, None)
    server_id: str = body["id"]
    return server_id


def _connect(test_db: Session, server_id: str, user_id: str) -> None:
    test_db.add(
        McpUserConnection(
            server_id=uuid.UUID(server_id),
            user_id=user_id,
            encrypted_access_token=encrypt_secret(SECRETS[0]),
            encrypted_refresh_token=encrypt_secret(SECRETS[1]),
            client_id="dynamic",
            client_auth_method=AUTH_METHOD_NONE,
            token_endpoint=f"{ISSUER}/token",
        )
    )
    test_db.commit()


def _landing(response: Any) -> dict[str, str]:
    """The dashboard route a callback redirected to, and its query."""
    assert response.status_code == 303, response.text
    location = response.headers["location"]
    assert location.startswith(f"{ORIGIN}/#/mcp-connections/callback?"), location
    return {key: value[0] for key, value in parse_qs(location.partition("?")[2]).items()}


def test_a_member_signs_in_lists_their_connection_and_disconnects(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, user_key: dict[str, Any]
) -> None:
    workspace = _workspace(test_db, user_key)
    server_id = _per_user_server(client, master_key_header, workspace)
    mine = f"{API_ROOT}/workspaces/{workspace}/mcp-connections"

    started = client.post(f"{mine}/{server_id}", json={"return_to": "/tools/mcp-servers"}, headers=master_key_header)
    assert started.status_code == 200, started.text
    authorization_url = started.json()["authorization_url"]
    query = {key: value[0] for key, value in parse_qs(urlsplit(authorization_url).query).items()}
    assert authorization_url.startswith(f"{ISSUER}/authorize?")
    assert query["redirect_uri"] == f"{ORIGIN}{API_ROOT}/mcp-connections/callback"
    assert query["code_challenge_method"] == "S256"

    # The authorization server sends the browser back with no credential of ours: nothing is spent here.
    returned = client.get(
        f"{API_ROOT}/mcp-connections/callback",
        params={"state": query["state"], "code": "the-code"},
        follow_redirects=False,
    )
    assert _landing(returned) == {"state": query["state"], "code": "the-code"}
    assert client.get(mine, headers=master_key_header).json() == {"data": []}

    refused = client.post(f"{API_ROOT}/mcp-connections/complete", json={"state": query["state"], "code": "the-code"})
    assert refused.status_code in (401, 403), "finishing a sign-in takes the session that started it"
    done = client.post(
        f"{API_ROOT}/mcp-connections/complete",
        json={"state": query["state"], "code": "the-code"},
        headers=master_key_header,
    )
    assert done.status_code == 200, done.text
    assert done.json() == {
        "server_id": server_id,
        "workspace_id": str(workspace),
        "server_name": "drive",
        "return_to": "/tools/mcp-servers",
    }
    replayed = client.post(
        f"{API_ROOT}/mcp-connections/complete",
        json={"state": query["state"], "code": "the-code"},
        headers=master_key_header,
    )
    assert replayed.status_code == 400, replayed.text

    listed = client.get(mine, headers=master_key_header)
    connected = client.get(
        f"{API_ROOT}/workspaces/{workspace}/mcp-servers/{server_id}/connections", headers=master_key_header
    )
    assert [entry["server_id"] for entry in listed.json()["data"]] == [server_id]
    assert listed.json()["data"][0]["renewable"] is True
    assert len(connected.json()["data"]) == 1
    for body in (started.text, done.text, listed.text, connected.text):
        for secret in SECRETS:
            assert secret not in body

    assert client.delete(f"{mine}/{server_id}", headers=master_key_header).status_code == 204
    assert client.get(mine, headers=master_key_header).json() == {"data": []}
    assert client.delete(f"{mine}/{server_id}", headers=master_key_header).status_code == 404


def test_a_manager_removes_another_users_connection(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, user_key: dict[str, Any]
) -> None:
    workspace = _workspace(test_db, user_key)
    server_id = _per_user_server(client, master_key_header, workspace)
    _connect(test_db, server_id, "alice")
    connections = f"{API_ROOT}/workspaces/{workspace}/mcp-servers/{server_id}/connections"

    listed = client.get(connections, headers=master_key_header)
    removed = client.delete(f"{connections}/alice", headers=master_key_header)

    assert [(entry["user_id"], entry["email"]) for entry in listed.json()["data"]] == [("alice", None)]
    assert removed.status_code == 204, removed.text
    assert client.get(connections, headers=master_key_header).json() == {"data": []}
    assert client.delete(f"{connections}/alice", headers=master_key_header).status_code == 404
    assert client.get(connections, headers=_bearer(user_key)).status_code in (401, 403)


def test_the_callback_lands_on_the_dashboard_whatever_came_back(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, user_key: dict[str, Any]
) -> None:
    workspace = _workspace(test_db, user_key)
    server_id = _per_user_server(client, master_key_header, workspace)
    started = client.post(
        f"{API_ROOT}/workspaces/{workspace}/mcp-connections/{server_id}", json={}, headers=master_key_header
    )
    state = parse_qs(urlsplit(started.json()["authorization_url"]).query)["state"][0]
    callback = f"{API_ROOT}/mcp-connections/callback"

    assert _landing(client.get(callback, follow_redirects=False)) == {"error": "invalid_request"}
    assert _landing(client.get(callback, params={"state": "never", "code": "c"}, follow_redirects=False)) == {
        "error": "unknown_state"
    }
    denied = client.get(callback, params={"state": state, "error": "access_denied"}, follow_redirects=False)
    assert _landing(denied) == {"error": "access_denied"}

    assert test_db.execute(select(McpOAuthFlow)).first() is None, "a refused sign-in leaves nothing to finish"
    assert _landing(client.get(callback, params={"state": state, "code": "c"}, follow_redirects=False)) == {
        "error": "unknown_state"
    }


def test_starting_a_sign_in_is_refused_where_it_cannot_work(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    user_key: dict[str, Any],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(test_db, user_key)
    servers = f"{API_ROOT}/workspaces/{workspace}/mcp-servers"
    mine = f"{API_ROOT}/workspaces/{workspace}/mcp-connections"
    shared = client.post(servers, json={"name": "shared", "url": PUBLIC_URL}, headers=master_key_header)
    per_user = _per_user_server(client, master_key_header, workspace)

    assert shared.json()["credential_mode"] == "shared", "the default: nothing existing changes"
    assert client.post(f"{mine}/{shared.json()['id']}", json={}, headers=master_key_header).status_code == 409
    assert client.post(f"{mine}/{uuid.uuid4()}", json={}, headers=master_key_header).status_code == 404
    outside = client.post(f"{mine}/{per_user}", json={"return_to": "https://evil.example/"}, headers=master_key_header)
    assert outside.status_code == 400, outside.text

    monkeypatch.setattr(test_config, "public_base_url", None)
    unset = client.post(f"{mine}/{per_user}", json={}, headers=master_key_header)
    refused = client.post(
        servers, json={"name": "later", "url": PUBLIC_URL, "credential_mode": "per_user"}, headers=master_key_header
    )
    assert unset.status_code == 400 and "public_base_url" in unset.text
    assert refused.status_code == 400 and "public_base_url" in refused.text


def test_a_key_is_told_to_sign_in_until_its_user_has(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    user_key: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(test_db, user_key)
    server_id = _per_user_server(client, master_key_header, workspace)
    headers = _bearer(user_key)
    execute = {
        "mcp_server_id": server_id,
        "tool_name": "list_files",
        "arguments": {},
        "server_revision": "0" * 32,
        "client_execution_id": str(uuid.uuid4()),
    }
    completion = {
        "model": "openai:gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "mcp_server_ids": [server_id],
    }

    listed = client.get(f"{API_ROOT}/mcp/servers", headers=headers)
    tools = client.get(f"{API_ROOT}/mcp/servers/{server_id}/tools", headers=headers)
    executed = client.post(f"{API_ROOT}/mcp/execute", json=execute, headers=headers)
    completed = client.post(f"{API_ROOT}/chat/completions", json=completion, headers=headers)

    assert [(server["id"], server["sign_in_required"]) for server in listed.json()["servers"]] == [(server_id, True)]
    for refused in (tools, executed):
        assert refused.status_code == 403, refused.text
        assert refused.json()["code"] == MCP_SIGN_IN_REQUIRED
    assert completed.status_code == 403, completed.text
    assert completed.headers[ERROR_CODE_HEADER] == MCP_SIGN_IN_REQUIRED
    assert "drive" in completed.text and server_id in completed.text

    _connect(test_db, server_id, "alice")
    seen: list[ResolvedMcpServer] = []

    class _Session:
        async def list_tools(self, *, params: Any = None) -> ListToolsResult:
            return ListToolsResult(tools=[MCPTool(name="list_files", input_schema={"type": "object"})])

    @asynccontextmanager
    async def open_session(server: ResolvedMcpServer, *args: Any, **kwargs: Any) -> Any:
        seen.append(server)
        yield _Session()

    monkeypatch.setattr(mcp_stateless, "open_session", open_session)
    listed = client.get(f"{API_ROOT}/mcp/servers", headers=headers)
    tools = client.get(f"{API_ROOT}/mcp/servers/{server_id}/tools", headers=headers)

    assert [server["sign_in_required"] for server in listed.json()["servers"]] == [False]
    assert tools.status_code == 200, tools.text
    assert [tool["name"] for tool in tools.json()["tools"]] == ["list_files"]
    assert seen[0].authorization_token == SECRETS[0], "the call carries the key's own user's token"
    assert SECRETS[0] not in tools.text


def test_a_key_with_no_owner_is_told_a_per_user_server_does_not_exist(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, api_key_obj: dict[str, Any]
) -> None:
    """A key created without a user belongs to the shared default one, which is nobody's account."""
    workspace = _workspace(test_db, api_key_obj)
    server_id = _per_user_server(client, master_key_header, workspace)
    row = test_db.get(APIKey, api_key_obj["id"])
    assert row is not None and row.user_id == "default"
    # Even a connection stored under that user is not used: it would be one account behind every such key.
    _connect(test_db, server_id, "default")
    headers = _bearer(api_key_obj)

    listed = client.get(f"{API_ROOT}/mcp/servers", headers=headers)
    tools = client.get(f"{API_ROOT}/mcp/servers/{server_id}/tools", headers=headers)

    assert listed.json() == {"servers": []}
    assert tools.status_code == 404, tools.text
    assert tools.json()["code"] == "mcp_server_not_found"
