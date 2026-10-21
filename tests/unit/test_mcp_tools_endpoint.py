"""``GET /api/v1/mcp/servers/{id}/tools``: the stored-server discovery contract.

An application calls this once per server when it prepares a model or workflow
run, and every proposed call from that server reuses the answer. The response is
the whole authorized catalog or nothing, and it carries neither the stored URL,
the credential, nor the allowlist itself.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from mcp.types import ListToolsResult, ToolAnnotations
from mcp.types import Tool as MCPTool

from gateway.api.deps import get_workspace_mcp_servers, reset_config
from gateway.api.routes import mcp as mcp_route
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.models.mcp import ResolvedMcpServer
from gateway.services import mcp_stateless

SERVER_ID = uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e")
PUBLIC_URL = "https://93.184.216.34/mcp"
USER_AUTH = {"Authorization": "Bearer sk-workspace-key"}
WORKSPACE_ID = uuid.UUID("0b6f3c1e-8d2a-4f7b-9c5e-1a2b3c4d5e6f")
TOOLS_PATH = f"{API_ROOT}/mcp/servers/{SERVER_ID}/tools"


@pytest.fixture(autouse=True)
def _reset_globals() -> Iterator[None]:
    yield
    reset_config()
    reset_db()


def _stored(**overrides: Any) -> ResolvedMcpServer:
    fields: dict[str, Any] = {
        "id": SERVER_ID,
        "name": "github",
        "url": PUBLIC_URL,
        "authorization_token": "server-secret",
        "enabled": True,
        "allowed_tools": None,
    }
    fields.update(overrides)
    return ResolvedMcpServer(**fields)


class _FakeSession:
    def __init__(self) -> None:
        self.tools: list[MCPTool] = [
            MCPTool(
                name="create_issue",
                description="Create an issue",
                inputSchema={"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
                annotations=ToolAnnotations(readOnlyHint=False),
            )
        ]
        self.pages = 0
        self.delay_s = 0.0

    async def list_tools(self, cursor: str | None = None) -> ListToolsResult:
        self.pages += 1
        await asyncio.sleep(self.delay_s)
        return ListToolsResult(tools=self.tools)


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    fake = _FakeSession()

    @asynccontextmanager
    async def open_session(*args: Any, **kwargs: Any) -> Any:
        yield fake

    monkeypatch.setattr(mcp_stateless, "open_session", open_session)
    return fake


class _Servers:
    """The workspace's MCP servers, and a record of what was asked of them."""

    def __init__(self) -> None:
        self.server: ResolvedMcpServer | None = _stored()
        self.asked: list[tuple[uuid.UUID, uuid.UUID]] = []
        self.delay_s = 0.0

    async def resolve_many(self, workspace_id: uuid.UUID, server_ids: list[uuid.UUID]) -> Any:
        raise AssertionError("the stateless routes resolve one server")

    async def resolve_one(self, workspace_id: uuid.UUID, server_id: uuid.UUID) -> ResolvedMcpServer | None:
        self.asked.append((workspace_id, server_id))
        await asyncio.sleep(self.delay_s)
        return self.server


@pytest.fixture
def servers() -> _Servers:
    return _Servers()


@pytest.fixture(scope="module")
def database_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One database for the module: the route reads no row, so migrating it once is enough."""
    return f"sqlite:///{tmp_path_factory.mktemp('mcp') / 'mcp-tools.db'}"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, database_url: str, servers: _Servers) -> Iterator[TestClient]:
    """A standalone app whose key lookup answers one workspace key for any bearer token."""

    async def verify(raw_request: Request, *args: Any, **kwargs: Any) -> tuple[Any, bool]:
        if not raw_request.headers.get("Authorization"):
            raise HTTPException(status_code=401, detail="Missing API key")
        return SimpleNamespace(id=uuid.uuid4(), user_id=None), False

    async def workspace_of(*args: Any, **kwargs: Any) -> uuid.UUID:
        return WORKSPACE_ID

    monkeypatch.setattr(mcp_route, "verify_api_key_or_master_key", verify)
    monkeypatch.setattr(mcp_route, "resolve_workspace_id", workspace_of)
    app = create_app(GatewayConfig(database_url=database_url))
    app.dependency_overrides[get_workspace_mcp_servers] = lambda: servers
    with TestClient(app) as test_client:
        yield test_client


def test_the_live_catalog_is_returned_with_the_stored_server_revision(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "server_id": str(SERVER_ID),
        "server_revision": _stored().revision,
        "tools": [
            {
                "name": "create_issue",
                "description": "Create an issue",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
                "annotations": {"readOnlyHint": False},
            }
        ],
        "warnings": [],
    }
    assert response.headers["Otari-Request-ID"]
    assert servers.asked == [(WORKSPACE_ID, SERVER_ID)]


def test_the_response_discloses_no_url_credential_or_allowlist(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    """The allowlist's effect shows; its contents never do."""
    servers.server = _stored(allowed_tools=["create_issue", "never_listed"])

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 200, response.text
    assert [tool["name"] for tool in response.json()["tools"]] == ["create_issue"]
    for secret in ("93.184.216.34", "server-secret", "never_listed"):
        assert secret not in response.text, secret


def test_a_tool_the_allowlist_excludes_is_not_exposed(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    """A server-added tool is not exposed just because it was listed."""
    session.tools.append(MCPTool(name="delete_repo", description="danger", inputSchema={"type": "object"}))
    servers.server = _stored(allowed_tools=["create_issue"])

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert [tool["name"] for tool in response.json()["tools"]] == ["create_issue"]


def test_an_explicit_empty_allowlist_denies_every_tool_without_connecting(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    """An operator's deny-all is answered without touching the server."""
    stored = _stored(allowed_tools=[])
    servers.server = stored

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "server_id": str(SERVER_ID),
        "server_revision": stored.revision,
        "tools": [],
        "warnings": [],
    }
    assert session.pages == 0


def test_the_total_deadline_includes_authentication(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_authentication(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(10)

    monkeypatch.setattr(mcp_stateless, "DISCOVERY_TOTAL_TIMEOUT_S", 0.01)
    monkeypatch.setattr(mcp_route, "_authenticate", slow_authentication)

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "mcp_discovery_limit_exceeded"
    assert servers.asked == []
    assert session.pages == 0


def test_the_total_deadline_includes_server_resolution(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_stateless, "DISCOVERY_TOTAL_TIMEOUT_S", 0.01)
    servers.delay_s = 10

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "mcp_discovery_limit_exceeded"
    assert session.pages == 0


def test_the_total_deadline_includes_mcp_discovery(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_stateless, "DISCOVERY_TOTAL_TIMEOUT_S", 0.01)
    session.delay_s = 10

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "mcp_discovery_limit_exceeded"
    assert session.pages == 1


def test_an_unusable_descriptor_is_omitted_and_labeled(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    """The one permitted partial result, and its siblings survive."""
    session.tools.append(
        MCPTool(
            name="broken",
            description="an external reference",
            inputSchema={"type": "object", "properties": {"x": {"$ref": "https://attacker.example.com/s.json"}}},
        )
    )

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 200, response.text
    assert [tool["name"] for tool in response.json()["tools"]] == ["create_issue"]
    assert response.json()["warnings"] == [{"tool_name": "broken", "code": "mcp_tool_schema_unsupported"}]
    assert "attacker.example.com" not in response.text


def test_an_unexecutable_tool_name_is_omitted_and_labeled(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    session.tools.append(
        MCPTool(
            name="x" * (mcp_stateless.TOOL_NAME_MAX_LENGTH + 1),
            description="too long to execute",
            inputSchema={"type": "object"},
        )
    )

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 200, response.text
    assert [tool["name"] for tool in response.json()["tools"]] == ["create_issue"]
    assert response.json()["warnings"] == [
        {
            "tool_name": "x" * (mcp_stateless.TOOL_NAME_MAX_LENGTH + 1),
            "code": "mcp_tool_name_unsupported",
        }
    ]


def test_the_response_ceiling_includes_warnings_and_envelope(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def oversized_catalog(*args: Any, **kwargs: Any) -> mcp_stateless.DiscoveredCatalog:
        return mcp_stateless.DiscoveredCatalog(
            tools=[],
            warnings=[("x" * mcp_stateless.DISCOVERY_RESPONSE_MAX_BYTES, "mcp_tool_schema_unsupported")],
        )

    monkeypatch.setattr(mcp_route, "discover_stored_tools", oversized_catalog)

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "mcp_discovery_limit_exceeded"


def test_an_unknown_schema_keyword_survives_the_round_trip(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    """Otari never truncates, rewrites, or drops a keyword."""
    schema = {
        "$schema": "https://example.com/draft/2044-01/schema",
        "type": "object",
        "properties": {"title": {"type": "string", "x-vendor-widget": "textarea"}},
        "x-vendor-policy": {"retries": 3},
    }
    session.tools = [MCPTool(name="create_issue", description="d", inputSchema=schema)]

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.json()["tools"][0]["input_schema"] == schema


def test_a_pagination_failure_returns_no_partial_catalog(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursors = ["loop", "loop"]

    async def looping(cursor: str | None = None) -> ListToolsResult:
        session.pages += 1
        return ListToolsResult(tools=session.tools, nextCursor=cursors[min(session.pages - 1, 1)])

    monkeypatch.setattr(session, "list_tools", looping)

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 502, response.text
    payload = response.json()
    assert payload["code"] == "mcp_discovery_limit_exceeded"
    assert payload["execution_state"] == "not_started"
    assert "tools" not in payload


def test_an_unauthenticated_request_reaches_neither_resolution_nor_server(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    response = client.get(TOOLS_PATH)

    assert response.status_code == 401
    assert response.json()["code"] == "authentication_failed"
    assert servers.asked == []
    assert session.pages == 0


def test_a_disabled_server_is_not_found(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    servers.server = _stored(enabled=False)

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 404, response.text
    assert response.json()["code"] == "mcp_server_not_found"
    assert session.pages == 0


def test_an_unsafe_resolved_url_is_refused_before_connecting(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    servers.server = _stored(url="http://169.254.169.254/mcp")

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "unsafe_mcp_url"
    assert session.pages == 0


def test_a_non_uuid_server_id_is_refused(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    response = client.get(f"{API_ROOT}/mcp/servers/not-a-uuid/tools", headers=USER_AUTH)

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "invalid_request"
    assert servers.asked == []


def test_an_id_naming_no_server_is_not_found(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    servers.server = None

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 404, response.text
    assert response.json()["code"] == "mcp_server_not_found"
    assert session.pages == 0
