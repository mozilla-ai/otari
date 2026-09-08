"""``GET /v1/mcp/servers/{id}/tools``: the stored-server discovery contract.

An application calls this once per server when it prepares a model or workflow
run, and every proposed call from that server reuses the answer. The response is
the whole authorized catalog or nothing (R-DISC-5), and it carries neither the
stored URL, the credential, nor the allowlist itself (R-DISC-2).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp.types import ListToolsResult, ToolAnnotations
from mcp.types import Tool as MCPTool

from gateway.api.deps import reset_config
from gateway.api.routes import _platform as platform_module
from gateway.api.routes import mcp as mcp_route
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.models.mcp import ResolvedMcpServer
from gateway.services import mcp_stateless

SERVER_ID = uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e")
PUBLIC_URL = "https://93.184.216.34/mcp"
USER_AUTH = {"Authorization": "Bearer platform-user-token"}
TOOLS_PATH = f"/v1/mcp/servers/{SERVER_ID}/tools"


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


class _Platform:
    def __init__(self) -> None:
        self.servers: list[dict[str, Any]] | None = [_stored().model_dump(mode="json")]
        self.status_code = 200
        self.bodies: list[dict[str, Any]] = []
        self.delay_s = 0.0


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> _Platform:
    fake = _Platform()

    async def post(*, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> Any:
        fake.bodies.append(body)
        await asyncio.sleep(fake.delay_s)
        payload = {"servers": fake.servers} if fake.status_code == 200 else {"detail": "refused"}
        return httpx.Response(fake.status_code, json=payload)

    monkeypatch.setattr(platform_module, "_post_platform", post)
    return fake


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw-test-token")
    app = create_app(GatewayConfig(mode="hybrid", platform={"base_url": "http://platform.test/api/v1"}))
    with TestClient(app) as test_client:
        yield test_client


def test_the_live_catalog_is_returned_with_the_stored_server_revision(
    client: TestClient,
    platform: _Platform,
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
    assert response.headers["X-Otari-Request-ID"]


def test_the_response_discloses_no_url_credential_or_allowlist(
    client: TestClient,
    platform: _Platform,
    session: _FakeSession,
) -> None:
    """R-DISC-2, R-CAT-2: the allowlist's effect shows; its contents never do."""
    platform.servers = [_stored(allowed_tools=["create_issue", "never_listed"]).model_dump(mode="json")]

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 200, response.text
    assert [tool["name"] for tool in response.json()["tools"]] == ["create_issue"]
    for secret in ("93.184.216.34", "server-secret", "never_listed"):
        assert secret not in response.text, secret


def test_a_tool_the_allowlist_excludes_is_not_exposed(
    client: TestClient,
    platform: _Platform,
    session: _FakeSession,
) -> None:
    """R-DISC-1: a server-added tool is not exposed just because it was listed."""
    session.tools.append(MCPTool(name="delete_repo", description="danger", inputSchema={"type": "object"}))
    platform.servers = [_stored(allowed_tools=["create_issue"]).model_dump(mode="json")]

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert [tool["name"] for tool in response.json()["tools"]] == ["create_issue"]


def test_an_explicit_empty_allowlist_denies_every_tool_without_connecting(
    client: TestClient,
    platform: _Platform,
    session: _FakeSession,
) -> None:
    """R-RES-4: an operator's deny-all is answered without touching the server."""
    stored = _stored(allowed_tools=[])
    platform.servers = [stored.model_dump(mode="json")]

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
    platform: _Platform,
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
    assert platform.bodies == []
    assert session.pages == 0


def test_the_total_deadline_includes_platform_resolution(
    client: TestClient,
    platform: _Platform,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_stateless, "DISCOVERY_TOTAL_TIMEOUT_S", 0.01)
    platform.delay_s = 10

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "mcp_discovery_limit_exceeded"
    assert session.pages == 0


def test_the_total_deadline_includes_mcp_discovery(
    client: TestClient,
    platform: _Platform,
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
    platform: _Platform,
    session: _FakeSession,
) -> None:
    """R-SCHEMA-3: the one permitted partial result, and its siblings survive."""
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
    platform: _Platform,
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
    platform: _Platform,
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
    platform: _Platform,
    session: _FakeSession,
) -> None:
    """R-SCHEMA-2: Otari never truncates, rewrites, or drops a keyword."""
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
    platform: _Platform,
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


def test_an_unauthenticated_request_reaches_neither_platform_nor_server(
    client: TestClient,
    platform: _Platform,
    session: _FakeSession,
) -> None:
    response = client.get(TOOLS_PATH)

    assert response.status_code == 401
    assert response.json()["code"] == "authentication_failed"
    assert platform.bodies == []
    assert session.pages == 0


def test_a_disabled_server_is_not_found(
    client: TestClient,
    platform: _Platform,
    session: _FakeSession,
) -> None:
    platform.servers = [_stored(enabled=False).model_dump(mode="json")]

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 404, response.text
    assert response.json()["code"] == "mcp_server_not_found"
    assert session.pages == 0


def test_an_unsafe_resolved_url_is_refused_before_connecting(
    client: TestClient,
    platform: _Platform,
    session: _FakeSession,
) -> None:
    platform.servers = [_stored(url="http://169.254.169.254/mcp").model_dump(mode="json")]

    response = client.get(TOOLS_PATH, headers=USER_AUTH)

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "unsafe_mcp_url"
    assert session.pages == 0


def test_a_non_uuid_server_id_is_refused(
    client: TestClient,
    platform: _Platform,
    session: _FakeSession,
) -> None:
    response = client.get("/v1/mcp/servers/not-a-uuid/tools", headers=USER_AUTH)

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "invalid_request"
    assert platform.bodies == []
