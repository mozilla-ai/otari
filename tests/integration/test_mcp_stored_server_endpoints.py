"""The standalone half of the caller-orchestrated MCP endpoints.

The hybrid half is unit-tested in ``tests/unit/test_mcp_execute_endpoint.py``
and ``tests/unit/test_mcp_tools_endpoint.py``, where the platform resolver is
the only seam. Here the resolution is real: a key authenticates, its workspace
is derived from the key row, and the stored server, its encrypted credential and
its allowlist come out of the database. What that buys, and what only this half
can show, is that the request-scoped database session is handed back before any
outbound work and that a master key cannot reach a tenant's stored server.

URLs are public IP literals, so ``validate_mcp_url`` never reaches a resolver
and the suite does not need egress.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPTool
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from gateway.api.routes import mcp as mcp_route
from gateway.core.config import API_ROOT
from gateway.core.database import release_session
from gateway.inflight import InFlightRegistry
from gateway.models.entities import APIKey, User, WorkspaceMcpServer
from gateway.models.mcp import ResolvedMcpServer
from gateway.models.tenancy import Organization, Workspace
from gateway.services import mcp_stateless
from gateway.services.secret_box import encrypt_secret, generate_secret_key

from .conftest import build_test_client

PUBLIC_URL = "https://93.184.216.34/mcp"
# The same public host over cleartext, for the credential rule rather than the
# private-address one.
CLEARTEXT_URL = "http://93.184.216.34/mcp"
CLIENT_EXECUTION_ID = "6e51b3bc-6f48-4c68-a61e-0786bc80cd67"

RESULT = CallToolResult(content=[TextContent(type="text", text="Created issue #42")])


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tools = [MCPTool(name="create_issue", description="Create an issue", inputSchema={"type": "object"})]
        self.pages = 0
        self.on_call: Any = None

    async def list_tools(self, cursor: str | None = None) -> ListToolsResult:
        self.pages += 1
        return ListToolsResult(tools=self.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        if self.on_call is not None:
            self.on_call()
        return RESULT


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    fake = _FakeSession()

    @asynccontextmanager
    async def open_session(*args: Any, **kwargs: Any) -> Any:
        _EVENTS.append("mcp_session_opened")
        yield fake

    monkeypatch.setattr(mcp_stateless, "open_session", open_session)
    return fake


_EVENTS: list[str] = []


@pytest.fixture(autouse=True)
def _events(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record the order of the two steps that must not be interleaved."""
    _EVENTS.clear()
    original = release_session

    async def spy(db: Any) -> bool:
        released = await original(db)
        _EVENTS.append("session_released")
        return released

    monkeypatch.setattr(mcp_route, "release_session", spy)
    yield _EVENTS


def _workspace_id(test_db: Session, api_key_id: str) -> uuid.UUID:
    key = test_db.get(APIKey, api_key_id)
    assert key is not None and key.workspace_id is not None
    return key.workspace_id


def _other_workspace(test_db: Session) -> uuid.UUID:
    """A second workspace in the deployment, to hold a server this key may not reach."""
    organization = test_db.query(Organization).first()
    assert organization is not None
    workspace = Workspace(name="Other", organization_id=organization.id)
    test_db.add(workspace)
    test_db.commit()
    test_db.refresh(workspace)
    return workspace.id


def _store_server(
    test_db: Session,
    workspace_id: uuid.UUID,
    *,
    url: str = PUBLIC_URL,
    token: str | None = "ghp_token",
    enabled: bool = True,
    allowed_tools: list[str] | None = None,
) -> WorkspaceMcpServer:
    row = WorkspaceMcpServer(
        workspace_id=workspace_id,
        name="github",
        url=url,
        encrypted_token=encrypt_secret(token) if token else None,
        enabled=enabled,
        allowed_tools=allowed_tools,
    )
    test_db.add(row)
    test_db.commit()
    test_db.refresh(row)
    return row


def _revision(row: WorkspaceMcpServer, *, token: str | None = "ghp_token") -> str:
    return ResolvedMcpServer(
        id=row.id,
        name=row.name,
        url=row.url,
        authorization_token=token,
        enabled=row.enabled,
        allowed_tools=row.allowed_tools,
    ).revision


def _execute_body(row: WorkspaceMcpServer, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "mcp_server_id": str(row.id),
        "tool_name": "create_issue",
        "arguments": {"title": "Approved title"},
        "server_revision": _revision(row),
        "client_execution_id": CLIENT_EXECUTION_ID,
    }
    body.update(overrides)
    return body


def test_a_workspace_key_executes_against_its_own_stored_server(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]), allowed_tools=["create_issue"])

    response = client.post(f"{API_ROOT}/mcp/execute", headers=api_key_header, json=_execute_body(row))

    assert response.status_code == 200, response.text
    assert response.json()["content"] == [{"type": "text", "text": "Created issue #42"}]
    assert session.calls == [("create_issue", {"title": "Approved title"})]


def test_the_stored_credential_is_decrypted_and_never_returned(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))

    tools = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=api_key_header)
    executed = client.post(f"{API_ROOT}/mcp/execute", headers=api_key_header, json=_execute_body(row))

    assert tools.status_code == 200, tools.text
    assert executed.status_code == 200, executed.text
    for secret in ("ghp_token", "93.184.216.34"):
        assert secret not in tools.text, secret
        assert secret not in executed.text, secret


def test_the_database_session_is_released_before_the_mcp_session_opens(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
    _events: list[str],
) -> None:
    """Execution step 5: a pooled connection must not be pinned across a remote call."""
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))

    response = client.post(f"{API_ROOT}/mcp/execute", headers=api_key_header, json=_execute_body(row))

    assert response.status_code == 200, response.text
    assert _events == ["session_released", "mcp_session_opened"]


def test_discovery_also_releases_the_session_before_connecting(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
    _events: list[str],
) -> None:
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))

    response = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=api_key_header)

    assert response.status_code == 200, response.text
    assert _events == ["session_released", "mcp_session_opened"]


def test_the_call_appears_in_the_in_flight_registry_while_it_runs(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    """Execution step 7, through the standard middleware lifecycle."""
    registry: InFlightRegistry = client.app.state.inflight  # type: ignore[attr-defined]
    seen: list[list[str]] = []
    session.on_call = lambda: seen.append([entry.endpoint for entry in registry.snapshot()])
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))

    response = client.post(f"{API_ROOT}/mcp/execute", headers=api_key_header, json=_execute_body(row))

    assert response.status_code == 200, response.text
    assert seen == [["/v1/mcp/execute"]]
    assert registry.snapshot() == [], "the middleware deregisters once the response is sent"


def test_discovery_registers_under_its_own_endpoint_while_it_runs(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator watching in-flight work sees which of the two endpoints is running."""
    registry: InFlightRegistry = client.app.state.inflight  # type: ignore[attr-defined]
    seen: list[list[str]] = []
    original = session.list_tools

    async def watched(cursor: str | None = None) -> ListToolsResult:
        seen.append([entry.endpoint for entry in registry.snapshot()])
        return await original(cursor)

    monkeypatch.setattr(session, "list_tools", watched)
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))

    response = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=api_key_header)

    assert response.status_code == 200, response.text
    assert seen == [["/v1/mcp/servers/{mcp_server_id}/tools"]]
    assert registry.snapshot() == []


def test_the_authenticated_principal_is_rate_limited_before_any_outbound_access(
    test_config: Any,
    postgres_url: str,
    clean_database: None,
    session: _FakeSession,
) -> None:
    """R-ADM-2: the limit is charged to the key's own subject, before the MCP call."""
    config = test_config.model_copy(update={"rate_limit_rpm": 1})
    for rate_limited_client in build_test_client(config):
        master = {"Otari-Key": f"Bearer {config.master_key}"}
        key = rate_limited_client.post(f"{API_ROOT}/keys", json={"key_name": "rl"}, headers=master).json()
        header = {"Otari-Key": f"Bearer {key['key']}"}
        with Session(create_engine(postgres_url)) as db:
            row = _store_server(db, _workspace_id(db, key["id"]))
            body = _execute_body(row)

        first = rate_limited_client.post(f"{API_ROOT}/mcp/execute", headers=header, json=body)
        second = rate_limited_client.post(f"{API_ROOT}/mcp/execute", headers=header, json=body)

        assert first.status_code == 200, first.text
        assert second.status_code == 429, second.text
        assert second.json()["code"] == "rate_limit_exceeded"
        assert second.json()["execution_state"] == "not_started"
        assert second.headers["Retry-After"]
        assert len(session.calls) == 1, "the refused request never reached the server"


def test_a_blocked_user_cannot_drive_an_outbound_mcp_call(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    """The block is a kill switch, and these endpoints are something to kill.

    ``users.blocked`` is read in ``reserve_budget``, which these routes never
    reach because they reserve nothing. Without a check of their own, blocking
    someone would stop their completions and leave them driving mutating MCP
    tools through the gateway.
    """
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))
    body = _execute_body(row)
    user = test_db.get(User, api_key_obj["user_id"])
    assert user is not None
    user.blocked = True
    test_db.add(user)
    test_db.commit()

    executed = client.post(f"{API_ROOT}/mcp/execute", headers=api_key_header, json=body)
    tools = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=api_key_header)

    for response in (executed, tools):
        assert response.status_code == 401, response.text
        assert response.json()["code"] == "authentication_failed"
        assert response.json()["execution_state"] == "not_started"
    assert session.calls == []
    assert session.pages == 0


def test_an_unblocked_user_is_unaffected(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    """The other half of the block, so the check cannot pass by refusing everyone."""
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))
    user = test_db.get(User, api_key_obj["user_id"])
    assert user is not None and user.blocked is False

    response = client.post(f"{API_ROOT}/mcp/execute", headers=api_key_header, json=_execute_body(row))

    assert response.status_code == 200, response.text


def test_a_stored_token_is_never_sent_over_cleartext_http(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    """The URL-safety rule the route's ``has_authorization_token`` argument feeds.

    A public ``http://`` host, so the private-address rule cannot short-circuit
    this and the cleartext-credential rule is what has to fire. Its pair below
    is what makes the argument itself tested: the same URL with no stored token
    has nothing to leak and is allowed through.
    """
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]), url=CLEARTEXT_URL, token="ghp_token")

    response = client.post(
        f"{API_ROOT}/mcp/execute",
        headers=api_key_header,
        json=_execute_body(row, server_revision=_revision(row)),
    )

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "unsafe_mcp_url"
    assert response.json()["execution_state"] == "not_started"
    assert session.calls == []
    assert "ghp_token" not in response.text


def test_the_same_cleartext_url_is_allowed_when_there_is_no_token_to_leak(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]), url=CLEARTEXT_URL, token=None)

    response = client.post(
        f"{API_ROOT}/mcp/execute",
        headers=api_key_header,
        json=_execute_body(row, server_revision=_revision(row, token=None)),
    )

    assert response.status_code == 200, response.text
    assert session.calls == [("create_issue", {"title": "Approved title"})]


def test_a_master_key_cannot_reach_a_workspaces_stored_server(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    master_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    """Operator credentials hold no workspace, so no stored server is theirs to run."""
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))

    executed = client.post(f"{API_ROOT}/mcp/execute", headers=master_key_header, json=_execute_body(row))
    tools = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=master_key_header)

    for response in (executed, tools):
        assert response.status_code == 404, response.text
        assert response.json()["code"] == "mcp_server_not_found"
    assert session.calls == []
    assert session.pages == 0


def test_a_server_stored_in_another_workspace_is_not_found(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    row = _store_server(test_db, _other_workspace(test_db))

    response = client.post(f"{API_ROOT}/mcp/execute", headers=api_key_header, json=_execute_body(row))

    assert response.status_code == 404, response.text
    assert response.json()["code"] == "mcp_server_not_found"
    assert session.calls == []


def test_a_disabled_stored_server_is_not_found(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]), enabled=False)

    response = client.post(f"{API_ROOT}/mcp/execute", headers=api_key_header, json=_execute_body(row))

    assert response.status_code == 404, response.text
    assert session.calls == []


def test_a_credential_that_will_not_decrypt_is_a_sanitized_five_hundred(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connecting without the credential the workspace configured is not an option."""
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    response = client.post(f"{API_ROOT}/mcp/execute", headers=api_key_header, json=_execute_body(row))

    assert response.status_code == 500, response.text
    assert response.json()["code"] == "mcp_credentials_unavailable"
    assert response.json()["execution_state"] == "not_started"
    assert session.calls == []


def test_the_stored_allowlist_narrows_discovery_and_execution_alike(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]), allowed_tools=["create_issue"])
    session.tools.append(MCPTool(name="delete_repo", description="danger", inputSchema={"type": "object"}))

    tools = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=api_key_header)
    refused = client.post(
        f"{API_ROOT}/mcp/execute",
        headers=api_key_header,
        json=_execute_body(row, tool_name="delete_repo"),
    )

    assert [tool["name"] for tool in tools.json()["tools"]] == ["create_issue"]
    assert refused.status_code == 403, refused.text
    assert refused.json()["code"] == "mcp_tool_not_allowed"
    assert session.calls == []


def test_the_discovered_revision_is_what_execution_accepts(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    """The two endpoints agree on the revision, which is the whole point of it."""
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))

    discovered = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=api_key_header).json()
    executed = client.post(
        f"{API_ROOT}/mcp/execute",
        headers=api_key_header,
        json=_execute_body(row, server_revision=discovered["server_revision"]),
    )

    assert executed.status_code == 200, executed.text


def test_a_configuration_change_after_discovery_is_refused_without_contacting_the_server(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    session: _FakeSession,
) -> None:
    row = _store_server(test_db, _workspace_id(test_db, api_key_obj["id"]))
    discovered = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=api_key_header).json()

    row.url = "https://93.184.216.35/mcp"
    test_db.add(row)
    test_db.commit()

    response = client.post(
        f"{API_ROOT}/mcp/execute",
        headers=api_key_header,
        json=_execute_body(row, server_revision=discovered["server_revision"]),
    )

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "mcp_server_changed"
    assert session.calls == []
