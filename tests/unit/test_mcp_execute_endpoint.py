"""``POST /api/v1/mcp/execute``: the caller-orchestrated execution contract.

Otari executes one exact call an application has already authorized. It proves
no approval and claims none; what it enforces is authentication,
stored-server access, the stored allowlist, URL safety, and its own bounds.

Exercised with the MCP server port and the key lookup substituted, so the
route, the revision check and the bounded execution all run for real. The same
ladder against a real key and a stored row is in
``tests/integration/test_mcp_stored_server_endpoints.py``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from mcp.types import CallToolResult, TextContent

from gateway import log_config
from gateway.api.deps import get_workspace_mcp_servers, reset_config
from gateway.api.routes import mcp as mcp_route
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.models.mcp import ResolvedMcpServer
from gateway.services import mcp_stateless
from gateway.services.secret_box import SecretDecryptionError

SERVER_ID = uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e")
CLIENT_EXECUTION_ID = "6e51b3bc-6f48-4c68-a61e-0786bc80cd67"
# A public IP literal, so URL safety never reaches a DNS resolver.
PUBLIC_URL = "https://93.184.216.34/mcp"
USER_AUTH = {"Authorization": "Bearer sk-workspace-key"}
WORKSPACE_ID = uuid.UUID("0b6f3c1e-8d2a-4f7b-9c5e-1a2b3c4d5e6f")

RESULT = CallToolResult(
    content=[TextContent(type="text", text="Created issue #42")],
    structuredContent={"issue_number": 42},
    isError=False,
)


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
        "allowed_tools": ["create_issue"],
    }
    fields.update(overrides)
    return ResolvedMcpServer(**fields)


class _FakeSession:
    """The remote MCP server, as the route's bounded session sees it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.listed = 0
        self.result = RESULT
        self.call_error: BaseException | None = None

    async def list_tools(self, cursor: str | None = None) -> Any:
        self.listed += 1
        raise AssertionError("execution must never call list_tools")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        if self.call_error is not None:
            raise self.call_error
        return self.result


class _Servers:
    """The workspace's MCP servers, and a record of what was asked of them."""

    def __init__(self) -> None:
        self.server: ResolvedMcpServer | None = _stored()
        self.raises: Exception | None = None
        self.asked: list[tuple[uuid.UUID, uuid.UUID]] = []
        self.delay_s = 0.0

    async def resolve_many(self, workspace_id: uuid.UUID, server_ids: list[uuid.UUID]) -> Any:
        raise AssertionError("the stateless routes resolve one server")

    async def resolve_one(self, workspace_id: uuid.UUID, server_id: uuid.UUID) -> ResolvedMcpServer | None:
        await asyncio.sleep(self.delay_s)
        self.asked.append((workspace_id, server_id))
        if self.raises is not None:
            raise self.raises
        return self.server


@pytest.fixture
def servers() -> _Servers:
    return _Servers()


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    fake = _FakeSession()

    @asynccontextmanager
    async def open_session(*args: Any, **kwargs: Any) -> Any:
        yield fake

    monkeypatch.setattr(mcp_stateless, "open_session", open_session)
    return fake


@pytest.fixture(scope="module")
def database_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One database for the module: the route reads no row, so migrating it once is enough."""
    return f"sqlite:///{tmp_path_factory.mktemp('mcp') / 'mcp-execute.db'}"


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


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "mcp_server_id": str(SERVER_ID),
        "tool_name": "create_issue",
        "arguments": {"title": "Approved title", "body": "Approved body"},
        "server_revision": _stored().revision,
        "client_execution_id": CLIENT_EXECUTION_ID,
    }
    body.update(overrides)
    return body


def _error(response: Any) -> dict[str, Any]:
    payload = dict(response.json())
    assert payload.pop("request_id"), "every error body carries the opaque request id"
    return payload


# --------------------------------------------------------------------------- #
# The authorized call
# --------------------------------------------------------------------------- #


def test_the_exact_authorized_call_runs_and_the_native_result_comes_back(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 200, response.text
    assert response.json() == {
        "content": [{"type": "text", "text": "Created issue #42"}],
        "structuredContent": {"issue_number": 42},
        "isError": False,
    }
    assert session.calls == [("create_issue", {"title": "Approved title", "body": "Approved body"})]
    assert session.listed == 0, "authorization comes from the stored allowlist, not live discovery"
    assert servers.asked == [(WORKSPACE_ID, SERVER_ID)]


def test_the_request_id_is_returned_on_success_without_touching_the_result(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    """The successful body stays the native MCP result, so the id rides a header."""
    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.headers["Otari-Request-ID"]
    assert "request_id" not in response.json()


def test_execution_timings_include_server_resolution(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = Mock()
    monkeypatch.setattr(log_config.logger, "info", info)
    servers.delay_s = 0.02

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 200, response.text
    info.assert_called_once()
    duration_ms = info.call_args.args[6]
    phases = dict(field.split("=", 1) for field in info.call_args.args[7].split())
    resolve_ms = float(phases["resolve_ms"])
    assert resolve_ms >= 15
    assert duration_ms >= resolve_ms


def test_a_server_reported_error_is_a_definitive_two_hundred(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    session.result = CallToolResult(content=[TextContent(type="text", text="denied")], isError=True)

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 200, response.text
    assert response.json()["isError"] is True


def test_an_absent_allowlist_admits_any_live_tool(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    """``null`` carries the same meaning it does in the managed loop."""
    stored = _stored(allowed_tools=None)
    servers.server = stored

    response = client.post(
        f"{API_ROOT}/mcp/execute",
        headers=USER_AUTH,
        json=_body(tool_name="anything_at_all", server_revision=stored.revision),
    )

    assert response.status_code == 200, response.text
    assert session.calls == [("anything_at_all", {"title": "Approved title", "body": "Approved body"})]


# --------------------------------------------------------------------------- #
# Refusals that open no MCP connection
# --------------------------------------------------------------------------- #


def test_an_unauthenticated_request_reaches_neither_resolution_nor_server(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    response = client.post(f"{API_ROOT}/mcp/execute", json=_body())

    assert response.status_code == 401
    assert _error(response) == {
        "detail": "Authentication failed",
        "code": "authentication_failed",
        "execution_state": "not_started",
    }
    assert servers.asked == []
    assert session.calls == []


@pytest.mark.parametrize("allowed_tools", [[], ["read_issue"]])
def test_a_tool_outside_the_stored_allowlist_is_refused_before_connecting(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    allowed_tools: list[str],
) -> None:
    stored = _stored(allowed_tools=allowed_tools)
    servers.server = stored

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body(server_revision=stored.revision))

    assert response.status_code == 403, response.text
    assert _error(response)["code"] == "mcp_tool_not_allowed"
    assert _error(response)["execution_state"] == "not_started"
    assert session.calls == []


def test_a_stale_revision_is_refused_without_a_second_resolver_call(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    """The comparison is in memory, over the resolution already needed."""
    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body(server_revision="stale-revision"))

    assert response.status_code == 409, response.text
    assert _error(response)["code"] == "mcp_server_changed"
    assert len(servers.asked) == 1
    assert session.calls == []


def test_a_disabled_server_is_not_found(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    stored = _stored(enabled=False)
    servers.server = stored

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body(server_revision=stored.revision))

    assert response.status_code == 404, response.text
    assert _error(response)["code"] == "mcp_server_not_found"
    assert session.calls == []


def test_an_id_naming_no_server_is_not_found(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    servers.server = None

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 404, response.text
    assert _error(response) == {
        "detail": "MCP server not found",
        "code": "mcp_server_not_found",
        "execution_state": "not_started",
    }
    assert session.calls == []


def test_a_stored_credential_that_cannot_be_decrypted_is_refused_before_connecting(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    """Connecting without the configured credential would send an unauthenticated request."""
    servers.raises = SecretDecryptionError("stored credential cannot be decrypted")

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 500, response.text
    assert _error(response)["code"] == "mcp_credentials_unavailable"
    assert _error(response)["execution_state"] == "not_started"
    assert session.calls == []


def test_an_unsafe_resolved_url_is_refused_before_connecting(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    stored = _stored(url="http://169.254.169.254/mcp")
    servers.server = stored

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body(server_revision=stored.revision))

    assert response.status_code == 400, response.text
    assert _error(response) == {
        "detail": "MCP server URL is unsafe",
        "code": "unsafe_mcp_url",
        "execution_state": "not_started",
    }
    assert session.calls == []


# --------------------------------------------------------------------------- #
# Request validation, normalized into the shared error shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    [
        {"client_execution_id": "not-a-uuid"},
        {"client_execution_id": None},
        {"mcp_server_id": "not-a-uuid"},
        {"tool_name": ""},
        {"tool_name": "x" * 257},
        {"arguments": ["not", "an", "object"]},
        {"server_revision": ""},
        {"server_revision": "has spaces"},
        {"server_revision": "x" * 129},
    ],
)
def test_an_invalid_request_is_refused_before_resolution_or_logging(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    overrides: dict[str, Any],
) -> None:
    """Arbitrary caller text must not reach a resolver, a log, or telemetry."""
    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body(**overrides))

    assert response.status_code == 422, response.text
    assert _error(response) == {
        "detail": "MCP request is invalid",
        "code": "invalid_request",
        "execution_state": "not_started",
    }
    assert servers.asked == []
    assert session.calls == []


def test_a_non_json_parse_failure_uses_the_shared_invalid_request_error(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_to_parse(_request: Request) -> Any:
        raise RecursionError

    monkeypatch.setattr(Request, "json", fail_to_parse)

    response = client.post(
        f"{API_ROOT}/mcp/execute",
        headers={**USER_AUTH, "Content-Type": "application/json"},
        content=b"{}",
    )

    assert response.status_code == 422, response.text
    assert _error(response) == {
        "detail": "MCP request is invalid",
        "code": "invalid_request",
        "execution_state": "not_started",
    }
    assert servers.asked == []
    assert session.calls == []


def test_a_local_service_unavailability_keeps_its_status(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise HTTPException(status_code=503, detail="internal detail")

    monkeypatch.setattr("gateway.api.routes.mcp._authenticate", unavailable)

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 503, response.text
    assert _error(response) == {
        "detail": "MCP service is unavailable",
        "code": "service_unavailable",
        "execution_state": "not_started",
    }
    assert servers.asked == []
    assert session.calls == []


def test_an_inline_server_configuration_is_not_accepted(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    """A caller never transmits a URL, a credential, or a policy."""
    body = _body()
    body["server"] = {"name": "github", "url": "https://attacker.example.com/mcp"}

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=body)

    assert response.status_code == 422, response.text
    assert _error(response)["code"] == "invalid_request"
    assert servers.asked == []


def test_oversized_arguments_are_refused(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    response = client.post(
        f"{API_ROOT}/mcp/execute",
        headers=USER_AUTH,
        json=_body(arguments={"body": "x" * (mcp_stateless.ARGUMENTS_MAX_BYTES + 1)}),
    )

    assert response.status_code == 422, response.text
    assert _error(response)["code"] == "invalid_request"
    assert session.calls == []


def test_too_deeply_nested_arguments_are_refused(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    nested: dict[str, Any] = {}
    for _ in range(mcp_stateless.ARGUMENTS_MAX_DEPTH + 2):
        nested = {"next": nested}

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body(arguments=nested))

    assert response.status_code == 422, response.text
    assert _error(response)["code"] == "invalid_request"
    assert session.calls == []


# --------------------------------------------------------------------------- #
# The dispatch boundary
# --------------------------------------------------------------------------- #


def test_a_failure_after_dispatch_is_an_unknown_outcome(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    session.call_error = RuntimeError("server-secret connection reset")

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 502, response.text
    assert _error(response) == {
        "detail": "MCP execution outcome is unknown",
        "code": "mcp_outcome_unknown",
        "execution_state": "outcome_unknown",
    }
    assert "server-secret" not in response.text


def test_a_deadline_after_dispatch_is_an_unknown_outcome(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_stateless, "CALL_TIMEOUT_S", 0.01)

    async def never_answer(name: str, arguments: dict[str, Any]) -> CallToolResult:
        await asyncio.sleep(10)
        raise AssertionError("unreachable")

    monkeypatch.setattr(session, "call_tool", never_answer)

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 504, response.text
    assert _error(response)["execution_state"] == "outcome_unknown"


def test_the_total_deadline_includes_server_resolution_and_logs_the_outcome(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = Mock()
    monkeypatch.setattr(log_config.logger, "info", info)
    monkeypatch.setattr(mcp_stateless, "EXECUTION_TOTAL_TIMEOUT_S", 0.01)
    servers.delay_s = 10

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 502, response.text
    assert _error(response) == {
        "detail": "MCP server connection failed",
        "code": "mcp_connection_failed",
        "execution_state": "not_started",
    }
    info.assert_called_once()
    assert info.call_args.args[4:6] == ("mcp_connection_failed", "not_started")
    assert session.calls == []


def test_an_oversized_result_is_an_unknown_outcome(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    session.result = CallToolResult(content=[TextContent(type="text", text="x" * (mcp_stateless.RESULT_MAX_BYTES + 1))])

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 502, response.text
    assert _error(response)["code"] == "mcp_result_too_large"
    assert _error(response)["execution_state"] == "outcome_unknown"


def test_no_error_body_carries_a_url_credential_argument_or_exception_text(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing sensitive reaches the one boundary a caller can read."""
    warning = Mock()
    monkeypatch.setattr(log_config.logger, "warning", warning)
    session.call_error = RuntimeError(f"{PUBLIC_URL} server-secret title=Approved title")

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    for secret in ("93.184.216.34", "server-secret", "Approved title", "RuntimeError"):
        assert secret not in response.text, secret
    logged = " ".join(str(call) for call in warning.call_args_list)
    for secret in ("93.184.216.34", "server-secret", "Approved title"):
        assert secret not in logged, secret


@pytest.mark.parametrize("path", [f"{API_ROOT}/mcp/execute", f"{API_ROOT}/mcp/servers/{SERVER_ID}/tools"])
def test_a_capacity_refusal_is_the_only_failure_that_invites_a_retry(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """Capacity is transient, so it says so; nothing else advertises a retry."""
    gate = mcp_stateless.ConcurrencyGate(limit=1, admission_timeout_s=0.01)
    monkeypatch.setattr(mcp_stateless, "EXECUTION_GATE", gate)
    monkeypatch.setattr(mcp_stateless, "DISCOVERY_GATE", gate)

    async def hold_the_only_slot() -> Any:
        async with gate.slot():
            return await asyncio.to_thread(
                lambda: (
                    client.post(path, headers=USER_AUTH, json=_body())
                    if path.endswith("execute")
                    else client.get(path, headers=USER_AUTH)
                )
            )

    response = asyncio.run(hold_the_only_slot())

    assert response.status_code == 503, response.text
    assert response.headers["Retry-After"] == "1"
    assert _error(response)["code"] in {"mcp_capacity_unavailable", "mcp_discovery_capacity_unavailable"}
    assert _error(response)["execution_state"] == "not_started"


def test_a_failure_after_dispatch_advertises_no_retry(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
) -> None:
    session.call_error = RuntimeError("connection reset")

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 502
    assert "Retry-After" not in response.headers


@pytest.mark.parametrize(
    "failure",
    [
        asyncio.CancelledError(),
        ExceptionGroup("unhandled errors in a TaskGroup", [httpx.ConnectError("all attempts failed")]),
    ],
)
def test_a_real_transport_failure_shape_still_gets_the_error_contract(
    client: TestClient,
    servers: _Servers,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    """A dead server must not escape the contract into a bare 500.

    The SDK yields inside an anyio task group, so a closed port surfaces as a
    bare ``CancelledError`` and a shutdown as an ``ExceptionGroup``. Neither is
    the tidy ``Exception`` a substituted session raises, and the earlier version
    of this endpoint only ever saw the tidy one.
    """

    @asynccontextmanager
    async def refuse(*args: Any, **kwargs: Any) -> Any:
        raise failure
        yield  # pragma: no cover - unreachable, keeps this an async generator

    monkeypatch.setattr(mcp_stateless, "open_session", refuse)

    response = client.post(f"{API_ROOT}/mcp/execute", headers=USER_AUTH, json=_body())

    assert response.status_code == 502, response.text
    assert _error(response) == {
        "detail": "MCP server connection failed",
        "code": "mcp_connection_failed",
        "execution_state": "not_started",
    }
