"""Endpoint coverage for stateless MCP tool execution."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import ANY, AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from mcp.types import CallToolResult, TextContent

from gateway import log_config
from gateway.api.deps import reset_config
from gateway.api.routes import mcp as mcp_route
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.services.url_safety import UnsafeURLError

AUTH = {"Authorization": "Bearer sk-test-master"}
PUBLIC_URL = "https://93.184.216.34/mcp"


@pytest.fixture(autouse=True)
def _reset_globals() -> Iterator[None]:
    yield
    reset_config()
    reset_db()


def _standalone_client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            GatewayConfig(
                database_url=f"sqlite:///{tmp_path / 'mcp-execute.db'}",
                master_key="sk-test-master",
            )
        )
    )


def _hybrid_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw-test-token")
    return TestClient(
        create_app(
            GatewayConfig(
                mode="hybrid",
                platform={"base_url": "http://platform.test/api/v1"},
            )
        )
    )


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "server": {
            "name": "github",
            "url": PUBLIC_URL,
            "authorization_token": "server-secret",
            "allowed_tools": ["list_issues"],
        },
        "tool_name": "list_issues",
        "arguments": {"repository": "mozilla-ai/otari"},
    }
    body.update(overrides)
    return body


async def _safe_url(*args: Any, **kwargs: Any) -> None:
    return None


class _FakePool:
    result = CallToolResult(
        content=[TextContent(type="text", text="two open issues")],
        structuredContent={"count": 2},
        isError=False,
    )
    owns = True
    enter_error: Exception | None = None
    call_error: Exception | None = None
    exit_error: Exception | None = None
    instances: list[_FakePool] = []

    def __init__(self, configs: list[Any]) -> None:
        self.configs = configs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        type(self).instances.append(self)

    async def __aenter__(self) -> _FakePool:
        if self.enter_error is not None:
            raise self.enter_error
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self.exit_error is not None:
            raise self.exit_error

    def owns_tool(self, name: str) -> bool:
        return self.owns

    async def call_tool_result(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        if self.call_error is not None:
            raise self.call_error
        return self.result


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> type[_FakePool]:
    _FakePool.instances = []
    _FakePool.owns = True
    _FakePool.enter_error = None
    _FakePool.call_error = None
    _FakePool.exit_error = None
    _FakePool.result = CallToolResult(
        content=[TextContent(type="text", text="two open issues")],
        structuredContent={"count": 2},
        isError=False,
    )
    monkeypatch.setattr(mcp_route, "MCPClientPool", _FakePool)
    monkeypatch.setattr(mcp_route, "validate_mcp_url", _safe_url)
    return _FakePool


def test_executes_exact_approved_call_and_returns_native_result(
    tmp_path: Path,
    fake_pool: type[_FakePool],
) -> None:
    with _standalone_client(tmp_path) as client:
        response = client.post("/v1/mcp/execute", headers=AUTH, json=_body())

    assert response.status_code == 200, response.text
    assert response.json() == {
        "content": [{"type": "text", "text": "two open issues"}],
        "structuredContent": {"count": 2},
        "isError": False,
    }
    pool = fake_pool.instances[0]
    assert len(pool.configs) == 1
    assert pool.configs[0].name == "github"
    assert pool.configs[0].authorization_token == "server-secret"
    assert pool.calls == [("list_issues", {"repository": "mozilla-ai/otari"})]


def test_server_reported_error_remains_a_typed_success(
    tmp_path: Path,
    fake_pool: type[_FakePool],
) -> None:
    fake_pool.result = CallToolResult(
        content=[TextContent(type="text", text="permission denied")],
        isError=True,
    )
    with _standalone_client(tmp_path) as client:
        response = client.post("/v1/mcp/execute", headers=AUTH, json=_body())

    assert response.status_code == 200, response.text
    assert response.json()["isError"] is True
    assert response.json()["content"] == [{"type": "text", "text": "permission denied"}]


def test_cleanup_failure_does_not_hide_a_successful_mutating_call(
    tmp_path: Path,
    fake_pool: type[_FakePool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = Mock()
    monkeypatch.setattr(log_config.logger, "warning", warning)
    fake_pool.exit_error = RuntimeError("server-secret cleanup failure")

    with _standalone_client(tmp_path) as client:
        warning.reset_mock()
        response = client.post("/v1/mcp/execute", headers=AUTH, json=_body())

    assert response.status_code == 200, response.text
    assert response.json()["content"] == [{"type": "text", "text": "two open issues"}]
    assert fake_pool.instances[0].calls == [
        ("list_issues", {"repository": "mozilla-ai/otari"})
    ]
    warning.assert_called_once_with(
        "Stateless MCP cleanup failed error_class=%s",
        "RuntimeError",
    )


def test_requires_authentication_before_contacting_server(
    tmp_path: Path,
    fake_pool: type[_FakePool],
) -> None:
    with _standalone_client(tmp_path) as client:
        response = client.post("/v1/mcp/execute", json=_body())

    assert response.status_code == 401
    assert fake_pool.instances == []


@pytest.mark.parametrize("allowed_tools", [[], ["read_issue"]])
def test_configured_allowlist_is_enforced_before_contacting_server(
    tmp_path: Path,
    fake_pool: type[_FakePool],
    allowed_tools: list[str],
) -> None:
    server = _body()["server"]
    server["allowed_tools"] = allowed_tools
    with _standalone_client(tmp_path) as client:
        response = client.post("/v1/mcp/execute", headers=AUTH, json=_body(server=server))

    assert response.status_code == 403, response.text
    assert response.json() == {"detail": mcp_route.MCP_TOOL_NOT_ALLOWED_DETAIL}
    assert fake_pool.instances == []


def test_unsafe_url_is_rejected_before_contacting_server(
    tmp_path: Path,
    fake_pool: type[_FakePool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject(*args: Any, **kwargs: Any) -> None:
        raise UnsafeURLError("MCP server URL is unsafe")

    monkeypatch.setattr(mcp_route, "validate_mcp_url", reject)
    with _standalone_client(tmp_path) as client:
        response = client.post("/v1/mcp/execute", headers=AUTH, json=_body())

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "MCP server URL is unsafe"}
    assert fake_pool.instances == []


def test_tool_must_exist_in_live_discovery(
    tmp_path: Path,
    fake_pool: type[_FakePool],
) -> None:
    fake_pool.owns = False
    with _standalone_client(tmp_path) as client:
        response = client.post("/v1/mcp/execute", headers=AUTH, json=_body())

    assert response.status_code == 404, response.text
    assert response.json() == {"detail": mcp_route.MCP_TOOL_NOT_FOUND_DETAIL}
    assert fake_pool.instances[0].calls == []


@pytest.mark.parametrize("stage", ["connect", "call"])
def test_transport_and_protocol_failures_are_sanitized(
    tmp_path: Path,
    fake_pool: type[_FakePool],
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    warning = Mock()
    monkeypatch.setattr(log_config.logger, "warning", warning)
    private_message = "server-secret repository=mozilla-ai/otari private-result"
    if stage == "connect":
        fake_pool.enter_error = RuntimeError(private_message)
    else:
        fake_pool.call_error = RuntimeError(private_message)

    with _standalone_client(tmp_path) as client:
        # App startup uses the same process-wide logger for configuration
        # warnings. Isolate the request under test from those lifespan calls.
        warning.reset_mock()
        response = client.post("/v1/mcp/execute", headers=AUTH, json=_body())

    assert response.status_code == 502, response.text
    assert response.json() == {"detail": mcp_route.MCP_EXECUTION_FAILED_DETAIL}
    assert private_message not in response.text
    warning.assert_called_once_with(
        "Stateless MCP request failed error_class=%s",
        "RuntimeError",
    )


def test_arguments_must_be_a_json_object(tmp_path: Path, fake_pool: type[_FakePool]) -> None:
    with _standalone_client(tmp_path) as client:
        response = client.post(
            "/v1/mcp/execute",
            headers=AUTH,
            json=_body(arguments=["not", "an", "object"]),
        )

    assert response.status_code == 422
    assert fake_pool.instances == []


def test_hybrid_mode_authenticates_through_empty_mcp_resolution(
    monkeypatch: pytest.MonkeyPatch,
    fake_pool: type[_FakePool],
) -> None:
    resolve = AsyncMock(return_value=[])
    monkeypatch.setattr(mcp_route, "_resolve_platform_mcp_servers", resolve)
    with _hybrid_client(monkeypatch) as client:
        response = client.post(
            "/v1/mcp/execute",
            headers={"Authorization": "Bearer platform-user-token"},
            json=_body(),
        )

    assert response.status_code == 200, response.text
    resolve.assert_awaited_once_with(ANY, "platform-user-token", [])


def test_hybrid_mode_requires_user_token(
    monkeypatch: pytest.MonkeyPatch,
    fake_pool: type[_FakePool],
) -> None:
    resolve = AsyncMock(return_value=[])
    monkeypatch.setattr(mcp_route, "_resolve_platform_mcp_servers", resolve)
    with _hybrid_client(monkeypatch) as client:
        response = client.post("/v1/mcp/execute", json=_body())

    assert response.status_code == 401
    resolve.assert_not_awaited()
    assert fake_pool.instances == []
