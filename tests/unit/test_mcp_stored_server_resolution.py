"""Hybrid resolution of one stored MCP server (R-RES-1, R-RES-2, R-RES-3).

The resolver answer is the authorization input for both stored-server
endpoints. A current peer can echo the requested id and enabled state; a legacy
peer returns one enabled connection config or an empty list for a disabled
server. Malformed or ambiguous answers still fail closed.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from conftest import InstallControlPlane
from gateway.adapters.mcp_server_adapter import RemoteMcpServers
from gateway.api.routes.mcp import _Principal, _resolve_server
from gateway.exceptions.control_plane_exceptions import ControlPlaneError
from gateway.exceptions.tools_exceptions import McpServerResolutionFailedError
from gateway.ports.mcp_server_port import McpServerScope
from gateway.services.mcp_stateless import (
    CODE_RESOLUTION_FAILED,
    CODE_SERVER_NOT_FOUND,
    ExecutionState,
    McpExecutionError,
)

SERVER_ID = uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e")
OTHER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _scope(user_token: str = "tk_user") -> McpServerScope:
    """The scope a hybrid caller supplies, which carries a token and no workspace."""
    return McpServerScope(workspace_id=None, user_token=user_token)


def _config() -> Any:
    cfg = MagicMock()
    cfg.platform = {"base_url": "https://platform.local", "resolve_timeout_ms": 5000}
    cfg.platform_token = "gw_test_token"
    return cfg


def _entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": str(SERVER_ID),
        "name": "github",
        "url": "https://mcp.example.com/mcp",
        "authorization_token": "server-secret",
        "enabled": True,
        "purpose_hint": "issues",
        "allowed_tools": ["create_issue"],
    }
    entry.update(overrides)
    return entry


def _platform_returns(
    payload: Any,
    control_plane_transport: InstallControlPlane,
    status_code: int = 200,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_post(*, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> Any:
        captured["body"] = body
        captured["url"] = url
        return httpx.Response(status_code, json=payload)

    control_plane_transport(fake_post)
    return captured


@pytest.mark.asyncio
async def test_the_matching_entry_is_resolved(
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    captured = _platform_returns({"servers": [_entry()]}, control_plane_transport)

    server = await RemoteMcpServers(_config()).resolve_one(_scope(), SERVER_ID)
    assert server is not None

    assert server.id == SERVER_ID
    assert server.url == "https://mcp.example.com/mcp"
    assert server.authorization_token == "server-secret"
    assert server.enabled is True
    assert server.allowed_tools == ["create_issue"]
    assert captured["body"] == {"mcp_server_ids": [str(SERVER_ID)]}


@pytest.mark.asyncio
async def test_a_legacy_entry_is_bound_to_the_only_requested_id(
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    entry = _entry()
    del entry["id"]
    del entry["enabled"]
    _platform_returns({"servers": [entry]}, control_plane_transport)

    server = await RemoteMcpServers(_config()).resolve_one(_scope(), SERVER_ID)
    assert server is not None

    assert server.id == SERVER_ID
    assert server.enabled is True


@pytest.mark.asyncio
async def test_a_disabled_server_resolves_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    """The 404 belongs to the route's outcome ladder, which both modes share."""
    _platform_returns({"servers": [_entry(enabled=False)]}, control_plane_transport)

    server = await RemoteMcpServers(_config()).resolve_one(_scope(), SERVER_ID)
    assert server is not None

    assert server.enabled is False


@pytest.mark.asyncio
async def test_a_legacy_empty_answer_is_server_not_found(
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    """Legacy peers omit a disabled server, which has the same public outcome."""
    _platform_returns({"servers": []}, control_plane_transport)

    assert await RemoteMcpServers(_config()).resolve_one(_scope(), SERVER_ID) is None


@pytest.mark.asyncio
async def test_an_absent_allowlist_stays_absent(
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    """``null`` and a missing key both mean "every live tool" (R-RES-4)."""
    entry = _entry()
    del entry["allowed_tools"]
    _platform_returns({"servers": [entry]}, control_plane_transport)

    server = await RemoteMcpServers(_config()).resolve_one(_scope(), SERVER_ID)
    assert server is not None

    assert server.allowed_tools is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"servers": [_entry(), _entry(id=str(OTHER_ID))]},
        {"servers": [_entry(id=str(OTHER_ID))]},
        {"servers": [_entry(id="not-a-uuid")]},
        {"servers": [_entry(url=None)]},
        {"servers": [_entry(enabled="yes")]},
        {"servers": [_entry(allowed_tools="create_issue")]},
        {"servers": "github"},
        {},
        [],
    ],
)
async def test_anything_but_one_matching_entry_is_a_resolution_failure(
    payload: Any,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    _platform_returns(payload, control_plane_transport)

    with pytest.raises(McpServerResolutionFailedError) as raised:
        await RemoteMcpServers(_config()).resolve_one(_scope(), SERVER_ID)

    assert raised.value.status_code == 502


@pytest.mark.asyncio
async def test_the_platforms_own_refusal_is_left_for_the_route_to_classify(
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    _platform_returns({"detail": "no such server"}, control_plane_transport, status_code=404)

    with pytest.raises(ControlPlaneError) as raised:
        await RemoteMcpServers(_config()).resolve_one(_scope(), SERVER_ID)

    assert raised.value.status_code == 404


class _PortAnswering:
    """A port that gives ``resolve_one`` one fixed answer."""

    def __init__(self, answer: Any = None, raises: Exception | None = None) -> None:
        self._answer = answer
        self._raises = raises

    async def resolve_many(self, scope: McpServerScope, server_ids: list[uuid.UUID]) -> Any:
        raise NotImplementedError

    async def resolve_one(self, scope: McpServerScope, server_id: uuid.UUID) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._answer


@pytest.mark.parametrize("workspace_id", [None, uuid.uuid4()], ids=["a peer holds the rows", "this deployment does"])
@pytest.mark.asyncio
async def test_an_id_reaching_nothing_is_the_same_refusal_wherever_the_rows_live(
    workspace_id: uuid.UUID | None,
) -> None:
    """Both deployments refuse an unreachable ID with one code, state and status."""
    principal = _Principal(user_token="tk_user", workspace_id=workspace_id)

    with pytest.raises(McpExecutionError) as raised:
        await _resolve_server(principal, _PortAnswering(answer=None), SERVER_ID)

    assert raised.value.code == CODE_SERVER_NOT_FOUND
    assert raised.value.execution_state is ExecutionState.NOT_STARTED
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_an_unreadable_answer_is_the_resolution_failure_this_contract_publishes() -> None:
    """The port's failure reaches the caller as a 502, not as an unhandled 500."""
    principal = _Principal(user_token="tk_user", workspace_id=None)
    port = _PortAnswering(raises=McpServerResolutionFailedError())

    with pytest.raises(McpExecutionError) as raised:
        await _resolve_server(principal, port, SERVER_ID)

    assert raised.value.code == CODE_RESOLUTION_FAILED
    assert raised.value.execution_state is ExecutionState.NOT_STARTED
    assert raised.value.status_code == 502
