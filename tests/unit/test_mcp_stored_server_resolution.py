"""Hybrid resolution of one stored MCP server (R-RES-1, R-RES-2, R-RES-3).

The resolver answer is the authorization input for both stored-server
endpoints, so it is parsed strictly: exactly one entry, whose id is the id that
was asked for. Anything else is a resolution failure rather than a best guess,
because the alternative is executing a caller-authorized call against a server
the caller did not name.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import HTTPException

from gateway.api.routes import _platform as platform_module
from gateway.api.routes._platform import _resolve_platform_mcp_server
from gateway.services.mcp_stateless import ExecutionState, McpExecutionError

SERVER_ID = uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e")
OTHER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


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


def _platform_returns(payload: Any, monkeypatch: pytest.MonkeyPatch, status_code: int = 200) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_post(*, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> Any:
        captured["body"] = body
        captured["url"] = url
        return httpx.Response(status_code, json=payload)

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    return captured


@pytest.mark.asyncio
async def test_the_matching_entry_is_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _platform_returns({"servers": [_entry()]}, monkeypatch)

    server = await _resolve_platform_mcp_server(_config(), "tk_user", SERVER_ID)

    assert server.id == SERVER_ID
    assert server.url == "https://mcp.example.com/mcp"
    assert server.authorization_token == "server-secret"
    assert server.enabled is True
    assert server.allowed_tools == ["create_issue"]
    assert captured["body"] == {"mcp_server_ids": [str(SERVER_ID)]}


@pytest.mark.asyncio
async def test_a_disabled_server_resolves_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 404 belongs to the route's outcome ladder, which both modes share."""
    _platform_returns({"servers": [_entry(enabled=False)]}, monkeypatch)

    server = await _resolve_platform_mcp_server(_config(), "tk_user", SERVER_ID)

    assert server.enabled is False


@pytest.mark.asyncio
async def test_an_absent_allowlist_stays_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """``null`` and a missing key both mean "every live tool" (R-RES-4)."""
    entry = _entry()
    del entry["allowed_tools"]
    _platform_returns({"servers": [entry]}, monkeypatch)

    server = await _resolve_platform_mcp_server(_config(), "tk_user", SERVER_ID)

    assert server.allowed_tools is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"servers": []},
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
) -> None:
    _platform_returns(payload, monkeypatch)

    with pytest.raises(McpExecutionError) as raised:
        await _resolve_platform_mcp_server(_config(), "tk_user", SERVER_ID)

    assert raised.value.code == "mcp_resolution_failed"
    assert raised.value.execution_state is ExecutionState.NOT_STARTED
    assert raised.value.status_code == 502


@pytest.mark.asyncio
async def test_the_platforms_own_refusal_is_left_for_the_route_to_classify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _platform_returns({"detail": "no such server"}, monkeypatch, status_code=404)

    with pytest.raises(HTTPException) as raised:
        await _resolve_platform_mcp_server(_config(), "tk_user", SERVER_ID)

    assert raised.value.status_code == 404
