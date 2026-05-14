"""Unit tests for resolving workspace-scoped MCP server ids via the platform service."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from gateway.api.routes import chat as chat_module
from gateway.api.routes.chat import _resolve_platform_mcp_servers
from gateway.models.mcp import McpServerConfig


def _config(*, base_url: str | None = "https://platform.local") -> Any:
    cfg = MagicMock()
    cfg.platform = {"base_url": base_url, "resolve_timeout_ms": 5000} if base_url else {}
    cfg.platform_token = "gw_test_token"
    return cfg


def _ok_response(servers: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"servers": servers})


@pytest.mark.asyncio
async def test_resolve_returns_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(*, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return _ok_response(
            [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "name": "calendar",
                    "url": "https://cal.example/mcp",
                    "authorization_token": "ya29.x",
                    "purpose_hint": "scheduling",
                    "allowed_tools": ["list_events"],
                },
            ]
        )

    monkeypatch.setattr(chat_module, "_post_platform", fake_post)

    ids = [uuid.UUID("11111111-1111-1111-1111-111111111111")]
    out = await _resolve_platform_mcp_servers(_config(), "tk_user", ids)

    assert isinstance(out[0], McpServerConfig)
    assert out[0].name == "calendar"
    assert out[0].authorization_token == "ya29.x"
    assert out[0].purpose_hint == "scheduling"
    assert out[0].allowed_tools == ["list_events"]

    assert captured["url"].endswith("/gateway/mcp-servers/resolve")
    assert captured["headers"]["X-Gateway-Token"] == "gw_test_token"
    assert captured["headers"]["X-User-Token"] == "tk_user"
    assert captured["body"] == {"mcp_server_ids": ["11111111-1111-1111-1111-111111111111"]}


@pytest.mark.asyncio
async def test_resolve_empty_servers_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return _ok_response([])

    monkeypatch.setattr(chat_module, "_post_platform", fake_post)
    out = await _resolve_platform_mcp_servers(_config(), "tk", [uuid.uuid4()])
    assert out == []


@pytest.mark.asyncio
async def test_resolve_404_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(404, json={"detail": "MCPServer not found"})

    monkeypatch.setattr(chat_module, "_post_platform", fake_post)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_mcp_servers(_config(), "tk", [uuid.uuid4()])
    assert ei.value.status_code == 404
    assert ei.value.detail == "MCPServer not found"


@pytest.mark.asyncio
async def test_resolve_5xx_maps_to_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(503, text="busy")

    monkeypatch.setattr(chat_module, "_post_platform", fake_post)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_mcp_servers(_config(), "tk", [uuid.uuid4()])
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_network_error_maps_to_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        raise httpx.NetworkError("connection refused")

    monkeypatch.setattr(chat_module, "_post_platform", fake_post)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_mcp_servers(_config(), "tk", [uuid.uuid4()])
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_misconfigured_platform_500() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_mcp_servers(_config(base_url=None), "tk", [uuid.uuid4()])
    assert ei.value.status_code == 500
