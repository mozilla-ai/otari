"""Unit tests for resolving the workspace memory policy via the platform, plus the
gateway-managed memory tool-type extraction."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import HTTPException

from gateway.api.routes import _platform as platform_module
from gateway.api.routes._platform import _resolve_platform_memory
from gateway.api.routes._tools import Tool, _extract_memory_tool


def _config(*, base_url: str | None = "https://platform.local") -> Any:
    cfg = MagicMock()
    cfg.platform = {"base_url": base_url, "resolve_timeout_ms": 5000} if base_url else {}
    cfg.platform_token = "gw_test_token"
    return cfg


@pytest.mark.asyncio
async def test_resolve_returns_policy_and_dual_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(
        *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        return httpx.Response(200, json={"enabled": True, "scope": "member", "recall_limit": 5})

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)

    out = await _resolve_platform_memory(_config(), "tk_user")

    assert out["enabled"] is True
    assert out["scope"] == "member"
    assert captured["url"].endswith("/gateway/memory/resolve")
    assert captured["headers"]["X-Gateway-Token"] == "gw_test_token"
    assert captured["headers"]["X-User-Token"] == "tk_user"


@pytest.mark.asyncio
async def test_resolve_403_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(403, json={"detail": "memory disabled"})

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_memory(_config(), "tk")
    assert ei.value.status_code == 403
    assert ei.value.detail == "memory disabled"


@pytest.mark.asyncio
async def test_resolve_5xx_maps_to_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(503, text="busy")

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_memory(_config(), "tk")
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_network_error_maps_to_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        raise httpx.NetworkError("connection refused")

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_memory(_config(), "tk")
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_misconfigured_platform_500() -> None:
    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_memory(_config(base_url=None), "tk")
    assert ei.value.status_code == 500


def test_extract_memory_tool_pulls_only_otari_memory() -> None:
    tools: list[dict[str, Any]] = [
        {"type": "function", "function": {"name": "user_tool"}},
        {"type": Tool.MEMORY},
        {"type": "web_search"},  # provider-native passthrough, must stay
    ]
    entry, remaining = _extract_memory_tool(tools)
    assert entry == {"type": Tool.MEMORY}
    assert remaining == [
        {"type": "function", "function": {"name": "user_tool"}},
        {"type": "web_search"},
    ]


def test_extract_memory_tool_absent() -> None:
    entry, remaining = _extract_memory_tool([{"type": "web_search"}])
    assert entry is None
    assert remaining == [{"type": "web_search"}]
