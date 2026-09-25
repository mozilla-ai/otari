"""Unit tests for resolving the workspace code-execution policy via the platform."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from conftest import InstallControlPlane
from gateway.api.routes._platform import _resolve_platform_code_execution


def _config(*, base_url: str | None = "https://platform.local") -> Any:
    cfg = MagicMock()
    cfg.platform = {"base_url": base_url, "resolve_timeout_ms": 5000} if base_url else {}
    cfg.platform_token = "gw_test_token"
    return cfg


@pytest.mark.asyncio
async def test_resolve_returns_policy_dict(
    control_plane_transport: InstallControlPlane,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(
        *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "enabled": True,
                "tools": ["code_execution"],
                "default_purpose_hint": "prefer running code",
                "max_iterations": 5,
                "exec_timeout_s": 30,
            },
        )

    control_plane_transport(fake_post)

    out = await _resolve_platform_code_execution(_config(), "tk_user")

    assert out["enabled"] is True
    assert out["tools"] == ["code_execution"]
    assert out["default_purpose_hint"] == "prefer running code"
    assert out["max_iterations"] == 5

    assert captured["url"].endswith("/gateway/code-execution/resolve")
    assert captured["headers"]["X-Gateway-Token"] == "gw_test_token"
    assert captured["headers"]["X-User-Token"] == "tk_user"
    assert captured["body"] == {}


@pytest.mark.asyncio
async def test_resolve_403_passes_through(
    control_plane_transport: InstallControlPlane,
) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(403, json={"detail": "code execution disabled"})

    control_plane_transport(fake_post)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_code_execution(_config(), "tk")
    assert ei.value.status_code == 403
    assert ei.value.detail == "code execution disabled"


@pytest.mark.asyncio
async def test_resolve_429_passthrough_with_retry_after(
    control_plane_transport: InstallControlPlane,
) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(429, json={"detail": "slow down"}, headers={"Retry-After": "30"})

    control_plane_transport(fake_post)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_code_execution(_config(), "tk")
    assert ei.value.status_code == 429
    assert ei.value.headers == {"Retry-After": "30"}
    assert ei.value.detail == "slow down"


@pytest.mark.asyncio
async def test_resolve_5xx_maps_to_502(
    control_plane_transport: InstallControlPlane,
) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(503, text="busy")

    control_plane_transport(fake_post)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_code_execution(_config(), "tk")
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_422_collapses_to_502(
    control_plane_transport: InstallControlPlane,
) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(422, json={"detail": "schema mismatch"})

    control_plane_transport(fake_post)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_code_execution(_config(), "tk")
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_network_error_maps_to_502(
    control_plane_transport: InstallControlPlane,
) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        raise httpx.NetworkError("connection refused")

    control_plane_transport(fake_post)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_code_execution(_config(), "tk")
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_misconfigured_platform_500() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await _resolve_platform_code_execution(_config(base_url=None), "tk")
    assert ei.value.status_code == 500
