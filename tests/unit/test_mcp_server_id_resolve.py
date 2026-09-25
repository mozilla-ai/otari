"""Resolving workspace-scoped MCP server ids against a peer control plane."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from conftest import InstallControlPlane
from gateway.adapters.mcp_server_adapter import RemoteMcpServers
from gateway.exceptions.control_plane_exceptions import ControlPlaneError, ControlPlaneRefusedError
from gateway.exceptions.tools_exceptions import McpServerResolutionFailedError
from gateway.models.mcp import MAX_MCP_SERVER_IDS, McpServerConfig
from gateway.ports.mcp_server_port import McpServerScope
from gateway.services.tenancy.workspace_mcp_server_service import MAX_MCP_SERVERS_PER_WORKSPACE


def _scope(user_token: str = "tk_user") -> McpServerScope:
    """The scope a hybrid caller supplies, which carries a token and no workspace."""
    return McpServerScope(workspace_id=None, user_token=user_token)


def _config(*, base_url: str | None = "https://platform.local") -> Any:
    cfg = MagicMock()
    cfg.platform = {"base_url": base_url, "resolve_timeout_ms": 5000} if base_url else {}
    cfg.platform_token = "gw_test_token"
    return cfg


def _ok_response(servers: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"servers": servers})


@pytest.mark.asyncio
async def test_repeated_ids_are_sent_once_in_request_order(
    control_plane_transport: InstallControlPlane,
) -> None:
    """Hybrid de-duplicates ids like the standalone path, so a repeat cannot resolve twice.

    Two resolved servers sharing a name are a 500 in `prepare_gateway_tools`,
    and the protocol does not say what the platform answers for a repeated id,
    so a caller naming one twice must not be able to trip that (otari#792
    review).
    """
    captured: dict[str, Any] = {}

    async def fake_post(
        *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        captured["body"] = body
        return _ok_response([])

    control_plane_transport(fake_post)

    first = uuid.UUID("11111111-1111-1111-1111-111111111111")
    second = uuid.UUID("22222222-2222-2222-2222-222222222222")
    await RemoteMcpServers(_config()).resolve_many(_scope(), [first, second, first])

    assert captured["body"]["mcp_server_ids"] == [str(first), str(second)]


@pytest.mark.asyncio
async def test_resolve_returns_configs(
    control_plane_transport: InstallControlPlane,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(
        *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return _ok_response(
            [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "name": "calendar",
                    "url": "https://example.com/mcp",
                    "authorization_token": "ya29.x",
                    "purpose_hint": "scheduling",
                    "allowed_tools": ["list_events"],
                },
            ]
        )

    control_plane_transport(fake_post)

    ids = [uuid.UUID("11111111-1111-1111-1111-111111111111")]
    out = await RemoteMcpServers(_config()).resolve_many(_scope(), ids)

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
async def test_resolve_empty_servers_returns_empty(
    control_plane_transport: InstallControlPlane,
) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return _ok_response([])

    control_plane_transport(fake_post)
    out = await RemoteMcpServers(_config()).resolve_many(_scope("tk"), [uuid.uuid4()])
    assert out == []


@pytest.mark.asyncio
async def test_resolve_404_passes_through(
    control_plane_transport: InstallControlPlane,
) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(404, json={"detail": "MCPServer not found"})

    control_plane_transport(fake_post)

    with pytest.raises(ControlPlaneError) as ei:
        await RemoteMcpServers(_config()).resolve_many(_scope("tk"), [uuid.uuid4()])
    assert ei.value.status_code == 404
    assert ei.value.message == "MCPServer not found"


@pytest.mark.asyncio
async def test_resolve_5xx_maps_to_502(
    control_plane_transport: InstallControlPlane,
) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(503, text="busy")

    control_plane_transport(fake_post)

    with pytest.raises(ControlPlaneError) as ei:
        await RemoteMcpServers(_config()).resolve_many(_scope("tk"), [uuid.uuid4()])
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_network_error_maps_to_502(
    control_plane_transport: InstallControlPlane,
) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        raise httpx.NetworkError("connection refused")

    control_plane_transport(fake_post)

    with pytest.raises(ControlPlaneError) as ei:
        await RemoteMcpServers(_config()).resolve_many(_scope("tk"), [uuid.uuid4()])
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_misconfigured_platform_500() -> None:
    with pytest.raises(ControlPlaneError) as ei:
        await RemoteMcpServers(_config(base_url=None)).resolve_many(_scope("tk"), [uuid.uuid4()])
    assert ei.value.status_code == 500


@pytest.mark.asyncio
async def test_resolve_429_passthrough_with_retry_after(
    control_plane_transport: InstallControlPlane,
) -> None:
    """A platform 429 should forward verbatim (status + Retry-After header)
    so clients can back off correctly, matching `_resolve_platform_credentials`."""

    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(429, json={"detail": "slow down"}, headers={"Retry-After": "30"})

    control_plane_transport(fake_post)

    with pytest.raises(ControlPlaneRefusedError) as ei:
        await RemoteMcpServers(_config()).resolve_many(_scope("tk"), [uuid.uuid4()])
    assert ei.value.status_code == 429
    assert ei.value.retry_after == "30"
    assert ei.value.message == "slow down"


@pytest.mark.asyncio
async def test_resolve_402_passthrough(
    control_plane_transport: InstallControlPlane,
) -> None:
    """402 (payment required / quota) should forward verbatim, matching
    `_resolve_platform_credentials`'s behavior for the same code."""

    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(402, json={"detail": "quota exhausted"})

    control_plane_transport(fake_post)

    with pytest.raises(ControlPlaneError) as ei:
        await RemoteMcpServers(_config()).resolve_many(_scope("tk"), [uuid.uuid4()])
    assert ei.value.status_code == 402
    assert ei.value.message == "quota exhausted"


@pytest.mark.asyncio
async def test_resolve_422_collapses_to_502(
    control_plane_transport: InstallControlPlane,
) -> None:
    """422 from the platform indicates a gateway↔platform schema mismatch,
    not something the caller can usefully act on — collapse to 502 like
    `_resolve_platform_credentials` does."""

    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(422, json={"detail": "schema mismatch"})

    control_plane_transport(fake_post)

    with pytest.raises(ControlPlaneError) as ei:
        await RemoteMcpServers(_config()).resolve_many(_scope("tk"), [uuid.uuid4()])
    assert ei.value.status_code == 502


def test_the_request_bound_admits_every_server_a_workspace_can_hold() -> None:
    """`MAX_MCP_SERVER_IDS` justifies itself by matching the per-workspace cap, so pin that.

    They are declared separately on purpose (a wire bound has to be checkable at
    parse time, with no database and no mode to consult), which is exactly what
    lets them drift. Raising the service cap alone would leave a workspace able
    to store more servers than one request may name, surfacing as a 422 with
    nothing pointing at the cause.
    """
    assert MAX_MCP_SERVER_IDS >= MAX_MCP_SERVERS_PER_WORKSPACE


@pytest.mark.asyncio
async def test_an_answer_omitting_servers_resolves_to_none(
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    """A peer that names no servers at all is served, not refused."""

    async def fake_post(*, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> Any:
        return httpx.Response(200, json={})

    control_plane_transport(fake_post)

    out = await RemoteMcpServers(_config()).resolve_many(
        _scope("tk"), [uuid.uuid4()]
    )

    assert out == []


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param([], id="the answer is not an object"),
        pytest.param({"servers": None}, id="servers is null"),
        pytest.param({"servers": {"a": 1}}, id="servers is an object"),
        pytest.param({"servers": "none"}, id="servers is a string"),
    ],
)
@pytest.mark.asyncio
async def test_an_unreadable_answer_is_a_resolution_failure(
    payload: Any,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    """An unreadable answer never resolves to no servers.

    Resolving it to none would serve a request that named stored servers without any of them, and bill it.
    """

    async def fake_post(*, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> Any:
        return httpx.Response(200, json=payload)

    control_plane_transport(fake_post)

    with pytest.raises(McpServerResolutionFailedError):
        await RemoteMcpServers(_config()).resolve_many(
            _scope("tk"), [uuid.uuid4()]
        )
