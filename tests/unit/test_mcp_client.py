"""Unit coverage for MCPClientPool behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

from gateway.models.mcp import McpServerConfig
from gateway.services.mcp_client import MCPClientPool, _ConnectedServer


@pytest.mark.asyncio
async def test_aenter_rejects_duplicate_server_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two configs sharing a name raise instead of the second silently overwriting the first."""

    async def fake_connect(self: MCPClientPool, cfg: McpServerConfig) -> _ConnectedServer:
        return _ConnectedServer(name=cfg.name, session=object())  # type: ignore[arg-type]

    monkeypatch.setattr(MCPClientPool, "_connect", fake_connect)

    configs = [
        McpServerConfig(name="tools", url="https://93.184.216.34/mcp"),
        McpServerConfig(name="tools", url="https://93.184.216.35/mcp"),
    ]
    pool = MCPClientPool(configs)
    with pytest.raises(ValueError, match="tools"):
        await pool.__aenter__()


@pytest.mark.asyncio
async def test_aenter_connects_distinct_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct names still connect normally; the guard only fires on a repeat."""

    connected: list[str] = []

    async def fake_connect(self: MCPClientPool, cfg: McpServerConfig) -> _ConnectedServer:
        connected.append(cfg.name)
        return _ConnectedServer(name=cfg.name, session=object())  # type: ignore[arg-type]

    monkeypatch.setattr(MCPClientPool, "_connect", fake_connect)

    configs = [
        McpServerConfig(name="a", url="https://93.184.216.34/mcp"),
        McpServerConfig(name="b", url="https://93.184.216.35/mcp"),
    ]
    async with MCPClientPool(configs) as pool:
        assert set(pool._servers) == {"a", "b"}
    assert connected == ["a", "b"]


@pytest.mark.asyncio
async def test_empty_allowed_tools_exposes_no_discovered_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit empty allowlist denies every tool; only None means unrestricted."""

    @asynccontextmanager
    async def fake_transport(*args: Any, **kwargs: Any) -> AsyncIterator[tuple[object, object, object]]:
        yield object(), object(), object()

    class FakeSession:
        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def initialize(self) -> None:
            return None

        async def list_tools(self) -> SimpleNamespace:
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="list_issues",
                        description="List issues",
                        inputSchema={"type": "object"},
                    )
                ]
            )

    monkeypatch.setattr("gateway.services.mcp_client.streamablehttp_client", fake_transport)
    monkeypatch.setattr("gateway.services.mcp_client.ClientSession", lambda *args: FakeSession())

    config = McpServerConfig(
        name="tools",
        url="https://93.184.216.34/mcp",
        allowed_tools=[],
    )
    async with MCPClientPool([config]) as pool:
        assert pool.openai_tools == []
        assert pool.owns_tool("list_issues") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("is_error", [False, True])
async def test_call_tool_outcome_preserves_server_error_status(is_error: bool) -> None:
    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="fixture result")],
                isError=is_error,
            )
        )
    )
    pool = MCPClientPool([])
    pool._servers["fixture"] = _ConnectedServer(
        name="fixture",
        session=cast(Any, session),
    )
    pool._tool_owner["lookup"] = "fixture"

    outcome = await pool.call_tool_outcome("lookup", {"id": 755})

    assert pool.server_name_for_tool("lookup") == "fixture"
    assert outcome.is_error is is_error
    assert outcome.content == ("[tool error] fixture result" if is_error else "fixture result")
    assert outcome.activity_content == "fixture result"
    assert outcome.transport_error is False
    session.call_tool.assert_awaited_once_with("lookup", {"id": 755})


@pytest.mark.asyncio
async def test_call_tool_sanitizes_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SimpleNamespace(
        call_tool=AsyncMock(side_effect=RuntimeError("https://internal.test/?token=secret"))
    )
    pool = MCPClientPool([])
    pool._servers["fixture"] = _ConnectedServer(
        name="fixture",
        session=cast(Any, session),
    )
    pool._tool_owner["lookup"] = "fixture"
    warnings: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "gateway.services.mcp_client.logger.warning",
        lambda message, *args: warnings.append((message, *args)),
    )

    content = await pool.call_tool("lookup", {"id": 755})

    assert content == "[tool error] MCP tool execution failed"
    assert warnings == [("MCP tool %s execution failed: %s", "lookup", "RuntimeError")]
    assert "internal.test" not in str(warnings)
    assert "secret" not in str(warnings)


@pytest.mark.asyncio
async def test_call_tool_result_extraction_returns_native_mcp_result() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="fixture result")],
        structuredContent={"issue": 791},
        isError=True,
    )
    session = SimpleNamespace(call_tool=AsyncMock(return_value=result))
    pool = MCPClientPool([])
    pool._servers["fixture"] = _ConnectedServer(
        name="fixture",
        session=cast(Any, session),
    )
    pool._tool_owner["lookup"] = "fixture"

    returned = await pool._call_tool_result("lookup", {"id": 791})  # noqa: SLF001

    assert returned is result
    assert returned.structuredContent == {"issue": 791}
    assert returned.isError is True
    session.call_tool.assert_awaited_once_with("lookup", {"id": 791})


# --------------------------------------------------------------------------- #
# Transport safety
# --------------------------------------------------------------------------- #


class _FakeSession:
    """Enough of a ``ClientSession`` for ``_connect`` to finish."""

    def __init__(self, *args: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(tools=[])


def _capture_transport(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Substitute the MCP transport and record the kwargs the pool opens it with."""
    captured: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_transport(url: str, **kwargs: Any) -> Any:
        captured["url"] = url
        captured.update(kwargs)
        yield (None, None, None)

    monkeypatch.setattr("gateway.services.mcp_client.streamablehttp_client", fake_transport)
    monkeypatch.setattr("gateway.services.mcp_client.ClientSession", _FakeSession)
    return captured


@pytest.mark.asyncio
async def test_the_transport_keeps_the_sdk_timeout_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_transport(monkeypatch)
    config = McpServerConfig(name="tools", url="https://93.184.216.34/mcp")

    async with MCPClientPool([config]):
        pass

    factory = captured["httpx_client_factory"]
    default_client = factory(None, None, None)
    transport_timeout = httpx.Timeout(30.0, read=300.0)
    configured_client = factory(None, transport_timeout, None)
    try:
        assert default_client.timeout == httpx.Timeout(30.0)
        assert configured_client.timeout == transport_timeout
    finally:
        await default_client.aclose()
        await configured_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "location"),
    [
        ("https://93.184.216.34/mcp", "/mcp/"),
        ("http://93.184.216.34/mcp", "https://93.184.216.34/mcp/"),
    ],
)
async def test_safe_redirect_is_followed(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    location: str,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(307, headers={"location": location})
        return httpx.Response(200)

    captured = _capture_transport(monkeypatch)
    config = McpServerConfig(name="tools", url=base_url)

    async with MCPClientPool([config]):
        pass

    client = captured["httpx_client_factory"](None, None, None)
    client._transport = httpx.MockTransport(handler)  # noqa: SLF001
    async with client:
        response = await client.post(base_url, json={"method": "tools/list"})

    assert response.status_code == 200
    assert len(seen) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [
        "http://169.254.169.254/",
        "http://93.184.216.34/mcp",
        "https://93.184.216.34:8443/mcp",
    ],
)
async def test_unsafe_redirect_is_blocked_before_sending(
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(307, headers={"location": location})

    base_url = "https://93.184.216.34/mcp"
    captured = _capture_transport(monkeypatch)
    config = McpServerConfig(name="tools", url=base_url, authorization_token="ghp_token")

    async with MCPClientPool([config]):
        pass

    client = captured["httpx_client_factory"]({"Authorization": "Bearer ghp_token"}, None, None)
    client._transport = httpx.MockTransport(handler)  # noqa: SLF001
    async with client:
        with pytest.raises(httpx.RequestError, match="outside the validated origin"):
            await client.post(base_url, json={"method": "tools/list"})

    assert len(seen) == 1
    assert seen[0].url.host == "93.184.216.34"
