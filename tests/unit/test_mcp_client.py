"""Unit coverage for MCPClientPool behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

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
    session.call_tool.assert_awaited_once_with("lookup", {"id": 755})


@pytest.mark.asyncio
async def test_call_tool_result_returns_native_mcp_result() -> None:
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

    returned = await pool.call_tool_result("lookup", {"id": 791})

    assert returned is result
    assert returned.structuredContent == {"issue": 791}
    assert returned.isError is True
    session.call_tool.assert_awaited_once_with("lookup", {"id": 791})
