"""Bounded ``tools/list`` collection (R-DISC-1, R-DISC-4, L-DISC-* ceilings).

The live catalog of a remote MCP server is untrusted and paginated. Discovery
follows the cursor, stops at the first ceiling it reaches, and either returns
the complete authorized catalog or nothing: a partial page would be read by
an application as the whole authorization and risk-policy input (R-DISC-5).
"""

from __future__ import annotations

from typing import Any

import pytest
from mcp.types import ListToolsResult
from mcp.types import Tool as MCPTool

from gateway.services.mcp_stateless import (
    DISCOVERY_MAX_EXAMINED,
    DISCOVERY_MAX_PAGES,
    McpDiscoveryRefused,
    collect_tools,
)


def _tool(name: str, description: str = "a tool") -> MCPTool:
    return MCPTool(name=name, description=description, inputSchema={"type": "object"})


class _FakeSession:
    """A ``ClientSession`` stand-in that serves prepared ``tools/list`` pages."""

    def __init__(self, pages: list[Any]) -> None:
        self._pages = pages
        self.cursors: list[str | None] = []

    async def list_tools(self, cursor: str | None = None) -> Any:
        self.cursors.append(cursor)
        return self._pages[len(self.cursors) - 1]


def _page(tools: list[MCPTool], next_cursor: str | None = None) -> ListToolsResult:
    return ListToolsResult(tools=tools, nextCursor=next_cursor)


@pytest.mark.asyncio
async def test_a_single_page_catalog_is_returned_whole() -> None:
    session = _FakeSession([_page([_tool("a"), _tool("b")])])

    tools = await collect_tools(session, allowed_tools=None)

    assert [t.name for t in tools] == ["a", "b"]
    assert session.cursors == [None]


@pytest.mark.asyncio
async def test_the_cursor_is_followed_to_the_end() -> None:
    session = _FakeSession(
        [
            _page([_tool("a")], next_cursor="p2"),
            _page([_tool("b")], next_cursor="p3"),
            _page([_tool("c")]),
        ]
    )

    tools = await collect_tools(session, allowed_tools=None)

    assert [t.name for t in tools] == ["a", "b", "c"]
    assert session.cursors == [None, "p2", "p3"]


@pytest.mark.asyncio
async def test_only_the_allowlist_intersection_is_returned() -> None:
    """A tool the server added is not exposed merely because it was listed."""
    session = _FakeSession([_page([_tool("a"), _tool("newly_added"), _tool("b")])])

    tools = await collect_tools(session, allowed_tools=["b", "a", "never_listed"])

    assert [t.name for t in tools] == ["a", "b"]


@pytest.mark.asyncio
async def test_pagination_stops_once_every_allowlisted_name_is_found() -> None:
    session = _FakeSession(
        [
            _page([_tool("a")], next_cursor="p2"),
            _page([_tool("b")], next_cursor="p3"),
            _page([_tool("c")]),
        ]
    )

    tools = await collect_tools(session, allowed_tools=["a", "b"])

    assert [t.name for t in tools] == ["a", "b"]
    assert session.cursors == [None, "p2"]


@pytest.mark.asyncio
async def test_a_repeated_cursor_fails_the_whole_request() -> None:
    session = _FakeSession(
        [
            _page([_tool("a")], next_cursor="loop"),
            _page([_tool("b")], next_cursor="loop"),
        ]
    )

    with pytest.raises(McpDiscoveryRefused):
        await collect_tools(session, allowed_tools=None)


@pytest.mark.asyncio
async def test_the_page_ceiling_fails_the_whole_request() -> None:
    pages = [_page([_tool(f"t{i}")], next_cursor=f"p{i + 1}") for i in range(DISCOVERY_MAX_PAGES + 1)]
    session = _FakeSession(pages)

    with pytest.raises(McpDiscoveryRefused):
        await collect_tools(session, allowed_tools=None)

    assert len(session.cursors) == DISCOVERY_MAX_PAGES


@pytest.mark.asyncio
async def test_the_examined_descriptor_ceiling_counts_allowlist_removals_too() -> None:
    removed = [_tool(f"denied{i}") for i in range(DISCOVERY_MAX_EXAMINED + 1)]
    session = _FakeSession([_page(removed)])

    with pytest.raises(McpDiscoveryRefused):
        await collect_tools(session, allowed_tools=["a"])


@pytest.mark.asyncio
async def test_a_conflicting_duplicate_tool_name_fails_the_whole_request() -> None:
    session = _FakeSession(
        [
            _page([_tool("a", "one thing")], next_cursor="p2"),
            _page([_tool("a", "something else")]),
        ]
    )

    with pytest.raises(McpDiscoveryRefused):
        await collect_tools(session, allowed_tools=None)


@pytest.mark.asyncio
async def test_an_identical_repeat_of_one_tool_is_not_a_conflict() -> None:
    session = _FakeSession(
        [
            _page([_tool("a")], next_cursor="p2"),
            _page([_tool("a")]),
        ]
    )

    tools = await collect_tools(session, allowed_tools=None)

    assert [t.name for t in tools] == ["a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", ["not-a-list", [{"name": "a"}], [None]])
async def test_a_malformed_page_fails_the_whole_request(tools: Any) -> None:
    class _Page:
        nextCursor = None

    page = _Page()
    page.tools = tools  # type: ignore[attr-defined]
    session = _FakeSession([page])

    with pytest.raises(McpDiscoveryRefused):
        await collect_tools(session, allowed_tools=None)
