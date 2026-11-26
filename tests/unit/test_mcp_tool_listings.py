"""The kept MCP tool listings: how long one lives, what it is keyed on, and one load per key at a time."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from mcp.types import Tool as MCPTool

from gateway.models.mcp import ResolvedMcpServer
from gateway.models.tools import McpCredentialMode
from gateway.services.mcp_stateless import listing_key
from gateway.services.tools import McpToolListings

TOOLS = [MCPTool(name="create_issue", description="Create an issue", input_schema={"type": "object"})]


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Loader:
    def __init__(self, tools: list[MCPTool] = TOOLS) -> None:
        self.calls = 0
        self.tools = tools

    async def __call__(self) -> list[MCPTool]:
        self.calls += 1
        return self.tools


@pytest.mark.asyncio
async def test_a_listing_is_served_until_it_expires() -> None:
    clock = _Clock()
    listings = McpToolListings(ttl_s=60, clock=clock)
    load = _Loader()

    assert await listings.get_or_load("k", load) == (TOOLS, 0.0)
    clock.now += 59
    assert await listings.get_or_load("k", load) == (TOOLS, 59.0)
    clock.now += 1
    assert await listings.get_or_load("k", load) == (TOOLS, 0.0)
    assert load.calls == 2


@pytest.mark.asyncio
async def test_refresh_loads_again_and_replaces_the_kept_listing() -> None:
    listings = McpToolListings(clock=_Clock())
    load = _Loader()

    await listings.get_or_load("k", load)
    await listings.get_or_load("k", load, refresh=True)
    await listings.get_or_load("k", load)

    assert load.calls == 2


@pytest.mark.asyncio
async def test_concurrent_misses_on_one_key_load_once() -> None:
    listings = McpToolListings()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def slow() -> list[MCPTool]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return TOOLS

    first = asyncio.create_task(listings.get_or_load("k", slow))
    await started.wait()
    others = [asyncio.create_task(listings.get_or_load("k", slow)) for _ in range(5)]
    await asyncio.sleep(0)
    release.set()

    results = await asyncio.gather(first, *others)

    assert calls == 1
    assert all(tools == TOOLS for tools, _ in results)
    assert listings._flights == {}


@pytest.mark.asyncio
async def test_a_failed_load_keeps_nothing_and_the_next_caller_loads() -> None:
    listings = McpToolListings()
    load = _Loader()

    async def broken() -> list[MCPTool]:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        await listings.get_or_load("k", broken)
    await listings.get_or_load("k", load)

    assert load.calls == 1


@pytest.mark.asyncio
async def test_the_least_recently_used_listing_leaves_when_the_byte_ceiling_is_reached() -> None:
    listings = McpToolListings()
    await listings.get_or_load("probe", _Loader())
    size = listings._bytes
    listings = McpToolListings(max_bytes=size * 2)
    load = _Loader()

    await listings.get_or_load("a", load)
    await listings.get_or_load("b", load)
    await listings.get_or_load("a", load)
    await listings.get_or_load("c", load)
    await listings.get_or_load("a", load)
    await listings.get_or_load("b", load)

    assert load.calls == 4


@pytest.mark.asyncio
async def test_a_listing_larger_than_the_ceiling_is_not_kept() -> None:
    listings = McpToolListings(max_bytes=1)
    load = _Loader()

    await listings.get_or_load("k", load)
    await listings.get_or_load("k", load)

    assert load.calls == 2


@pytest.mark.asyncio
async def test_empty_listings_are_bounded_by_count() -> None:
    listings = McpToolListings(max_entries=2)
    load = _Loader(tools=[])

    for key in ("a", "b", "c"):
        await listings.get_or_load(key, load)
    await listings.get_or_load("a", load)

    assert len(listings._entries) == 2
    assert load.calls == 4


@pytest.mark.asyncio
async def test_storing_a_listing_sweeps_the_expired_ones() -> None:
    clock = _Clock()
    listings = McpToolListings(ttl_s=60, clock=clock)
    load = _Loader()
    await listings.get_or_load("old", load)
    clock.now += 30
    await listings.get_or_load("recent", load)
    clock.now += 30

    await listings.get_or_load("new", load)

    assert list(listings._entries) == ["recent", "new"]
    assert listings._bytes == sum(entry.size for entry in listings._entries.values())


def _server(**overrides: object) -> ResolvedMcpServer:
    fields: dict[str, object] = {
        "id": uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e"),
        "name": "github",
        "url": "https://93.184.216.34/mcp",
        "authorization_token": "user-a-token",
        "credential_revision": McpCredentialMode.PER_USER,
    }
    fields.update(overrides)
    return ResolvedMcpServer.model_validate(fields)


def test_two_users_of_a_per_user_server_share_a_revision_but_never_a_listing() -> None:
    mine, theirs = _server(), _server(authorization_token="user-b-token")

    assert mine.revision == theirs.revision
    assert listing_key(mine) != listing_key(theirs)


def test_an_edit_to_the_server_changes_the_key() -> None:
    assert listing_key(_server()) != listing_key(_server(allowed_tools=["create_issue"]))
    assert listing_key(_server()) != listing_key(_server(url="https://93.184.216.35/mcp"))


def test_the_key_carries_no_credential() -> None:
    assert "user-a-token" not in listing_key(_server())
