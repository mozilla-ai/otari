"""Each stored MCP server's raw tool listing, kept for a short while so tool discovery need not reconnect on every call.

What is kept is the server's own answer to ``tools/list`` before any caller's access rule or screening
applies, so every caller that shares a key shares the entry and still sees only what it may. The key
is ``mcp_stateless.listing_key``: an edit to the server or a different credential misses by construction.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp.types import Tool as MCPTool

LISTING_TTL_S = 300.0
"""How long a listing is served before the server is asked again: the bound on missing a tool it changed itself."""

LISTINGS_MAX_BYTES = 32 * 1024 * 1024
"""Ceiling on the estimated memory of every kept listing together; the least recently used leave first."""

_MEMORY_PER_JSON_BYTE = 8
"""Parsed ``Tool`` objects measure about five times their JSON; the estimate rounds that up."""

_ENTRY_OVERHEAD_BYTES = 1024

LISTINGS_MAX_ENTRIES = 4096
"""Ceiling on how many listings are kept, which bounds the empty ones the byte ceiling does not count."""


@dataclass(slots=True)
class _Entry:
    tools: list[MCPTool]
    size: int
    loaded_at: float


@dataclass(slots=True)
class _Flight:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class McpToolListings:
    """A per-process, bounded, time-limited store of raw tool listings, with one load per key at a time.

    Concurrent misses on one key wait for the first load instead of each opening a session, and a
    failed load stores nothing, so the next caller tries the server itself.
    """

    def __init__(
        self,
        *,
        ttl_s: float = LISTING_TTL_S,
        max_bytes: int = LISTINGS_MAX_BYTES,
        max_entries: int = LISTINGS_MAX_ENTRIES,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._ttl_s = ttl_s
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._bytes = 0
        self._flights: dict[str, _Flight] = {}

    async def get_or_load(
        self,
        key: str,
        load: Callable[[], Awaitable[list[MCPTool]]],
        *,
        refresh: bool = False,
    ) -> tuple[list[MCPTool], float]:
        """The listing for ``key`` and its age in seconds, loading it when none is fresh or ``refresh`` asks.

        A refresh accepts a listing another caller loaded after the refresh was asked for.
        """
        asked_at = self._clock()
        if not refresh and (hit := self._fresh(key, None)) is not None:
            return hit
        flight = self._flights.setdefault(key, _Flight())
        flight.users += 1
        try:
            async with flight.lock:
                if (hit := self._fresh(key, asked_at if refresh else None)) is not None:
                    return hit
                tools = await load()
                self._keep(key, tools)
                return tools, 0.0
        finally:
            flight.users -= 1
            if flight.users == 0:
                del self._flights[key]

    def _fresh(self, key: str, loaded_since: float | None) -> tuple[list[MCPTool], float] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        age = self._clock() - entry.loaded_at
        if age >= self._ttl_s:
            self._drop(key)
            return None
        if loaded_since is not None and entry.loaded_at <= loaded_since:
            return None
        self._entries.move_to_end(key)
        return entry.tools, age

    def _keep(self, key: str, tools: list[MCPTool]) -> None:
        self._drop(key)
        self._sweep()
        size = _ENTRY_OVERHEAD_BYTES + _MEMORY_PER_JSON_BYTE * sum(
            len(tool.model_dump_json(by_alias=True, exclude_none=True)) for tool in tools
        )
        if size > self._max_bytes:
            return
        while self._entries and (self._bytes + size > self._max_bytes or len(self._entries) >= self._max_entries):
            self._drop(next(iter(self._entries)))
        self._entries[key] = _Entry(tools=tools, size=size, loaded_at=self._clock())
        self._bytes += size

    def _sweep(self) -> None:
        """Drop every expired listing, so one nobody reads again does not hold memory until it is pushed out."""
        now = self._clock()
        for key in [key for key, entry in self._entries.items() if now - entry.loaded_at >= self._ttl_s]:
            self._drop(key)

    def _drop(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._bytes -= entry.size
