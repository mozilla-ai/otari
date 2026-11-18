"""Which workspaces have an MCP server attached to every request, as the request path reads it.

Every request made with a key holding ``mcp:execute`` asks this, and nearly
every workspace has no such server, so the answer is held in a process-wide
set rather than read per request. It is refreshed on a TTL and right after a
server is written, the arrangement ``services/providers/_provider_endpoint_cache``
uses: a write reaches its own worker at once and every other worker within
``ATTACHED_MCP_SERVERS_CACHE_TTL_SECONDS``.

A stale entry is harmless for a server that was detached: the request goes on
to read the workspace's attached servers and finds none. One newly attached on
another worker is left out of that worker's requests until its next refresh.
"""

import asyncio
import uuid
from typing import ClassVar

from gateway.core.unit_of_work import create_unit_of_work
from gateway.log_config import logger
from gateway.repositories.tools import WorkspaceMcpServerRepository

ATTACHED_MCP_SERVERS_CACHE_TTL_SECONDS = 30.0


class AttachedMcpServerIndex:
    """The workspaces with an enabled server marked ``attach_by_default``, as this worker last read them."""

    _workspaces: ClassVar[frozenset[uuid.UUID]] = frozenset()
    # Refreshes started, and the number of the newest one applied.
    _refresh_count: ClassVar[int] = 0
    _applied_refresh: ClassVar[int] = 0

    @classmethod
    def has_attached(cls, workspace_id: uuid.UUID | None) -> bool:
        """Whether the workspace has an enabled server attached to every request."""
        return workspace_id is not None and workspace_id in cls._workspaces

    @classmethod
    async def refresh(cls, servers: WorkspaceMcpServerRepository) -> None:
        """Reload the set in one query and swap it in one step.

        Numbered before the read, so a refresh whose read began earlier never
        replaces what a later one already applied: a write refreshes after it
        commits, and the periodic refresher must not undo that with an older read.
        """
        cls._refresh_count += 1
        number = cls._refresh_count
        workspaces = await servers.attached_workspace_ids()
        if number < cls._applied_refresh:
            return
        cls._workspaces, cls._applied_refresh = workspaces, number

    @classmethod
    def reset(cls) -> None:
        """Drop the set so the next load starts clean (startup and tests)."""
        cls._workspaces, cls._applied_refresh = frozenset(), 0

    @classmethod
    async def _refresh_on_a_session_of_its_own(cls) -> None:
        async with create_unit_of_work() as uow, uow:
            await cls.refresh(WorkspaceMcpServerRepository(uow))

    @classmethod
    async def load_at_startup(cls) -> None:
        """Prime the set before the first request. A failure is logged, not raised."""
        cls.reset()
        try:
            await cls._refresh_on_a_session_of_its_own()
        except Exception:
            logger.exception("Failed to load the workspaces with attached MCP servers; continuing with none cached")

    @classmethod
    async def run_refresher(cls, interval: float = ATTACHED_MCP_SERVERS_CACHE_TTL_SECONDS) -> None:
        """Reload the set forever, so another worker's writes arrive. Every error is retried next tick."""
        while True:
            await asyncio.sleep(interval)
            try:
                await cls._refresh_on_a_session_of_its_own()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Attached MCP server refresh failed; retrying in %ss", interval, exc_info=True)
