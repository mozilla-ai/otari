"""Which workspaces have an MCP server attached to every request, as the request path reads it.

A workspace attaches one of its own servers, or one of its organization's: through the server's own
``attach_by_default``, or through the workspace's settings on it. The sets held here are a superset: an
organization's attached server counts for every workspace of it, and a workspace's setting counts whether
or not the server is on, since the request goes on to read the servers themselves.

Every request made with a key holding ``mcp:execute`` asks this, and nearly
every workspace has no such server, so the answer is held in a process-wide
set rather than read per request. It is refreshed on a TTL and right after a
server is written, the arrangement ``services/providers/_provider_endpoint_cache``
uses: a write reaches its own worker at once and every other worker on that
worker's next successful refresh, which runs every
``ATTACHED_MCP_SERVERS_CACHE_TTL_SECONDS``.

A stale entry never keeps a detached, disabled or deleted server in a request:
the request goes on to read the workspace's attached servers and finds none.
One newly attached on another worker is left out of that worker's requests
until its next refresh.
"""

import asyncio
import uuid
from collections.abc import Callable
from typing import ClassVar

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import create_unit_of_work
from gateway.log_config import logger
from gateway.repositories.tools import WorkspaceMcpServerOverrideRepository, WorkspaceMcpServerRepository

ATTACHED_MCP_SERVERS_CACHE_TTL_SECONDS = 30.0


class AttachedMcpServerIndex:
    """The workspaces and organizations with a server attached to every request, as this worker last read them."""

    _workspaces: ClassVar[frozenset[uuid.UUID]] = frozenset()
    _organizations: ClassVar[frozenset[uuid.UUID]] = frozenset()
    # Refreshes started, and the number of the newest one applied.
    _refresh_count: ClassVar[int] = 0
    _applied_refresh: ClassVar[int] = 0

    @classmethod
    def has_attached(cls, workspace_id: uuid.UUID | None, organization_id: uuid.UUID | None = None) -> bool:
        """Whether the workspace, of ``organization_id``, may have an enabled server attached to every request."""
        if workspace_id is None:
            return False
        return workspace_id in cls._workspaces or (
            organization_id is not None and organization_id in cls._organizations
        )

    @classmethod
    async def refresh(
        cls, servers: WorkspaceMcpServerRepository, overrides: WorkspaceMcpServerOverrideRepository
    ) -> None:
        """Reload the sets, three reads, and swap them in one step.

        Numbered before the read, so a refresh whose read began earlier never
        replaces what a later one already applied: a write refreshes after it
        commits, and the periodic refresher must not undo that with an older read.
        """
        cls._refresh_count += 1
        number = cls._refresh_count
        workspaces = await servers.attached_workspace_ids() | await overrides.attaching_workspace_ids()
        organizations = await servers.attached_organization_ids()
        if number < cls._applied_refresh:
            return
        cls._workspaces, cls._organizations, cls._applied_refresh = workspaces, organizations, number

    @classmethod
    async def refresh_after_write(cls) -> None:
        """Refresh on a session of its own once a write has committed, best effort.

        A failure is logged and left to the periodic refresher: the write is
        already durable, so its caller must not be told it failed.
        """
        await cls._refresh_or_log(logger.warning, "Attached MCP server refresh after a write failed")

    @classmethod
    def reset(cls) -> None:
        """Drop the set so the next load starts clean (startup and tests)."""
        cls._workspaces, cls._organizations, cls._applied_refresh = frozenset(), frozenset(), 0

    @classmethod
    async def _refresh_on_a_session_of_its_own(cls) -> None:
        async with create_unit_of_work() as uow, uow:
            await cls.refresh(WorkspaceMcpServerRepository(uow), WorkspaceMcpServerOverrideRepository(uow))

    @classmethod
    async def _refresh_or_log(cls, log: Callable[..., None], message: str) -> None:
        """Refresh on a session of its own, logging a failure as ``message`` rather than raising it."""
        try:
            await cls._refresh_on_a_session_of_its_own()
        except DATABASE_ERRORS as exc:
            log("%s: %s", message, describe_database_error(exc))
        except Exception as exc:  # noqa: BLE001 - the periodic refresher retries
            log("%s: %s", message, type(exc).__name__)

    @classmethod
    async def load_at_startup(cls) -> None:
        """Prime the set before the first request. A failure is logged, not raised."""
        cls.reset()
        await cls._refresh_or_log(
            logger.error, "Failed to load the workspaces with attached MCP servers; continuing with none cached"
        )

    @classmethod
    async def run_refresher(cls, interval: float = ATTACHED_MCP_SERVERS_CACHE_TTL_SECONDS) -> None:
        """Reload the set forever, so another worker's writes arrive. Every error is retried next tick."""
        while True:
            await asyncio.sleep(interval)
            await cls._refresh_or_log(logger.warning, f"Attached MCP server refresh failed; retrying in {interval}s")
