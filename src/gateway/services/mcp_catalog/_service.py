"""The MCP server catalog: searching it, an organization's featured shelf, and the sync that fills it."""

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.mcp_catalog_exceptions import (
    McpCatalogFeaturedLimitReachedError,
    McpCatalogFeaturedNotFoundError,
    McpCatalogServerNotFoundError,
    McpCatalogSyncDisabledError,
)
from gateway.log_config import logger
from gateway.models.mcp_catalog import (
    OUTCOME_COMPLETE,
    OUTCOME_FAILED,
    OUTCOME_PARTIAL,
    McpCatalogFeatured,
    McpCatalogServer,
    McpCatalogSyncState,
)
from gateway.models.tenancy import Organization, User
from gateway.repositories.mcp_catalog import (
    McpCatalogFeaturedRepository,
    McpCatalogServerRepository,
    McpCatalogSyncStateRepository,
)
from gateway.schemas.mcp_catalog import (
    CatalogStatus,
    McpCatalogEntry,
    McpCatalogFeaturedListPublic,
    McpCatalogFeaturedPublic,
    McpCatalogFeatureRequest,
    McpCatalogServerPublic,
    McpCatalogServersPublic,
    McpCatalogSyncStatus,
)
from gateway.services.mcp_catalog._registry import PAGE_SIZE, FetchPage, RegistryError, entry_from_registry_item

MAX_FEATURED_PER_ORGANIZATION = 50
# The registry listed about 61,000 servers on 2026-11-10, which a sync read in under two minutes. This leaves
# it room to grow eightfold before a sync stops early, and a sync that does says so.
MAX_PAGES = 500_000 // PAGE_SIZE
MAX_SYNC_DURATION = timedelta(minutes=30)
# A running sync moves its heartbeat with every page, and a page takes two minutes at the very worst.
ABANDONED_AFTER = timedelta(minutes=10)
# An unfinished crawl is picked up where it stopped for this long. After it, the pages it read are too old
# to count as one reading of the registry, and the next sync starts over.
RESUME_WITHIN = timedelta(hours=6)
# A scheduled sync that did not finish tries again this soon, inside RESUME_WITHIN, instead of a full interval later.
RETRY_AFTER = timedelta(hours=1)
_MAX_ERROR_LENGTH = 300


class OrganizationAccess(Protocol):
    """What the catalog needs of the organizations domain: whose shelf a caller sees, and who may change it."""

    async def get_active_organization_for_user(self, user: User) -> Organization: ...

    async def require_active_organization_management_access(
        self, *, user: User, organization: Organization
    ) -> object: ...


class _SyncStopped(Exception):
    """A sync reached one of its own bounds before the registry's last page."""


class _SyncTakenOver(Exception):
    """Another replica claimed the sync this run was carrying out, having taken it for dead."""


def _server_public(row: McpCatalogServer) -> McpCatalogServerPublic:
    return McpCatalogServerPublic(
        id=row.id,
        name=row.name,
        title=row.title,
        description=row.description,
        url=row.url,
        requires_token=row.requires_token,
        website_url=row.website_url,
        version=row.version,
        source=row.source,
    )


def _featured_public(row: McpCatalogFeatured, listed: tuple[uuid.UUID, str] | None) -> McpCatalogFeaturedPublic:
    """``listed`` is the id and URL the catalog holds for this entry's name today, or None when it holds none."""
    status: CatalogStatus = "removed" if listed is None else "current" if listed[1] == row.url else "changed"
    return McpCatalogFeaturedPublic(
        id=row.id,
        name=row.name,
        title=row.title,
        description=row.description,
        url=row.url,
        requires_token=row.requires_token,
        website_url=row.website_url,
        source=row.source,
        catalog_status=status,
        catalog_server_id=listed[0] if listed is not None else None,
        created_at=row.created_at,
    )


class McpCatalogService:
    """Searches the catalog, keeps each organization's featured shelf, and runs the sync when one is due."""

    def __init__(
        self,
        uow: UnitOfWork,
        servers: McpCatalogServerRepository,
        featured: McpCatalogFeaturedRepository,
        state: McpCatalogSyncStateRepository,
        organizations: OrganizationAccess | None,
        config: GatewayConfig,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._uow = uow
        self._servers = servers
        self._featured = featured
        self._state = state
        self._organizations = organizations
        self._config = config
        self._clock = clock

    async def search(self, *, query: str, skip: int, limit: int) -> McpCatalogServersPublic:
        """A page of the catalog. An empty query pages through all of it."""
        async with self._uow:
            rows, count = await self._servers.search(query, skip=skip, limit=limit)
            return McpCatalogServersPublic(data=[_server_public(row) for row in rows], count=count)

    def _organization_access(self) -> OrganizationAccess:
        if self._organizations is None:
            msg = "This MCP catalog service was built for the sync worker and cannot answer for an organization"
            raise RuntimeError(msg)
        return self._organizations

    async def list_featured(self, *, user: User) -> McpCatalogFeaturedListPublic:
        """The shelf of the organization the caller acts in. Any of its members may read it."""
        organization = await self._organization_access().get_active_organization_for_user(user)
        async with self._uow:
            rows = await self._featured.list_for_organization(organization.id)
            listed = await self._servers.listed_by_name([row.name for row in rows])
            return McpCatalogFeaturedListPublic(
                data=[_featured_public(row, listed.get(row.name)) for row in rows], count=len(rows)
            )

    async def feature(self, *, user: User, request: McpCatalogFeatureRequest) -> McpCatalogFeaturedPublic:
        """Put a catalog server on the caller's organization's shelf, as the catalog lists it now.

        Featuring a server that is already there takes its copy again, which is
        how an admin accepts a URL the catalog has since changed.
        """
        organization_id = await self._managed_organization(user)
        async with self._uow:
            server = await self._servers.get(request.server_id)
            if server is None:
                raise McpCatalogServerNotFoundError(request.server_id)
            copy = {
                "title": server.title,
                "description": server.description,
                "url": server.url,
                "requires_token": server.requires_token,
                "website_url": server.website_url,
                "source": server.source,
            }
            row = await self._featured.get_by_name(organization_id, server.name)
            if row is not None:
                row = await self._featured.update(row, copy)
            else:
                if await self._featured.count_for_organization(organization_id) >= MAX_FEATURED_PER_ORGANIZATION:
                    raise McpCatalogFeaturedLimitReachedError(MAX_FEATURED_PER_ORGANIZATION)
                # None when a concurrent request featured the same server first; that entry takes the copy.
                row = await self._featured.add(organization_id, server.name, copy)
                if row is None:
                    existing = await self._featured.get_by_name(organization_id, server.name)
                    if existing is None:
                        raise McpCatalogServerNotFoundError(request.server_id)
                    row = await self._featured.update(existing, copy)
            return _featured_public(row, (server.id, server.url))

    async def unfeature(self, *, user: User, featured_id: uuid.UUID) -> None:
        """Take an entry off the caller's organization's shelf."""
        organization_id = await self._managed_organization(user)
        async with self._uow:
            row = await self._featured.get_in_organization(organization_id, featured_id)
            if row is None:
                raise McpCatalogFeaturedNotFoundError(featured_id)
            await self._featured.delete(row)

    async def _managed_organization(self, user: User) -> uuid.UUID:
        organizations = self._organization_access()
        organization = await organizations.get_active_organization_for_user(user)
        await organizations.require_active_organization_management_access(user=user, organization=organization)
        return organization.id

    async def sync_status(self) -> McpCatalogSyncStatus:
        async with self._uow:
            return await self._status(await self._state.get_or_create())

    async def request_sync(self) -> McpCatalogSyncStatus:
        """Ask for a sync now. A worker starts it within a poll, on whichever replica claims it."""
        if self._config.mcp_catalog_sync == "off":
            raise McpCatalogSyncDisabledError
        async with self._uow:
            state = await self._state.update(await self._state.get_or_create(), {"sync_requested": True})
            return await self._status(state)

    async def _status(self, state: McpCatalogSyncState) -> McpCatalogSyncStatus:
        mode = self._config.mcp_catalog_sync
        running = state.running and (
            state.heartbeat_at is not None and state.heartbeat_at >= self._clock() - ABANDONED_AFTER
        )
        return McpCatalogSyncStatus(
            mode=mode,
            running=running,
            # Nothing will start it while the sync is off, so it is not reported as on its way.
            requested=state.sync_requested and mode != "off",
            outcome=state.outcome,  # type: ignore[arg-type]  # only _finish writes it, from the three constants
            error=state.error,
            started_at=state.crawl_started_at,
            finished_at=state.finished_at,
            last_completed_at=state.last_completed_at,
            next_sync_at=state.next_sync_at if mode == "scheduled" else None,
            pages=state.pages,
            fetched=state.fetched,
            kept=state.kept,
            removed=state.removed,
            server_count=await self._servers.count(),
        )

    async def sync_if_due(self, fetch_page: FetchPage) -> McpCatalogSyncStatus | None:
        """Claim a due sync and carry it out, or return None when none is due or another replica claimed it.

        Each page is fetched outside any transaction and stored in one of its
        own, with the cursor of the page after it, so a sync that stops keeps
        what it read and the next one resumes there. Only a sync that reached
        the registry's last page deletes the servers it did not see.
        """
        mode = self._config.mcp_catalog_sync
        if mode == "off":
            return None
        now = self._clock()
        async with self._uow:
            # The claim is an update, so the row has to exist first.
            await self._state.get_or_create()
            claimed = await self._state.claim(
                now,
                scheduled=mode == "scheduled",
                stale_before=now - ABANDONED_AFTER,
                next_sync_at=now + timedelta(hours=self._config.mcp_catalog_sync_interval_hours),
            )
            if not claimed:
                return None
            state = await self._state.get_or_create()
            resumed = (
                state.cursor is not None
                and state.crawl_started_at is not None
                and state.crawl_started_at >= now - RESUME_WITHIN
            )
            if not resumed:
                state = await self._state.update(
                    state, {"crawl_started_at": now, "cursor": None, "pages": 0, "fetched": 0, "kept": 0}
                )
            state = await self._state.update(state, {"removed": 0, "error": None})
            crawl_started_at, cursor, pages = state.crawl_started_at or now, state.cursor, state.pages

        beat = now
        stored_a_page = False
        try:
            while True:
                if pages >= MAX_PAGES:
                    raise _SyncStopped(f"stopped after {MAX_PAGES} pages")
                if self._clock() - now >= MAX_SYNC_DURATION:
                    raise _SyncStopped(f"stopped after {int(MAX_SYNC_DURATION.total_seconds() // 60)} minutes")
                page = await fetch_page(cursor)
                # One page can list a name twice; the last one wins.
                entries: dict[str, McpCatalogEntry] = {}
                for item in page.items:
                    entry = entry_from_registry_item(item)
                    if entry is not None:
                        entries[entry.name] = entry
                cursor, pages = page.next_cursor, pages + 1
                async with self._uow:
                    next_beat = await self._hold(beat)
                    refused = await self._servers.upsert(list(entries.values()), seen_at=crawl_started_at)
                    state = await self._state.get_or_create()
                    await self._state.update(
                        state,
                        {
                            "cursor": cursor,
                            "pages": pages,
                            "fetched": state.fetched + len(page.items),
                            "kept": state.kept + len(entries) - refused,
                        },
                    )
                beat = next_beat
                stored_a_page = True
                if refused:
                    logger.warning("MCP catalog sync left out %s entries the database refused", refused)
                if cursor is None:
                    break
        except _SyncTakenOver:
            return None
        except (RegistryError, _SyncStopped) as exc:
            # A resumed sync that stored nothing may be holding a cursor the registry no longer takes.
            return await self._finish(beat, error=str(exc), start_over=resumed and not stored_a_page)
        except DATABASE_ERRORS as exc:
            logger.warning("MCP catalog sync could not store a page: %s", describe_database_error(exc))
            return await self._finish(beat, error="a page could not be stored in the database")
        except asyncio.CancelledError:
            # Shutdown. Left as running, the row would hold the next sync back until its heartbeat went stale.
            # The write is shielded from the cancellation, and a database already closing is not worth an error.
            try:
                await asyncio.shield(self._finish(beat, error="the gateway stopped before the sync finished"))
            except DATABASE_ERRORS:
                pass
            raise
        return await self._finish(beat, remove_unseen_before=crawl_started_at)

    async def _hold(self, beat: datetime) -> datetime:
        """Move the running sync's heartbeat on from ``beat``, proving this run still owns it.

        Called inside the block that stores a page. A run another replica took over
        finds its heartbeat gone and stops, without touching what the new run wrote.
        """
        now = self._clock()
        if not await self._state.hold(beat, now):
            raise _SyncTakenOver
        return now

    async def _finish(
        self,
        beat: datetime,
        *,
        error: str | None = None,
        start_over: bool = False,
        remove_unseen_before: datetime | None = None,
    ) -> McpCatalogSyncStatus | None:
        """End the run that last beat at ``beat``. None when another replica has taken the sync over."""
        finished = self._clock()
        try:
            async with self._uow:
                await self._hold(beat)
                state = await self._state.get_or_create()
                changes: dict[str, object] = {"running": False, "finished_at": finished}
                reached_the_end = error is None and remove_unseen_before is not None
                if reached_the_end and not state.kept:
                    # Every server leaving the registry at once is far less likely than a registry answering
                    # oddly, so a crawl that kept nothing removes nothing.
                    outcome = OUTCOME_FAILED
                    error = "the registry listed no server Otari can connect to; the catalog was left as it was"
                elif remove_unseen_before is not None and reached_the_end:
                    total = await self._servers.count()
                    unseen = await self._servers.count_unseen(remove_unseen_before)
                    if unseen * 2 > total:
                        # The same reasoning, one step down: a registry that lost most of its servers overnight
                        # is more likely cut short than emptied.
                        outcome = OUTCOME_PARTIAL
                        error = (
                            f"the registry listed {total - unseen} of the {total} servers in the catalog, "
                            "too few to trust; nothing was removed"
                        )
                    else:
                        outcome = OUTCOME_COMPLETE
                        changes |= {
                            "last_completed_at": finished,
                            "removed": await self._servers.delete_unseen(remove_unseen_before),
                        }
                else:
                    outcome = OUTCOME_PARTIAL if state.pages else OUTCOME_FAILED
                changes["outcome"] = outcome
                changes["error"] = error[:_MAX_ERROR_LENGTH] if error else None
                if reached_the_end or start_over:
                    changes["cursor"] = None
                if outcome != OUTCOME_COMPLETE and state.next_sync_at is not None:
                    changes["next_sync_at"] = min(state.next_sync_at, finished + RETRY_AFTER)
                return await self._status(await self._state.update(state, changes))
        except _SyncTakenOver:
            return None
