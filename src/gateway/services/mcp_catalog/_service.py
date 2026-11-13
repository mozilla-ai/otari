"""The MCP server catalog: searching it, its Popular shelf, server logos, and the sync that fills it."""

import asyncio
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Protocol
from urllib.parse import quote

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions.mcp_catalog_exceptions import (
    McpCatalogSyncDisabledError,
    McpCatalogSyncForbiddenError,
    McpCatalogSyncTooSoonError,
)
from gateway.log_config import logger
from gateway.models.mcp_catalog import (
    LOGO_OK,
    OUTCOME_COMPLETE,
    OUTCOME_FAILED,
    OUTCOME_PARTIAL,
    SOURCE_CURATED,
    SOURCE_REGISTRY,
    McpCatalogServer,
    McpCatalogSyncState,
    McpLogo,
)
from gateway.models.tenancy import User, Workspace
from gateway.repositories.mcp_catalog import (
    McpCatalogServerRepository,
    McpCatalogSyncStateRepository,
    McpLogoRepository,
    StoredLogoState,
)
from gateway.schemas.mcp_catalog import (
    McpCatalogEntry,
    McpCatalogServerPublic,
    McpCatalogServersPublic,
    McpCatalogSyncStatus,
)
from gateway.services.mcp_catalog._extras import EXTRA_SERVERS
from gateway.services.mcp_catalog._logo_fetch import LogoFetcher
from gateway.services.mcp_catalog._logos import (
    Logo,
    bundled_logo,
    catalog_key,
    catalog_logo_sources,
    host_key,
    host_logo_sources,
    logo_tone,
    publisher_kind,
)
from gateway.services.mcp_catalog._popular import POPULAR_SERVERS
from gateway.services.mcp_catalog._registry import (
    PAGE_SIZE,
    FetchPage,
    RegistryClient,
    RegistryError,
    entry_from_registry_item,
)

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
# How long a workspace admin waits after a sync ends before asking for another: a complete one, and any other.
# Deployment-wide, so however many admins click, the registry is read at most this often on their behalf.
REQUEST_COOLDOWN_AFTER_COMPLETE = timedelta(hours=1)
REQUEST_COOLDOWN_AFTER_FAILURE = timedelta(minutes=5)
# A catalog no sync has read the whole registry into for this long is offered a refresh.
STALE_AFTER = timedelta(days=1)
_MAX_ERROR_LENGTH = 300

# A logo found, or looked for and not found, is looked for again after this.
LOGO_RETRY_AFTER = timedelta(days=7)
# A look that ran out of time says nothing about the logo, so it is tried again sooner.
LOGO_RETRY_AFTER_TIMEOUT = timedelta(hours=1)
# One look tries several candidates, each with five seconds of its own; the whole look gets this.
LOGO_LOOK_SECONDS = 10.0
# Per process: looks running at once, and looks started in a minute. Past either, a page view
# that would start one gets no logo this time instead of reaching out to the internet.
MAX_CONCURRENT_LOGO_LOOKS = 4
MAX_LOGO_LOOKS_PER_MINUTE = 30


class LogoLookBudget:
    """How many logo looks this process may run now. Shared by every request, since page views start them."""

    def __init__(
        self,
        *,
        concurrent: int = MAX_CONCURRENT_LOGO_LOOKS,
        per_minute: int = MAX_LOGO_LOOKS_PER_MINUTE,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._concurrent = concurrent
        self._per_minute = per_minute
        self._clock = clock
        self._running: set[str] = set()
        self._started: deque[float] = deque()

    def start(self, key: str) -> bool:
        """Claim a look for ``key``, or refuse: one for it already running, or the process is at a limit."""
        now = self._clock()
        while self._started and self._started[0] <= now - 60:
            self._started.popleft()
        if key in self._running or len(self._running) >= self._concurrent or len(self._started) >= self._per_minute:
            return False
        self._running.add(key)
        self._started.append(now)
        return True

    def finish(self, key: str) -> None:
        self._running.discard(key)


LOGO_LOOKS = LogoLookBudget()


class LogoFetching(Protocol):
    """Fetches logo candidates; see ``LogoFetcher``."""

    async def fetch(self, url: str) -> Logo | None: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class LogoRef:
    """What a list says about one server's logo, without its image."""

    # The image's digest when one is stored or bundled; the URL that serves it carries it.
    etag: str | None
    luminance: float | None
    # Never looked for, or due to be looked for again: asking for it starts a look.
    pending: bool

    def url(self, path: str) -> str | None:
        """``path`` versioned by the image it serves, bare while a look is due, or None when there is none."""
        # A due look runs only when the logo is asked for, which a browser holding the immutable copy never does.
        if self.pending:
            return path
        return f"{path}?v={self.etag}" if self.etag is not None else None


def catalog_logo_path(name: str) -> str:
    return f"{API_ROOT}/mcp-catalog/logos/{quote(name, safe='/')}"


class WorkspaceManagement(Protocol):
    """What the catalog needs of the tenancy domain: whether a caller may add MCP servers to a workspace."""

    async def resolve_visible_workspace(self, *, user: User, workspace_id: uuid.UUID) -> Workspace: ...

    async def require_workspace_management_access(self, *, user: User, workspace: Workspace) -> None: ...


class WorkspaceServers(Protocol):
    """What the catalog needs of the tools domain: the URLs of the servers a workspace already has."""

    async def urls_in_workspace(self, workspace_id: uuid.UUID) -> dict[uuid.UUID, str]: ...


def _stored_logo(row: McpLogo | None) -> Logo | None:
    """The image a stored row holds, if it found one."""
    if row is None or row.status != LOGO_OK or not row.content or not row.content_type or not row.etag:
        return None
    return Logo(row.content, row.content_type, row.etag, row.luminance)


class _SyncStopped(Exception):
    """A sync reached one of its own bounds before the registry's last page."""


class _CursorRepeated(Exception):
    """The registry answered a page with the cursor it was asked for, so reading on would make no progress."""


class _SyncTakenOver(Exception):
    """Another replica claimed the sync this run was carrying out, having taken it for dead."""


def _server_public(row: McpCatalogServer, logo: LogoRef) -> McpCatalogServerPublic:
    return McpCatalogServerPublic(
        id=row.id,
        name=row.name,
        title=row.title,
        description=row.description,
        url=row.url,
        requires_token=row.requires_token,
        website_url=row.website_url,
        icon_url=row.icon_url,
        logo_url=logo.url(catalog_logo_path(row.name)),
        logo_tone=logo_tone(logo.luminance),
        publisher_kind=publisher_kind(row.name, row.title, row.url, row.website_url),
        version=row.version,
        source=row.source,
    )


def _judged(entry: McpCatalogEntry) -> McpCatalogEntry:
    """``entry`` marked as its maker's own server or not, which is what a search lists first."""
    kind = publisher_kind(entry.name, entry.title, entry.url, entry.website_url)
    return entry.model_copy(update={"is_official": kind == "vendor"})


def _spellings(url: str) -> tuple[str, str]:
    """``url`` without and with a trailing slash, the two spellings of one endpoint a catalog row may hold."""
    bare = url.strip().rstrip("/")
    return bare, f"{bare}/"


def _pick(rows: Sequence[McpCatalogServer], preferred_name: str | None) -> McpCatalogServer:
    """One of the rows listed at a URL: the one under the expected name, or else the first by name."""
    return next((row for row in rows if row.name == preferred_name), rows[0])


def _ref(state: StoredLogoState | None, now: datetime) -> LogoRef:
    if state is None:
        return LogoRef(etag=None, luminance=None, pending=True)
    found = state.status == LOGO_OK
    return LogoRef(
        etag=state.etag if found else None,
        luminance=state.luminance if found else None,
        pending=state.retry_at <= now,
    )


def _bundled_ref(name: str) -> LogoRef | None:
    logo = bundled_logo(name)
    return LogoRef(etag=logo.etag, luminance=logo.luminance, pending=False) if logo is not None else None


# How often every replica looks for a due sync.
POLL_SECONDS = 60.0


class McpCatalogService:
    """Searches the catalog, answers the Popular shelf and logo lookups, and runs the sync when one is due."""

    def __init__(
        self,
        uow: UnitOfWork,
        servers: McpCatalogServerRepository,
        state: McpCatalogSyncStateRepository,
        workspaces: WorkspaceManagement | None,
        config: GatewayConfig,
        *,
        workspace_servers: WorkspaceServers | None = None,
        logos: McpLogoRepository | None = None,
        fetcher: Callable[[], LogoFetching] = LogoFetcher,
        look_budget: LogoLookBudget = LOGO_LOOKS,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._uow = uow
        self._servers = servers
        self._state = state
        self._workspaces = workspaces
        self._workspace_servers = workspace_servers
        self._config = config
        self._logos = logos if logos is not None else McpLogoRepository(uow)
        self._fetcher = fetcher
        self._look_budget = look_budget
        self._clock = clock

    async def search(
        self, *, query: str, skip: int, limit: int, user: User | None = None, workspace_id: uuid.UUID | None = None
    ) -> McpCatalogServersPublic:
        """A page of the catalog. An empty query pages through all of it.

        With a workspace ``user`` can see, the servers that workspace already has are left out.
        """
        workspace_id = await self._visible(user, workspace_id)
        async with self._uow:
            added = await self._added_urls(workspace_id)
            rows, count = await self._servers.search(query, skip=skip, limit=limit, exclude_urls=added)
            refs = await self._catalog_refs(rows)
        return McpCatalogServersPublic(data=[_server_public(row, refs[row.name]) for row in rows], count=count)

    async def popular(
        self, *, user: User | None = None, workspace_id: uuid.UUID | None = None
    ) -> McpCatalogServersPublic:
        """The hand-maintained Popular servers the catalog holds now, in the list's order.

        Matched by URL. Where several entries list one URL, the one under the expected name is shown.
        With a workspace ``user`` can see, the servers that workspace already has are left out.
        """
        workspace_id = await self._visible(user, workspace_id)
        async with self._uow:
            added = set(await self._added_urls(workspace_id))
            rows = await self._servers.by_urls([entry.url for entry in POPULAR_SERVERS if entry.url not in added])
            by_url: dict[str, list[McpCatalogServer]] = {}
            for row in rows:
                by_url.setdefault(row.url, []).append(row)
            picked = [_pick(by_url[entry.url], entry.name) for entry in POPULAR_SERVERS if entry.url in by_url]
            refs = await self._catalog_refs(picked)
        shelf = [_server_public(row, refs[row.name]) for row in picked]
        return McpCatalogServersPublic(data=shelf, count=len(shelf))

    async def servers_by_names(self, names: Sequence[str]) -> dict[str, McpCatalogServerPublic]:
        """The catalog's entry for each of ``names`` it lists now, by name."""
        if not names:
            return {}
        async with self._uow:
            rows = await self._servers.by_names(names)
            refs = await self._catalog_refs(rows)
        return {row.name: _server_public(row, refs[row.name]) for row in rows}

    async def _visible(self, user: User | None, workspace_id: uuid.UUID | None) -> uuid.UUID | None:
        """``workspace_id`` once ``user`` is known to see that workspace, or None when no workspace was named."""
        if user is None or workspace_id is None or self._workspaces is None:
            return None
        return (await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)).id

    async def _added_urls(self, workspace_id: uuid.UUID | None) -> list[str]:
        """Both spellings of the URL of each server the workspace has. Inside a block."""
        if workspace_id is None or self._workspace_servers is None:
            return []
        urls = (await self._workspace_servers.urls_in_workspace(workspace_id)).values()
        return [spelling for url in urls for spelling in _spellings(url)]

    async def _store_extras(self, seen_at: datetime) -> None:
        """Store the bundled servers the registry does not list, as seen by the crawl begun at ``seen_at``.

        One the registry lists, under the same name or at the same URL, is left to the registry, and its
        bundled row goes with the next complete sync. Inside a block.
        """
        entries: list[McpCatalogEntry] = []
        for extra in EXTRA_SERVERS:
            listed = [
                await self._servers.by_name(extra.name),
                *await self._servers.by_urls(list(_spellings(extra.url))),
            ]
            if any(row is not None and row.source == SOURCE_REGISTRY for row in listed):
                continue
            entries.append(
                _judged(
                    McpCatalogEntry(
                        name=extra.name,
                        title=extra.title,
                        description=extra.description,
                        url=extra.url,
                        requires_token=False,
                        website_url=extra.website_url,
                        icon_url=extra.icon_url,
                        version=None,
                        source=SOURCE_CURATED,
                    )
                )
            )
        await self._servers.upsert(entries, seen_at=seen_at)

    async def _catalog_refs(self, rows: Sequence[McpCatalogServer]) -> dict[str, LogoRef]:
        """Each row's logo, by name. Inside a block."""
        states = await self._logos.states([catalog_key(row.name) for row in rows])
        now = self._clock()
        return {row.name: _bundled_ref(row.name) or _ref(states.get(catalog_key(row.name)), now) for row in rows}

    async def _row_at(self, url: str) -> McpCatalogServer | None:
        """The catalog row listed at ``url``, trailing slash or not, a Popular entry preferred. Inside a block."""
        bare = _spellings(url)[0]
        rows = await self._servers.by_urls(list(_spellings(url)))
        if not rows:
            return None
        popular = {entry.url.rstrip("/"): entry.name for entry in POPULAR_SERVERS}
        return _pick(rows, popular.get(bare))

    async def logo_refs(self, urls: Sequence[str]) -> dict[str, LogoRef]:
        """The logo of the server at each of ``urls``, by URL: the catalog's where it lists the URL, else its host's.

        Starts no look. A URL with no host gets nothing.
        """
        async with self._uow:
            rows = await self._servers.by_urls([spelling for url in urls for spelling in _spellings(url)])
            popular = {entry.url.rstrip("/"): entry.name for entry in POPULAR_SERVERS}
            by_url: dict[str, list[McpCatalogServer]] = {}
            for row in rows:
                by_url.setdefault(row.url.rstrip("/"), []).append(row)
            keys: dict[str, tuple[str, str | None]] = {}
            for url in dict.fromkeys(urls):
                bare = _spellings(url)[0]
                if bare in by_url:
                    name = _pick(by_url[bare], popular.get(bare)).name
                    keys[url] = (catalog_key(name), name)
                else:
                    key = host_key(url)
                    if key is not None:
                        keys[url] = (key, None)
            states = await self._logos.states([key for key, _ in keys.values()])
        now = self._clock()
        return {
            url: (_bundled_ref(name) if name else None) or _ref(states.get(key), now)
            for url, (key, name) in keys.items()
        }

    async def logo(self, name: str) -> Logo | None:
        """A catalog server's logo, looked for first if it is due. None for a name the catalog does not list."""
        bundled = bundled_logo(name)
        async with self._uow:
            row = await self._servers.by_name(name)
            if row is None:
                return None
            sources = catalog_logo_sources(row.name, row.website_url, row.icon_url)
        if bundled is not None:
            return bundled
        return await self._look(catalog_key(name), sources)

    async def logo_for_url(self, url: str) -> Logo | None:
        """The logo of the server at ``url``: the catalog's where it lists the URL, else one from its host."""
        async with self._uow:
            row = await self._row_at(url)
            listed = (row.name, catalog_logo_sources(row.name, row.website_url, row.icon_url)) if row else None
        if listed is not None:
            name, sources = listed
            return bundled_logo(name) or await self._look(catalog_key(name), sources)
        key = host_key(url)
        return await self._look(key, host_logo_sources(url)) if key is not None else None

    async def _look(self, key: str, sources: Sequence[str]) -> Logo | None:
        """The stored logo for ``key``, looked for among ``sources`` first when a look is due and the budget allows.

        No transaction is open while the internet is asked.
        """
        async with self._uow:
            stored = await self._logos.find(key)
            due = stored is None or stored.retry_at <= self._clock()
            kept = _stored_logo(stored)
        # With the sync off, the operator has said the catalog reaches nothing outside: no logo is looked for.
        if not due or not sources or self._config.mcp_catalog_sync == "off" or not self._look_budget.start(key):
            return kept
        try:
            found: Logo | None = None
            source = ""
            timed_out = False
            fetcher = self._fetcher()
            try:
                async with asyncio.timeout(LOGO_LOOK_SECONDS):
                    for source in sources:
                        found = await fetcher.fetch(source)
                        if found is not None:
                            break
            except TimeoutError:
                timed_out = True
            finally:
                await fetcher.aclose()
            now = self._clock()
            async with self._uow:
                if found is not None:
                    await self._logos.store_found(
                        key,
                        content=found.content,
                        content_type=found.content_type,
                        etag=found.etag,
                        luminance=found.luminance,
                        source_url=source,
                        fetched_at=now,
                        retry_at=now + LOGO_RETRY_AFTER,
                    )
                else:
                    retry = LOGO_RETRY_AFTER_TIMEOUT if timed_out else LOGO_RETRY_AFTER
                    await self._logos.store_not_found(key, fetched_at=now, retry_at=now + retry)
        finally:
            self._look_budget.finish(key)
        return found or kept

    async def sync_status(self) -> McpCatalogSyncStatus:
        async with self._uow:
            return await self._status(await self._state.get_or_create())

    async def request_sync(
        self, *, user: User, workspace_id: uuid.UUID | None, operates_deployment: bool
    ) -> McpCatalogSyncStatus:
        """Ask for a sync now. A worker starts it within a poll, on whichever replica claims it.

        An operator may always ask. Anyone else names a workspace they may add MCP servers to, and is
        refused while ``next_request_at`` is ahead. A sync already running or asked for is left as it is.

        Raises:
            McpCatalogSyncDisabledError: the sync is off on this deployment.
            McpCatalogSyncForbiddenError: no workspace was named by a caller who does not operate the deployment.
            McpCatalogSyncTooSoonError: a sync ended too recently.
        """
        if self._config.mcp_catalog_sync == "off":
            raise McpCatalogSyncDisabledError
        if not operates_deployment:
            if workspace_id is None or self._workspaces is None:
                raise McpCatalogSyncForbiddenError
            workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
            await self._workspaces.require_workspace_management_access(user=user, workspace=workspace)
        async with self._uow:
            state = await self._state.get_or_create()
            status = await self._status(state)
            if status.running or status.requested:
                return status
            if not operates_deployment and status.next_request_at is not None:
                wait = status.next_request_at - self._clock()
                raise McpCatalogSyncTooSoonError(max(1, -(-int(wait.total_seconds()) // 60)))
            return await self._status(await self._state.update(state, {"sync_requested": True}))

    def _next_request_at(self, state: McpCatalogSyncState) -> datetime | None:
        if state.finished_at is None:
            return None
        cooldown = (
            REQUEST_COOLDOWN_AFTER_COMPLETE if state.outcome == OUTCOME_COMPLETE else REQUEST_COOLDOWN_AFTER_FAILURE
        )
        ready = state.finished_at + cooldown
        return ready if ready > self._clock() else None

    async def _status(self, state: McpCatalogSyncState) -> McpCatalogSyncStatus:
        mode = self._config.mcp_catalog_sync
        server_count = await self._servers.count()
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
            server_count=server_count,
            is_stale=server_count == 0
            or state.last_completed_at is None
            or state.last_completed_at < self._clock() - STALE_AFTER,
            next_request_at=self._next_request_at(state),
        )

    async def sync_if_due(self, fetch_page: FetchPage) -> McpCatalogSyncStatus | None:
        """Claim a due sync and carry it out, or return None when none is due or another replica claimed it.

        Each page is fetched outside any transaction and stored in one of its
        own, with the cursor of the page after it, so a sync that stops keeps
        what it read and the next one resumes there. Only a sync that reached
        the registry's last page deletes the servers it did not see. The bundled
        servers the registry does not list are stored as the sync starts.
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
                    state,
                    {
                        "crawl_started_at": now,
                        "servers_at_start": await self._servers.count(),
                        "cursor": None,
                        "pages": 0,
                        "fetched": 0,
                        "kept": 0,
                    },
                )
            state = await self._state.update(state, {"removed": 0, "error": None})
            crawl_started_at, cursor, pages = state.crawl_started_at or now, state.cursor, state.pages
            await self._store_extras(crawl_started_at)

        beat = now
        stored_a_page = False
        try:
            while True:
                if pages >= MAX_PAGES:
                    raise _SyncStopped(f"stopped after {MAX_PAGES} pages")
                if self._clock() - now >= MAX_SYNC_DURATION:
                    raise _SyncStopped(f"stopped after {int(MAX_SYNC_DURATION.total_seconds() // 60)} minutes")
                page = await fetch_page(cursor)
                if cursor is not None and page.next_cursor == cursor:
                    raise _CursorRepeated("the registry answered a page with the cursor it was asked for")
                # One page can list a name twice; the last one wins.
                entries: dict[str, McpCatalogEntry] = {}
                for item in page.items:
                    entry = entry_from_registry_item(item)
                    if entry is not None:
                        entries[entry.name] = _judged(entry)
                cursor, pages = page.next_cursor, pages + 1
                async with self._uow:
                    next_beat = await self._hold(beat)
                    await self._servers.upsert(list(entries.values()), seen_at=crawl_started_at)
                    state = await self._state.get_or_create()
                    await self._state.update(
                        state,
                        {
                            "cursor": cursor,
                            "pages": pages,
                            "fetched": state.fetched + len(page.items),
                            "kept": state.kept + len(entries),
                        },
                    )
                beat = next_beat
                stored_a_page = True
                if cursor is None:
                    break
        except _SyncTakenOver:
            return None
        except (RegistryError, _SyncStopped, _CursorRepeated) as exc:
            # A resumed sync that stored nothing may be holding a cursor the registry no longer takes, and a cursor
            # that repeated would only repeat again.
            start_over = (resumed and not stored_a_page) or isinstance(exc, _CursorRepeated)
            return await self._finish(beat, error=str(exc), start_over=start_over)
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
                    # Measured against the catalog as the crawl found it: servers it added do not dilute the loss.
                    total = state.servers_at_start
                    if total is None:
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

    @staticmethod
    async def poll(
        build: Callable[[UnitOfWork], "McpCatalogService"], config: GatewayConfig, interval: float = POLL_SECONDS
    ) -> None:
        """Poll for a due sync forever, as every replica does. Errors are logged and retried; cancelled at shutdown.

        Only the replica that claims a due sync carries it out. While ``mcp_catalog_sync`` is ``off`` a poll
        neither reads the database nor opens a connection anywhere.
        """
        while True:
            await asyncio.sleep(interval)
            if config.mcp_catalog_sync == "off":
                continue
            try:
                async with create_unit_of_work() as uow, RegistryClient() as registry:
                    status = await build(uow).sync_if_due(registry.fetch_page)
                if status is not None:
                    logger.info(
                        "MCP catalog sync %s: %s pages, %s kept, %s removed",
                        status.outcome,
                        status.pages,
                        status.kept,
                        status.removed,
                    )
            except asyncio.CancelledError:
                raise
            except DATABASE_ERRORS as exc:
                logger.warning(
                    "MCP catalog sync could not reach the database (%s); retrying", describe_database_error(exc)
                )
            except Exception as exc:
                # The type alone: a traceback through a database call would carry the statement's parameters.
                logger.warning("MCP catalog sync failed (%s); retrying in %ss", type(exc).__name__, interval)
