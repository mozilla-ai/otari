"""The MCP server catalog: the sync that fills it from recorded registry pages, who may ask for one, and its reads."""

import asyncio
import io
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.mcp_catalog_exceptions import (
    McpCatalogSyncDisabledError,
    McpCatalogSyncForbiddenError,
    McpCatalogSyncTooSoonError,
)
from gateway.exceptions.organizations_exceptions import NotAuthorizedError, WorkspaceNotFoundError
from gateway.models.mcp_catalog import LOGO_OK, McpCatalogServer, McpCatalogSyncState, McpLogo
from gateway.models.tenancy import Organization, User, Workspace
from gateway.models.tools import WorkspaceMcpServer
from gateway.repositories.mcp_catalog import (
    McpCatalogServerRepository,
    McpCatalogSyncStateRepository,
    McpLogoRepository,
)
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.tools import WorkspaceMcpServerRepository
from gateway.schemas.mcp_catalog import McpCatalogEntry
from gateway.services.mcp_catalog import Logo, McpCatalogService, RegistryError, RegistryPage
from gateway.services.mcp_catalog import _extras as extras_list
from gateway.services.mcp_catalog import _popular as popular_list
from gateway.services.mcp_catalog import _service as catalog_service
from gateway.services.mcp_catalog._registry import entry_from_registry_item
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tools import WorkspaceMcpServerUrls

from .conftest import _to_async_url, build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_CATALOG = f"{API_ROOT}/mcp-catalog"
_PAGES = Path(__file__).parent.parent / "fixtures" / "mcp_registry"
_PAGE_2_CURSOR = "agency.thirdhemisphere/ai-authority-index:1.0.0"
_KEPT = [
    "ac.inference.sh/mcp",
    "ac.snag/snag",
    "academy.apexai/apex-ai-academy",
    "ae.easyvoice/tts",
    "africa.lomi/mcp",
    "agency.thirdhemisphere/ai-authority-index",
    "ai.adadvisor/mcp-server",
]


@pytest.fixture(autouse=True)
def _no_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """No bundled extras, so a synced catalog holds the recorded pages and nothing else."""
    monkeypatch.setattr(catalog_service, "EXTRA_SERVERS", ())


_GRANOLA = extras_list.ExtraServer(
    name="ai.granola/mcp",
    title="Granola",
    description="Meeting notes.",
    url="https://mcp.granola.ai/mcp",
    website_url="https://www.granola.ai",
    icon_url="https://www.granola.ai/icon.png",
)


@pytest.fixture
def _granola_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog_service, "EXTRA_SERVERS", (_GRANOLA,))


def _recorded(name: str) -> RegistryPage:
    page = json.loads((_PAGES / f"{name}.json").read_text())
    return RegistryPage(items=page["servers"], next_cursor=page["metadata"].get("nextCursor"))


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 10, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


class _Registry:
    """Serves the recorded pages by cursor, keeping every cursor it was asked for.

    ``pages`` maps a cursor to the page it answers with, or to the error it raises.
    """

    def __init__(self, pages: dict[str | None, RegistryPage | Exception] | None = None) -> None:
        self.pages: dict[str | None, RegistryPage | Exception] = pages or {
            None: _recorded("page_1"),
            _PAGE_2_CURSOR: _recorded("page_2"),
        }
        self.asked: list[str | None] = []

    async def fetch_page(self, cursor: str | None) -> RegistryPage:
        self.asked.append(cursor)
        answer = self.pages[cursor]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _config(**settings: Any) -> GatewayConfig:
    return GatewayConfig(**{"mcp_catalog_sync": "manual", **settings})


def _png(color: tuple[int, int, int, int] = (0, 0, 0, 255), size: int = 8) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


_PNG = _png()


class _Fetcher:
    """Stands in for the internet: answers each logo candidate from ``images``, recording every URL asked."""

    def __init__(self, images: dict[str, bytes] | None = None) -> None:
        self.images = images or {}
        self.asked: list[str] = []

    def __call__(self) -> "_Fetcher":
        return self

    async def fetch(self, url: str) -> Logo | None:
        self.asked.append(url)
        content = self.images.get(url)
        return Logo.of(content, "image/png") if content is not None else None

    async def aclose(self) -> None:
        return None


def _service(
    db: AsyncSession,
    config: GatewayConfig,
    clock: _Clock,
    fetcher: _Fetcher | None = None,
    budget: catalog_service.LogoLookBudget | None = None,
) -> McpCatalogService:
    uow = UnitOfWork(db)
    return McpCatalogService(
        uow,
        McpCatalogServerRepository(uow),
        McpCatalogSyncStateRepository(uow),
        WorkspaceAccess(db, OrganizationService(db, membership_listener=None)),
        config,
        workspace_servers=WorkspaceMcpServerUrls(WorkspaceMcpServerRepository(uow)),
        fetcher=fetcher or _Fetcher(),
        look_budget=budget or catalog_service.LogoLookBudget(),
        clock=clock,
    )


# An operator's request names nobody: the service reads no user when the caller operates the deployment.
_OPERATOR = cast(User, None)


async def _ask(service: McpCatalogService) -> None:
    """Ask for a sync as an operator, who is never held to the cooldown."""
    await service.request_sync(user=_OPERATOR, workspace_id=None, operates_deployment=True)


async def _names(db: AsyncSession) -> list[str]:
    return sorted((await db.execute(select(McpCatalogServer.name))).scalars().all())


async def _synced(db: AsyncSession, registry: _Registry | None = None) -> McpCatalogService:
    """A service whose catalog holds the recorded pages."""
    service = _service(db, _config(), _Clock())
    await _ask(service)
    status = await service.sync_if_due((registry or _Registry()).fetch_page)
    assert status is not None and status.outcome == "complete"
    return service


async def _organization(db: AsyncSession, *, slug: str = "acme") -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(db: AsyncSession, organization: Organization, *, role: str) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=f"{role.title()} of {organization.slug}", active_organization_id=organization.id
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


async def _workspace(db: AsyncSession, organization: Organization, *, manager: User | None = None) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Default", organization_id=organization.id, created_by_user_id=None
    )
    if manager is not None:
        await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=manager.id, role="admin")
    await db.commit()
    return workspace


def _item(name: str, title: str, url: str) -> dict[str, Any]:
    """A registry item for a live server with one streamable HTTP remote."""
    return {
        "server": {
            "name": name,
            "title": title,
            "description": "",
            "version": "1.0.0",
            "remotes": [{"type": "streamable-http", "url": url}],
        },
        "_meta": {"io.modelcontextprotocol.registry/official": {"status": "active", "isLatest": True}},
    }


def _entry(name: str, url: str, **fields: Any) -> McpCatalogEntry:
    return McpCatalogEntry(
        name=name,
        title=fields.get("title", name),
        description=None,
        url=url,
        requires_token=False,
        website_url=fields.get("website_url"),
        icon_url=fields.get("icon_url"),
        version=None,
        source="registry",
    )


@pytest.mark.asyncio
async def test_a_requested_sync_copies_the_registry_page_by_page(async_db: AsyncSession) -> None:
    service = _service(async_db, _config(), _Clock())
    registry = _Registry()
    await _ask(service)

    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None
    assert (status.outcome, status.error, status.running, status.requested) == ("complete", None, False, False)
    assert (status.pages, status.fetched, status.kept, status.removed, status.server_count) == (2, 9, 7, 0, 7)
    assert status.last_completed_at == status.finished_at
    assert registry.asked == [None, _PAGE_2_CURSOR]
    assert await _names(async_db) == _KEPT


@pytest.mark.asyncio
async def test_nothing_syncs_until_it_is_asked_for_in_manual_mode(async_db: AsyncSession) -> None:
    service = _service(async_db, _config(), _Clock())
    registry = _Registry()

    assert await service.sync_if_due(registry.fetch_page) is None
    assert registry.asked == []
    assert (await service.sync_status()).outcome is None


@pytest.mark.asyncio
async def test_a_deployment_with_the_sync_off_never_contacts_the_registry(async_db: AsyncSession) -> None:
    service = _service(async_db, _config(mcp_catalog_sync="off"), _Clock())
    registry = _Registry()

    with pytest.raises(McpCatalogSyncDisabledError):
        await _ask(service)

    assert await service.sync_if_due(registry.fetch_page) is None
    assert registry.asked == []
    status = await service.sync_status()
    assert (status.mode, status.server_count, status.requested) == ("off", 0, False)
    assert (await service.search(query="", skip=0, limit=25)).count == 0


@pytest.mark.asyncio
async def test_a_scheduled_sync_runs_when_due_and_then_waits_out_its_interval(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(mcp_catalog_sync="scheduled", mcp_catalog_sync_interval_hours=12), clock)
    registry = _Registry()

    first = await service.sync_if_due(registry.fetch_page)
    assert first is not None and first.outcome == "complete"
    assert first.next_sync_at == clock.now + timedelta(hours=12)

    clock.advance(hours=11)
    assert await service.sync_if_due(registry.fetch_page) is None

    clock.advance(hours=2)
    second = await service.sync_if_due(registry.fetch_page)
    assert second is not None and second.outcome == "complete"
    assert registry.asked == [None, _PAGE_2_CURSOR, None, _PAGE_2_CURSOR]


@pytest.mark.asyncio
async def test_a_complete_sync_updates_changed_servers_and_removes_the_ones_gone(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    clock = _Clock()
    clock.advance(hours=1)
    service = _service(async_db, _config(), clock)
    page = _recorded("page_1")
    moved = json.loads(json.dumps(page.items[0]))
    moved["server"]["remotes"][0]["url"] = "https://new.inference.sh/mcp"
    # The second page is gone from the registry, and the first server moved.
    registry = _Registry({None: RegistryPage(items=[moved, *page.items[1:]], next_cursor=None)})

    await _ask(service)
    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None
    assert (status.outcome, status.kept, status.removed, status.server_count) == ("complete", 5, 2, 5)
    assert "africa.lomi/mcp" not in await _names(async_db)
    found = await service.search(query="inference", skip=0, limit=25)
    assert [server.url for server in found.data] == ["https://new.inference.sh/mcp"]


@pytest.mark.asyncio
async def test_a_sync_that_stops_midway_is_partial_and_removes_nothing(async_db: AsyncSession) -> None:
    await _synced(async_db)
    clock = _Clock()
    clock.advance(hours=1)
    service = _service(async_db, _config(), clock)
    only_first = RegistryPage(items=_recorded("page_1").items[:1], next_cursor=_PAGE_2_CURSOR)
    registry = _Registry({None: only_first, _PAGE_2_CURSOR: RegistryError("the registry answered HTTP 502")})

    await _ask(service)
    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None
    assert (status.outcome, status.error) == ("partial", "the registry answered HTTP 502")
    assert (status.pages, status.kept, status.removed, status.running) == (1, 1, 0, False)
    # Six servers were not on the one page read, and a partial crawl cannot tell that from their leaving.
    assert await _names(async_db) == _KEPT
    assert status.last_completed_at == datetime(2026, 11, 10, 9, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_the_next_sync_resumes_a_partial_one_at_its_cursor(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    registry = _Registry({None: _recorded("page_1"), _PAGE_2_CURSOR: RegistryError("network error (ReadTimeout)")})
    await _ask(service)
    partial = await service.sync_if_due(registry.fetch_page)
    assert partial is not None and partial.outcome == "partial"

    clock.advance(minutes=5)
    registry.pages[_PAGE_2_CURSOR] = _recorded("page_2")
    registry.asked.clear()
    await _ask(service)
    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None
    assert registry.asked == [_PAGE_2_CURSOR]
    assert (status.outcome, status.pages, status.fetched, status.kept, status.removed) == ("complete", 2, 9, 7, 0)
    assert status.started_at == datetime(2026, 11, 10, 9, 0, tzinfo=UTC)
    assert await _names(async_db) == _KEPT


@pytest.mark.asyncio
async def test_a_partial_sync_too_old_to_resume_starts_over(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    registry = _Registry({None: _recorded("page_1"), _PAGE_2_CURSOR: RegistryError("the registry answered HTTP 502")})
    await _ask(service)
    await service.sync_if_due(registry.fetch_page)

    clock.advance(hours=7)
    registry.pages[_PAGE_2_CURSOR] = _recorded("page_2")
    registry.asked.clear()
    await _ask(service)
    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None and status.outcome == "complete"
    assert registry.asked == [None, _PAGE_2_CURSOR]
    assert (status.pages, status.fetched) == (2, 9)


@pytest.mark.asyncio
async def test_a_resume_the_registry_refuses_starts_over_next_time(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    registry = _Registry({None: _recorded("page_1"), _PAGE_2_CURSOR: RegistryError("the registry answered HTTP 400")})
    await _ask(service)
    await service.sync_if_due(registry.fetch_page)

    clock.advance(minutes=5)
    await _ask(service)
    refused = await service.sync_if_due(registry.fetch_page)
    assert refused is not None and refused.outcome == "partial"

    clock.advance(minutes=5)
    registry.pages[_PAGE_2_CURSOR] = _recorded("page_2")
    registry.asked.clear()
    await _ask(service)
    await service.sync_if_due(registry.fetch_page)

    assert registry.asked == [None, _PAGE_2_CURSOR]


@pytest.mark.asyncio
async def test_a_sync_that_reads_nothing_fails_and_leaves_the_catalog(async_db: AsyncSession) -> None:
    await _synced(async_db)
    clock = _Clock()
    clock.advance(hours=1)
    service = _service(async_db, _config(), clock)

    await _ask(service)
    down = await service.sync_if_due(_Registry({None: RegistryError("the registry did not answer")}).fetch_page)
    clock.advance(hours=1)
    await _ask(service)
    empty = await service.sync_if_due(_Registry({None: RegistryPage(items=[], next_cursor=None)}).fetch_page)

    assert down is not None and (down.outcome, down.error) == ("failed", "the registry did not answer")
    assert empty is not None and empty.outcome == "failed"
    assert empty.error is not None and "no server Otari can connect to" in empty.error
    assert (empty.removed, empty.server_count) == (0, 7)
    assert await _names(async_db) == _KEPT


@pytest.mark.asyncio
async def test_a_sync_stops_at_its_page_bound_and_says_so(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(catalog_service, "MAX_PAGES", 1)
    service = _service(async_db, _config(), _Clock())
    registry = _Registry()

    await _ask(service)
    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None
    assert (status.outcome, status.error, status.pages) == ("partial", "stopped after 1 pages", 1)
    assert registry.asked == [None]


@pytest.mark.asyncio
async def test_a_sync_stops_at_its_time_bound(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)

    class _SlowRegistry(_Registry):
        async def fetch_page(self, cursor: str | None) -> RegistryPage:
            clock.advance(minutes=31)
            return await super().fetch_page(cursor)

    registry = _SlowRegistry()
    await _ask(service)
    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None
    assert (status.outcome, status.error, status.pages) == ("partial", "stopped after 30 minutes", 1)


@pytest.mark.asyncio
async def test_a_running_sync_is_not_started_twice_and_a_dead_one_is_taken_over(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    registry = _Registry()
    await _ask(service)

    class _Crash(BaseException):
        pass

    async def dies_on_the_second_page(cursor: str | None) -> RegistryPage:
        if cursor is not None:
            raise _Crash
        return await registry.fetch_page(cursor)

    with pytest.raises(_Crash):
        await service.sync_if_due(dies_on_the_second_page)

    clock.advance(minutes=2)
    await _ask(service)
    assert (await service.sync_status()).running is True
    assert await service.sync_if_due(registry.fetch_page) is None

    clock.advance(minutes=9)
    assert (await service.sync_status()).running is False
    registry.asked.clear()
    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None and status.outcome == "complete"
    assert registry.asked == [_PAGE_2_CURSOR]
    assert await _names(async_db) == _KEPT


@pytest.mark.asyncio
async def test_a_last_page_that_lists_a_fraction_of_the_catalog_removes_nothing(async_db: AsyncSession) -> None:
    await _synced(async_db)
    clock = _Clock()
    clock.advance(hours=1)
    service = _service(async_db, _config(), clock)
    # One server and no cursor: what a registry cut short, or changed under the gateway, would answer.
    cut_short = _Registry({None: RegistryPage(items=_recorded("page_1").items[:1], next_cursor=None)})

    await _ask(service)
    status = await service.sync_if_due(cut_short.fetch_page)

    assert status is not None
    assert (status.outcome, status.removed, status.server_count) == ("partial", 0, 7)
    assert status.error is not None and "1 of the 7 servers" in status.error
    assert await _names(async_db) == _KEPT
    # It read to the end, so there is no page to resume at.
    registry = _Registry()
    clock.advance(minutes=5)
    await _ask(service)
    await service.sync_if_due(registry.fetch_page)
    assert registry.asked == [None, _PAGE_2_CURSOR]


@pytest.mark.asyncio
async def test_new_servers_on_the_last_page_do_not_hide_the_loss_of_the_old_ones(async_db: AsyncSession) -> None:
    await _synced(async_db)
    clock = _Clock()
    clock.advance(hours=1)
    service = _service(async_db, _config(), clock)
    # One of the seven servers the catalog holds, beside eight it has never seen: six of seven are gone.
    newcomers = [_item(f"com.new/server-{n}", f"New {n}", f"https://new{n}.example.com/mcp") for n in range(8)]
    swapped = _Registry({None: RegistryPage(items=[_recorded("page_1").items[0], *newcomers], next_cursor=None)})

    await _ask(service)
    status = await service.sync_if_due(swapped.fetch_page)

    assert status is not None
    assert (status.outcome, status.removed, status.server_count) == ("partial", 0, 15)
    assert status.error is not None and "1 of the 7 servers" in status.error
    assert set(_KEPT) <= set(await _names(async_db))


@pytest.mark.asyncio
async def test_a_cursor_the_registry_repeats_stops_the_sync_and_the_next_one_starts_over(
    async_db: AsyncSession,
) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    stuck = RegistryPage(items=_recorded("page_2").items, next_cursor=_PAGE_2_CURSOR)
    registry = _Registry({None: _recorded("page_1"), _PAGE_2_CURSOR: stuck})

    await _ask(service)
    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None
    assert status.outcome == "partial"
    assert status.error == "the registry answered a page with the cursor it was asked for"
    # The page that repeated its cursor is not stored.
    assert (status.pages, status.removed) == (1, 0)
    assert registry.asked == [None, _PAGE_2_CURSOR]

    clock.advance(minutes=5)
    registry.pages[_PAGE_2_CURSOR] = _recorded("page_2")
    registry.asked.clear()
    await _ask(service)
    await service.sync_if_due(registry.fetch_page)

    assert registry.asked == [None, _PAGE_2_CURSOR]


@pytest.mark.asyncio
async def test_a_page_the_database_refuses_fails_the_sync_and_stores_none_of_it(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    def lets_a_nul_through(item: object) -> McpCatalogEntry | None:
        entry = entry_from_registry_item(item)
        if entry is not None and entry.name == "ac.snag/snag":
            return entry.model_copy(update={"description": "nul \x00 here"})
        return entry

    # The filter drops a NUL before it gets here; this is the sync's answer to one that slips through.
    monkeypatch.setattr(catalog_service, "entry_from_registry_item", lets_a_nul_through)
    service = _service(async_db, _config(), _Clock())

    await _ask(service)
    status = await service.sync_if_due(_Registry().fetch_page)

    assert status is not None
    assert (status.outcome, status.error) == ("failed", "a page could not be stored in the database")
    assert (status.pages, status.kept, status.running) == (0, 0, False)
    assert await _names(async_db) == []


@pytest.mark.asyncio
async def test_a_run_taken_for_dead_stops_without_touching_the_run_that_replaced_it(async_db: AsyncSession) -> None:
    clock = _Clock()
    slow = _service(async_db, _config(), clock)
    await _ask(slow)
    recorded = _Registry()

    async def stalls_then_answers(cursor: str | None) -> RegistryPage:
        if cursor is None:
            # While this page hangs, the heartbeat goes stale and another replica runs the sync to the end.
            clock.advance(minutes=11)
            replacement = await slow.sync_if_due(_Registry().fetch_page)
            assert replacement is not None and replacement.outcome == "complete"
            clock.advance(minutes=1)
        return await recorded.fetch_page(cursor)

    assert await slow.sync_if_due(stalls_then_answers) is None

    status = await slow.sync_status()
    assert (status.outcome, status.running, status.pages, status.kept) == ("complete", False, 2, 7)
    assert status.started_at == datetime(2026, 11, 10, 9, 11, tzinfo=UTC)
    # The stale run asked for its first page and stored nothing of it.
    assert recorded.asked == [None]


@pytest.mark.asyncio
async def test_a_sync_cancelled_at_shutdown_gives_its_claim_back(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    registry = _Registry()
    await _ask(service)

    async def cancelled_on_the_second_page(cursor: str | None) -> RegistryPage:
        if cursor is not None:
            raise asyncio.CancelledError
        return await registry.fetch_page(cursor)

    with pytest.raises(asyncio.CancelledError):
        await service.sync_if_due(cancelled_on_the_second_page)

    status = await service.sync_status()
    assert (status.running, status.outcome, status.pages) == (False, "partial", 1)
    assert status.error == "the gateway stopped before the sync finished"
    # The next one picks it up at once, where it stopped.
    registry.asked.clear()
    await _ask(service)
    resumed = await service.sync_if_due(registry.fetch_page)
    assert resumed is not None and resumed.outcome == "complete"
    assert registry.asked == [_PAGE_2_CURSOR]


@pytest.mark.asyncio
async def test_a_request_made_before_the_sync_was_switched_off_is_not_reported_as_pending(
    async_db: AsyncSession,
) -> None:
    await _ask(_service(async_db, _config(), _Clock()))

    status = await _service(async_db, _config(mcp_catalog_sync="off"), _Clock()).sync_status()

    assert (status.mode, status.requested) == ("off", False)


@pytest.mark.asyncio
async def test_search_matches_title_name_and_description_and_pages(async_db: AsyncSession) -> None:
    service = await _synced(async_db)

    everything = await service.search(query="", skip=0, limit=3)
    rest = await service.search(query="", skip=3, limit=25)
    by_name = await service.search(query="AFRICA.lomi", skip=0, limit=25)
    by_description = await service.search(query="fact-checked", skip=0, limit=25)
    nothing = await service.search(query="100%_", skip=0, limit=25)

    assert everything.count == 7
    titles = [server.title for server in [*everything.data, *rest.data]]
    assert titles == sorted(titles, key=str.lower)
    assert len(titles) == 7
    assert [server.name for server in by_name.data] == ["africa.lomi/mcp"]
    assert [server.name for server in by_description.data] == ["academy.apexai/apex-ai-academy"]
    assert (nothing.count, nothing.data) == (0, [])


@pytest.mark.asyncio
async def test_a_match_in_the_title_outranks_one_in_the_name_or_description(async_db: AsyncSession) -> None:
    service = await _synced(async_db)

    found = await service.search(query="ai", skip=0, limit=25)

    in_title = ["ai" in server.title.lower() for server in found.data]
    # Matches in the title lead, and the rest matched on their name or description alone.
    assert in_title == sorted(in_title, reverse=True)
    assert True in in_title and False in in_title
    assert [server.title for server in found.data if "ai" in server.title.lower()] == [
        "Apex AI Academy",
        "Third Hemisphere AI Authority Index",
    ]
    starts = await service.search(query="apex", skip=0, limit=25)
    assert [server.title for server in starts.data][0] == "Apex AI Academy"


@pytest.mark.asyncio
async def test_a_makers_own_server_leads_the_matches_of_equal_rank(async_db: AsyncSession) -> None:
    service = _service(async_db, _config(), _Clock())
    registry = _Registry(
        {
            None: RegistryPage(
                items=[
                    _item("io.github.pipeworx-io/figma", "Figma", "https://gateway.pipeworx.io/figma/mcp"),
                    _item("com.figma.mcp/mcp", "Figma MCP Server", "https://mcp.figma.com/mcp"),
                    _item("com.example/afigma", "A Figma helper", "https://mcp.example.com/mcp"),
                ],
                next_cursor=None,
            )
        }
    )
    await _ask(service)
    await service.sync_if_due(registry.fetch_page)

    found = await service.search(query="figma", skip=0, limit=25)

    # Both titles start with the query; Figma's own comes first though "Figma" sorts ahead of it.
    assert [(server.title, server.publisher_kind) for server in found.data] == [
        ("Figma MCP Server", "vendor"),
        ("Figma", "community"),
        ("A Figma helper", "community"),
    ]


@pytest.mark.asyncio
async def test_servers_a_workspace_already_has_are_left_out_for_whoever_can_see_it(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    acme = await _organization(async_db)
    member = await _member(async_db, acme, role="member")
    outsider = await _member(async_db, await _organization(async_db, slug="globex"), role="owner")
    workspace = await _workspace(async_db, acme, manager=member)
    # Stored with a trailing slash the catalog's URL does not have.
    async_db.add(WorkspaceMcpServer(workspace_id=workspace.id, name="Snag", url="https://mcp.snag.ac/mcp/"))
    await async_db.commit()

    everything = await service.search(query="", skip=0, limit=25)
    left = await service.search(query="", skip=0, limit=25, user=member, workspace_id=workspace.id)

    assert (everything.count, left.count) == (7, 6)
    assert "ac.snag/snag" not in [server.name for server in left.data]
    with pytest.raises(WorkspaceNotFoundError):
        await service.search(query="", skip=0, limit=25, user=outsider, workspace_id=workspace.id)
    with pytest.raises(WorkspaceNotFoundError):
        await service.popular(user=outsider, workspace_id=workspace.id)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_popular_list")
async def test_the_popular_shelf_leaves_out_what_the_workspace_has(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    acme = await _organization(async_db)
    member = await _member(async_db, acme, role="member")
    workspace = await _workspace(async_db, acme, manager=member)
    async_db.add(WorkspaceMcpServer(workspace_id=workspace.id, name="Snag", url="https://mcp.snag.ac/mcp"))
    await async_db.commit()

    shelf = await service.popular(user=member, workspace_id=workspace.id)

    assert [server.name for server in shelf.data] == ["africa.lomi/mcp"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_granola_extra")
async def test_a_bundled_extra_is_listed_searched_and_kept_by_every_sync(async_db: AsyncSession) -> None:
    fetcher = _Fetcher({"https://www.granola.ai/icon.png": _PNG})
    clock = _Clock()
    service = _service(async_db, _config(), clock, fetcher)
    await _ask(service)
    first = await service.sync_if_due(_Registry().fetch_page)
    clock.advance(hours=2)
    await _ask(service)
    second = await service.sync_if_due(_Registry().fetch_page)

    assert first is not None and second is not None
    # Counted in the catalog, not among what the registry gave, and not removed as unseen.
    assert (second.outcome, second.kept, second.removed, second.server_count) == ("complete", 7, 0, 8)
    found = (await service.search(query="granola", skip=0, limit=25)).data
    assert [(server.name, server.source, server.publisher_kind) for server in found] == [
        ("ai.granola/mcp", "curated", "vendor")
    ]
    assert await service.logo("ai.granola/mcp") is not None
    assert fetcher.asked == ["https://www.granola.ai/icon.png"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_granola_extra")
async def test_an_extra_gives_way_once_the_registry_lists_its_url(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    page = _recorded("page_1")
    listed = RegistryPage(
        items=[*page.items, _item("ai.granola/notes", "Granola", "https://mcp.granola.ai/mcp")], next_cursor=None
    )
    await _ask(service)
    await service.sync_if_due(_Registry({None: listed}).fetch_page)
    assert {"ai.granola/mcp", "ai.granola/notes"} <= set(await _names(async_db))

    clock.advance(hours=2)
    await _ask(service)
    await service.sync_if_due(_Registry({None: listed}).fetch_page)

    names = await _names(async_db)
    assert "ai.granola/notes" in names and "ai.granola/mcp" not in names


@pytest.mark.asyncio
async def test_an_extra_taken_off_the_list_leaves_with_the_next_complete_sync(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(catalog_service, "EXTRA_SERVERS", (_GRANOLA,))
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    await _ask(service)
    await service.sync_if_due(_Registry().fetch_page)
    assert "ai.granola/mcp" in await _names(async_db)

    monkeypatch.setattr(catalog_service, "EXTRA_SERVERS", ())
    clock.advance(hours=2)
    await _ask(service)
    await service.sync_if_due(_Registry().fetch_page)

    assert await _names(async_db) == _KEPT


@pytest.fixture
def _popular_list(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A Popular list naming two recorded servers, and one the catalog does not hold."""
    monkeypatch.setattr(
        catalog_service,
        "POPULAR_SERVERS",
        (
            popular_list.PopularServer("ac.snag/snag", "https://mcp.snag.ac/mcp"),
            popular_list.PopularServer("com.example/missing", "https://mcp.example.com/mcp"),
            popular_list.PopularServer("africa.lomi/mcp", "https://mcp.lomi.africa/mcp"),
        ),
    )
    yield


@pytest.mark.asyncio
@pytest.mark.usefixtures("_popular_list")
async def test_the_popular_shelf_lists_what_the_catalog_holds_in_the_lists_order(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    urls = {row.name: row.url for row in (await service.search(query="", skip=0, limit=25)).data}
    assert urls["ac.snag/snag"] == "https://mcp.snag.ac/mcp"
    assert urls["africa.lomi/mcp"] == "https://mcp.lomi.africa/mcp"

    shelf = await service.popular()

    assert [server.name for server in shelf.data] == ["ac.snag/snag", "africa.lomi/mcp"]
    assert shelf.count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("_popular_list")
async def test_a_popular_url_listed_twice_shows_the_expected_name(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    async with UnitOfWork(async_db) as uow:
        await McpCatalogServerRepository(uow).upsert(
            [_entry("aa.copycat/snag", "https://mcp.snag.ac/mcp", title="Copycat")], seen_at=_Clock().now
        )

    shelf = await service.popular()

    assert [server.name for server in shelf.data][0] == "ac.snag/snag"


@pytest.mark.asyncio
async def test_the_popular_shelf_is_empty_before_a_sync(async_db: AsyncSession) -> None:
    assert (await _service(async_db, _config(), _Clock()).popular()).data == []


async def _store(db: AsyncSession, *entries: McpCatalogEntry) -> None:
    async with UnitOfWork(db) as uow:
        await McpCatalogServerRepository(uow).upsert(list(entries), seen_at=_Clock().now)


@pytest.mark.asyncio
async def test_logos_are_matched_by_url_either_spelling_and_fall_back_to_the_host(async_db: AsyncSession) -> None:
    clock = _Clock()
    fetcher = _Fetcher({"https://one.example.com/i.png": _PNG, "https://elsewhere.org/favicon.ico": _PNG})
    service = _service(async_db, _config(), clock, fetcher)
    await _store(
        async_db,
        _entry("com.example/one", "https://one.example.com/mcp", icon_url="https://one.example.com/i.png"),
        _entry("com.example/two", "https://two.example.com/mcp/"),
    )

    before = await service.logo_refs(["https://one.example.com/mcp/", "https://mcp.elsewhere.org/mcp", "not a url"])
    one = await service.logo_for_url("https://one.example.com/mcp/")
    host = await service.logo_for_url("https://mcp.elsewhere.org/mcp")
    after = await service.logo_refs(["https://one.example.com/mcp", "https://mcp.elsewhere.org/mcp"])

    # Nothing was looked for until a logo was asked for; a list starts no look.
    assert set(before) == {"https://one.example.com/mcp/", "https://mcp.elsewhere.org/mcp"}
    assert all(ref.pending and ref.etag is None for ref in before.values())
    assert one is not None and host is not None
    assert fetcher.asked[0] == "https://one.example.com/i.png"
    # A server known only by its URL is looked for on its owner's site first.
    assert fetcher.asked[1:] == ["https://elsewhere.org/apple-touch-icon.png", "https://elsewhere.org/favicon.ico"]
    assert after["https://one.example.com/mcp"].etag == one.etag
    assert after["https://mcp.elsewhere.org/mcp"].url("/logo") == f"/logo?v={host.etag}"


@pytest.mark.asyncio
async def test_a_logo_is_looked_for_once_and_a_missing_one_is_remembered(async_db: AsyncSession) -> None:
    clock = _Clock()
    fetcher = _Fetcher()
    service = _service(async_db, _config(), clock, fetcher)
    await _store(async_db, _entry("com.example/one", "https://one.example.com/mcp", website_url="https://one.example"))

    assert await service.logo("com.example/one") is None
    asked = len(fetcher.asked)
    assert asked > 0
    assert await service.logo("com.example/one") is None
    assert len(fetcher.asked) == asked
    refs = await service.logo_refs(["https://one.example.com/mcp"])
    assert refs["https://one.example.com/mcp"].url("/logo") is None

    fetcher.images["https://example.com/favicon.ico"] = _PNG
    clock.advance(days=8)
    found = await service.logo("com.example/one")

    assert found is not None and found.content == _PNG and found.content_type == "image/png"
    assert len(fetcher.asked) > asked


@pytest.mark.asyncio
async def test_a_found_logo_is_kept_when_a_later_look_finds_nothing(async_db: AsyncSession) -> None:
    clock = _Clock()
    fetcher = _Fetcher({"https://example.com/favicon.ico": _PNG})
    service = _service(async_db, _config(), clock, fetcher)
    await _store(async_db, _entry("com.example/one", "https://one.example.com/mcp"))
    first = await service.logo("com.example/one")

    fetcher.images.clear()
    clock.advance(days=8)
    # Due for another look: the bare URL, which no browser keeps for a week, so the next view starts it.
    due = await service.logo_refs(["https://one.example.com/mcp"])

    assert first is not None
    assert due["https://one.example.com/mcp"].url("/logo") == "/logo"
    assert await service.logo("com.example/one") == first
    kept = await service.logo_refs(["https://one.example.com/mcp"])
    assert kept["https://one.example.com/mcp"].url("/logo") == f"/logo?v={first.etag}"


@pytest.mark.asyncio
async def test_a_bundled_logo_needs_no_fetch_and_an_unlisted_name_reaches_nothing(async_db: AsyncSession) -> None:
    fetcher = _Fetcher()
    service = _service(async_db, _config(), _Clock(), fetcher)
    await _store(async_db, _entry("com.stripe/mcp", "https://mcp.stripe.com", title="Stripe"))

    stripe = await service.logo("com.stripe/mcp")
    unknown = await service.logo("com.unknown/mcp")

    assert stripe is not None and stripe.content_type == "image/svg+xml"
    assert unknown is None
    assert fetcher.asked == []


@pytest.mark.asyncio
async def test_a_look_past_the_process_budget_fetches_nothing(async_db: AsyncSession) -> None:
    fetcher = _Fetcher()
    budget = catalog_service.LogoLookBudget(concurrent=4, per_minute=1)
    service = _service(async_db, _config(), _Clock(), fetcher, budget)
    await _store(
        async_db,
        _entry("com.example/one", "https://one.example.com/mcp"),
        _entry("com.example/two", "https://two.example.com/mcp"),
    )

    await service.logo("com.example/one")
    asked = len(fetcher.asked)
    await service.logo("com.example/two")

    assert asked > 0
    assert len(fetcher.asked) == asked
    # Refused, not recorded as missing: the next view after the minute looks for it.
    refs = await service.logo_refs(["https://two.example.com/mcp"])
    assert refs["https://two.example.com/mcp"].pending is True


def test_the_look_budget_counts_running_looks_and_a_minute_of_starts() -> None:
    now = [0.0]
    budget = catalog_service.LogoLookBudget(concurrent=1, per_minute=2, clock=lambda: now[0])

    assert budget.start("a") is True
    assert budget.start("b") is False
    budget.finish("a")
    assert budget.start("a") is True
    budget.finish("a")
    assert budget.start("c") is False
    now[0] = 61.0
    assert budget.start("c") is True


@pytest.mark.asyncio
async def test_catalog_entries_carry_their_logo_tone_and_publisher(async_db: AsyncSession) -> None:
    service = _service(async_db, _config(), _Clock())
    await _store(
        async_db,
        _entry("com.stripe/mcp", "https://mcp.stripe.com", title="Stripe"),
        _entry("io.github.pipeworx-io/intercom", "https://gateway.pipeworx.io/intercom/mcp", title="Intercom"),
    )

    found = {server.name: server for server in (await service.search(query="", skip=0, limit=10)).data}

    stripe = found["com.stripe/mcp"]
    assert stripe.publisher_kind == "vendor"
    assert stripe.logo_url is not None and stripe.logo_url.startswith(f"{API_ROOT}/mcp-catalog/logos/com.stripe/mcp?v=")
    # Stripe's mark is mid-bright, so it reads on both themes untiled.
    assert stripe.logo_tone is None
    community = found["io.github.pipeworx-io/intercom"]
    assert community.publisher_kind == "community"
    assert community.logo_url == f"{API_ROOT}/mcp-catalog/logos/io.github.pipeworx-io/intercom"


@pytest.mark.asyncio
async def test_a_workspace_admin_may_ask_for_a_sync_and_a_plain_member_may_not(async_db: AsyncSession) -> None:
    service = _service(async_db, _config(), _Clock())
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="member")
    member = await _member(async_db, organization, role="member")
    workspace = await _workspace(async_db, organization, manager=admin)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=member.id, role="member")
    await async_db.commit()

    with pytest.raises(NotAuthorizedError):
        await service.request_sync(user=member, workspace_id=workspace.id, operates_deployment=False)
    with pytest.raises(McpCatalogSyncForbiddenError):
        await service.request_sync(user=admin, workspace_id=None, operates_deployment=False)
    status = await service.request_sync(user=admin, workspace_id=workspace.id, operates_deployment=False)

    assert status.requested is True


@pytest.mark.asyncio
async def test_another_organizations_workspace_does_not_let_a_caller_sync(async_db: AsyncSession) -> None:
    service = _service(async_db, _config(), _Clock())
    acme = await _organization(async_db)
    globex = await _organization(async_db, slug="globex")
    outsider = await _member(async_db, globex, role="owner")
    workspace = await _workspace(async_db, acme)

    with pytest.raises(WorkspaceNotFoundError):
        await service.request_sync(user=outsider, workspace_id=workspace.id, operates_deployment=False)


@pytest.mark.asyncio
async def test_a_workspace_admin_waits_out_the_cooldown_and_an_operator_does_not(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner")
    workspace_id = (await _workspace(async_db, organization)).id
    await _ask(service)
    assert (await service.sync_if_due(_Registry().fetch_page)) is not None

    status = await service.sync_status()
    assert status.next_request_at == clock.now + catalog_service.REQUEST_COOLDOWN_AFTER_COMPLETE
    assert status.is_stale is False
    await async_db.refresh(owner)
    with pytest.raises(McpCatalogSyncTooSoonError):
        await service.request_sync(user=owner, workspace_id=workspace_id, operates_deployment=False)
    assert (await service.request_sync(user=_OPERATOR, workspace_id=None, operates_deployment=True)).requested

    clock.advance(hours=1, seconds=1)
    await service.sync_if_due(_Registry().fetch_page)
    clock.advance(hours=1, seconds=1)
    assert (await service.sync_status()).next_request_at is None
    await async_db.refresh(owner)
    assert (await service.request_sync(user=owner, workspace_id=workspace_id, operates_deployment=False)).requested


@pytest.mark.asyncio
async def test_asking_while_a_sync_is_pending_changes_nothing(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner")
    workspace = await _workspace(async_db, organization)
    await _ask(service)

    status = await service.request_sync(user=owner, workspace_id=workspace.id, operates_deployment=False)

    assert (status.requested, status.next_request_at) == (True, None)


@pytest.mark.asyncio
async def test_a_failed_sync_holds_admins_back_only_briefly(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    await _ask(service)
    await service.sync_if_due(_Registry({None: RegistryError("down")}).fetch_page)

    status = await service.sync_status()

    assert status.outcome == "failed"
    assert status.next_request_at == clock.now + catalog_service.REQUEST_COOLDOWN_AFTER_FAILURE
    assert status.is_stale is True


@pytest.mark.asyncio
async def test_the_catalog_goes_stale_a_day_after_its_last_complete_sync(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    await _ask(service)
    await service.sync_if_due(_Registry().fetch_page)

    clock.advance(hours=23)
    fresh = await service.sync_status()
    clock.advance(hours=2)
    stale = await service.sync_status()

    assert (fresh.is_stale, stale.is_stale) == (False, True)


@pytest.mark.asyncio
async def test_two_workers_polling_at_once_start_one_sync(async_db: AsyncSession, postgres_url: str) -> None:
    clock = _Clock()
    await _ask(_service(async_db, _config(), clock))
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def poll() -> bool:
        # Each worker claims and commits in a session of its own, so the loser waits on the winner's row lock.
        async with sessions() as session:
            claimed = await McpCatalogSyncStateRepository(session).claim(
                clock.now, scheduled=False, stale_before=clock.now, next_sync_at=clock.now
            )
            await session.commit()
            return claimed

    try:
        claims = await asyncio.gather(poll(), poll())
    finally:
        await engine.dispose()

    assert sorted(claims) == [False, True]
    row = (await async_db.execute(select(McpCatalogSyncState).execution_options(populate_existing=True))).scalar_one()
    assert (row.running, row.sync_requested) == (True, False)


@pytest.mark.asyncio
async def test_two_sessions_creating_the_sync_state_at_once_end_with_its_one_row(
    async_db: AsyncSession, postgres_url: str
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as first, sessions() as second:
            created = await McpCatalogSyncStateRepository(first).get_or_create()
            # The second finds no committed row, so its insert waits on the first's until that commits.
            racing = asyncio.create_task(McpCatalogSyncStateRepository(second).get_or_create())
            await asyncio.sleep(0.2)
            assert not racing.done()
            await first.commit()
            read = await racing
            await second.commit()
    finally:
        await engine.dispose()

    assert created.id == read.id
    rows = (await async_db.execute(select(McpCatalogSyncState))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_two_sessions_storing_one_new_logo_at_once_end_with_its_one_row(
    async_db: AsyncSession, postgres_url: str
) -> None:
    now = _Clock().now
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as first, sessions() as second:
            await McpLogoRepository(first).store_not_found("host:mcp.example.com", fetched_at=now, retry_at=now)
            # The second finds no committed row, so its insert waits on the first's, then updates that row.
            racing = asyncio.create_task(
                McpLogoRepository(second).store_found(
                    "host:mcp.example.com",
                    content=b"<svg/>",
                    content_type="image/svg+xml",
                    etag="e1",
                    luminance=None,
                    source_url="https://mcp.example.com/icon.svg",
                    fetched_at=now,
                    retry_at=now,
                )
            )
            await asyncio.sleep(0.2)
            assert not racing.done()
            await first.commit()
            await racing
            await second.commit()
    finally:
        await engine.dispose()

    rows = (await async_db.execute(select(McpLogo))).scalars().all()
    assert [(row.key, row.status, row.content) for row in rows] == [("host:mcp.example.com", LOGO_OK, b"<svg/>")]


@pytest.mark.asyncio
async def test_a_request_read_before_a_worker_claims_the_sync_does_not_queue_a_second_one(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    await service.sync_status()
    read_state = McpCatalogSyncStateRepository.get_or_create

    async def claimed_right_after_the_read(self: McpCatalogSyncStateRepository) -> McpCatalogSyncState:
        row = await read_state(self)
        monkeypatch.setattr(McpCatalogSyncStateRepository, "get_or_create", read_state)
        await self.claim(clock.now, scheduled=True, stale_before=clock.now, next_sync_at=clock.now)
        return row

    monkeypatch.setattr(McpCatalogSyncStateRepository, "get_or_create", claimed_right_after_the_read)
    status = await service.request_sync(user=_OPERATOR, workspace_id=None, operates_deployment=True)

    assert (status.running, status.requested) == (True, False)
    row = (await async_db.execute(select(McpCatalogSyncState).execution_options(populate_existing=True))).scalar_one()
    assert (row.running, row.sync_requested) == (True, False)


def test_the_catalog_is_empty_and_searchable_before_any_sync(client: TestClient) -> None:
    servers = client.get(f"{_CATALOG}/servers", headers=_MASTER)
    popular = client.get(f"{_CATALOG}/popular", headers=_MASTER)
    status = client.get(f"{_CATALOG}/sync", headers=_MASTER)

    assert servers.status_code == 200
    assert servers.json() == {"data": [], "count": 0}
    assert popular.json() == {"data": [], "count": 0}
    assert status.status_code == 200
    body = status.json()
    assert (body["mode"], body["running"], body["outcome"], body["server_count"]) == ("manual", False, None, 0)


@pytest.mark.parametrize("path", ["servers", "popular"])
def test_a_read_that_names_a_workspace_nobody_has_is_a_404(client: TestClient, path: str) -> None:
    response = client.get(f"{_CATALOG}/{path}?workspace_id=11111111-1111-1111-1111-111111111111", headers=_MASTER)

    assert response.status_code == 404


def test_sync_now_is_accepted_and_shows_as_requested(client: TestClient) -> None:
    response = client.post(f"{_CATALOG}/sync", headers=_MASTER)

    assert response.status_code == 202
    assert response.json()["requested"] is True
    assert client.get(f"{_CATALOG}/sync", headers=_MASTER).json()["requested"] is True


def test_sync_now_is_refused_while_the_sync_is_off(test_config: GatewayConfig, clean_database: None) -> None:
    for client in build_test_client(test_config.model_copy(update={"mcp_catalog_sync": "off"})):
        response = client.post(f"{_CATALOG}/sync", headers=_MASTER)
        status = client.get(f"{_CATALOG}/sync", headers=_MASTER)
        servers = client.get(f"{_CATALOG}/servers", headers=_MASTER)

    assert response.status_code == 409
    assert "mcp_catalog_sync" in response.json()["detail"]
    assert (status.json()["mode"], status.json()["requested"]) == ("off", False)
    assert servers.status_code == 200


def test_the_feature_can_be_switched_off_whole(test_config: GatewayConfig, clean_database: None) -> None:
    for client in build_test_client(test_config.model_copy(update={"mcp_catalog_enabled": False})):
        assert client.get(f"{_CATALOG}/servers", headers=_MASTER).status_code == 404
        assert client.get(f"{_CATALOG}/sync", headers=_MASTER).status_code == 404
        assert client.get(f"{_CATALOG}/popular", headers=_MASTER).status_code == 404
        assert "mcp_catalog" not in client.get(f"{API_ROOT}/bootstrap").json()["surfaces"]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", f"{_CATALOG}/servers"),
        ("GET", f"{_CATALOG}/sync"),
        ("POST", f"{_CATALOG}/sync"),
        ("GET", f"{_CATALOG}/popular"),
        ("GET", f"{_CATALOG}/logos/com.stripe/mcp"),
    ],
)
def test_a_caller_without_a_credential_is_refused(client: TestClient, method: str, path: str) -> None:
    assert client.request(method, path).status_code == 401


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "skip=-1", f"q={'x' * 201}"])
def test_a_search_out_of_bounds_is_refused(client: TestClient, query: str) -> None:
    assert client.get(f"{_CATALOG}/servers?{query}", headers=_MASTER).status_code == 422


def test_an_unlisted_catalog_logo_is_a_404_nobody_caches(client: TestClient) -> None:
    response = client.get(f"{_CATALOG}/logos/com.unknown/mcp", headers=_MASTER)

    assert response.status_code == 404
    assert "no-store" in response.headers["cache-control"]
    assert response.content == b""


@pytest.mark.asyncio
async def test_no_logo_is_looked_for_while_the_sync_is_off(async_db: AsyncSession) -> None:
    fetcher = _Fetcher({"https://example.com/favicon.ico": _PNG})
    service = _service(async_db, _config(mcp_catalog_sync="off"), _Clock(), fetcher)
    await _store(async_db, _entry("com.example/one", "https://one.example.com/mcp"))

    assert await service.logo("com.example/one") is None
    assert await service.logo_for_url("https://mcp.elsewhere.org/mcp") is None
    assert fetcher.asked == []
