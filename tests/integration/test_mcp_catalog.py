"""The MCP server catalog: the sync that fills it from recorded registry pages, its search, and the featured shelf."""

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.mcp_catalog_exceptions import (
    McpCatalogFeaturedLimitReachedError,
    McpCatalogFeaturedNotFoundError,
    McpCatalogServerNotFoundError,
    McpCatalogSyncDisabledError,
)
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.models.mcp_catalog import McpCatalogFeatured, McpCatalogServer, McpCatalogSyncState
from gateway.models.tenancy import Organization, User
from gateway.repositories.mcp_catalog import (
    McpCatalogFeaturedRepository,
    McpCatalogServerRepository,
    McpCatalogSyncStateRepository,
)
from gateway.repositories.tenancy import OrganizationMemberRepository, OrganizationRepository, UserRepository
from gateway.schemas.mcp_catalog import McpCatalogEntry, McpCatalogFeatureRequest
from gateway.services.mcp_catalog import McpCatalogService, RegistryError, RegistryPage
from gateway.services.mcp_catalog import _service as catalog_service
from gateway.services.tenancy import OrganizationService

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_CATALOG = f"{API_ROOT}/mcp-catalog"
_FEATURED = f"{API_ROOT}/organizations/me/mcp-catalog/featured"
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


def _service(db: AsyncSession, config: GatewayConfig, clock: _Clock) -> McpCatalogService:
    uow = UnitOfWork(db)
    return McpCatalogService(
        uow,
        McpCatalogServerRepository(uow),
        McpCatalogFeaturedRepository(uow),
        McpCatalogSyncStateRepository(uow),
        OrganizationService(db, membership_listener=None),
        config,
        clock=clock,
    )


async def _names(db: AsyncSession) -> list[str]:
    return sorted((await db.execute(select(McpCatalogServer.name))).scalars().all())


async def _synced(db: AsyncSession, registry: _Registry | None = None) -> McpCatalogService:
    """A service whose catalog holds the recorded pages."""
    service = _service(db, _config(), _Clock())
    await service.request_sync()
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


async def _server_id(db: AsyncSession, name: str) -> uuid.UUID:
    return (await db.execute(select(McpCatalogServer.id).where(McpCatalogServer.name == name))).scalar_one()


@pytest.mark.asyncio
async def test_a_requested_sync_copies_the_registry_page_by_page(async_db: AsyncSession) -> None:
    service = _service(async_db, _config(), _Clock())
    registry = _Registry()
    await service.request_sync()

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
        await service.request_sync()

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

    await service.request_sync()
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

    await service.request_sync()
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
    await service.request_sync()
    partial = await service.sync_if_due(registry.fetch_page)
    assert partial is not None and partial.outcome == "partial"

    clock.advance(minutes=5)
    registry.pages[_PAGE_2_CURSOR] = _recorded("page_2")
    registry.asked.clear()
    await service.request_sync()
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
    await service.request_sync()
    await service.sync_if_due(registry.fetch_page)

    clock.advance(hours=7)
    registry.pages[_PAGE_2_CURSOR] = _recorded("page_2")
    registry.asked.clear()
    await service.request_sync()
    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None and status.outcome == "complete"
    assert registry.asked == [None, _PAGE_2_CURSOR]
    assert (status.pages, status.fetched) == (2, 9)


@pytest.mark.asyncio
async def test_a_resume_the_registry_refuses_starts_over_next_time(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    registry = _Registry({None: _recorded("page_1"), _PAGE_2_CURSOR: RegistryError("the registry answered HTTP 400")})
    await service.request_sync()
    await service.sync_if_due(registry.fetch_page)

    clock.advance(minutes=5)
    await service.request_sync()
    refused = await service.sync_if_due(registry.fetch_page)
    assert refused is not None and refused.outcome == "partial"

    clock.advance(minutes=5)
    registry.pages[_PAGE_2_CURSOR] = _recorded("page_2")
    registry.asked.clear()
    await service.request_sync()
    await service.sync_if_due(registry.fetch_page)

    assert registry.asked == [None, _PAGE_2_CURSOR]


@pytest.mark.asyncio
async def test_a_sync_that_reads_nothing_fails_and_leaves_the_catalog(async_db: AsyncSession) -> None:
    await _synced(async_db)
    clock = _Clock()
    clock.advance(hours=1)
    service = _service(async_db, _config(), clock)

    await service.request_sync()
    down = await service.sync_if_due(_Registry({None: RegistryError("the registry did not answer")}).fetch_page)
    clock.advance(hours=1)
    await service.request_sync()
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

    await service.request_sync()
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
    await service.request_sync()
    status = await service.sync_if_due(registry.fetch_page)

    assert status is not None
    assert (status.outcome, status.error, status.pages) == ("partial", "stopped after 30 minutes", 1)


@pytest.mark.asyncio
async def test_a_running_sync_is_not_started_twice_and_a_dead_one_is_taken_over(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    registry = _Registry()
    await service.request_sync()

    class _Crash(BaseException):
        pass

    async def dies_on_the_second_page(cursor: str | None) -> RegistryPage:
        if cursor is not None:
            raise _Crash
        return await registry.fetch_page(cursor)

    with pytest.raises(_Crash):
        await service.sync_if_due(dies_on_the_second_page)

    clock.advance(minutes=2)
    await service.request_sync()
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

    await service.request_sync()
    status = await service.sync_if_due(cut_short.fetch_page)

    assert status is not None
    assert (status.outcome, status.removed, status.server_count) == ("partial", 0, 7)
    assert status.error is not None and "1 of the 7 servers" in status.error
    assert await _names(async_db) == _KEPT
    # It read to the end, so there is no page to resume at.
    registry = _Registry()
    clock.advance(minutes=5)
    await service.request_sync()
    await service.sync_if_due(registry.fetch_page)
    assert registry.asked == [None, _PAGE_2_CURSOR]


@pytest.mark.asyncio
async def test_an_entry_the_database_refuses_is_left_out_and_the_page_stored(async_db: AsyncSession) -> None:
    def entry(name: str, description: str) -> McpCatalogEntry:
        return McpCatalogEntry(
            name=name,
            title=name,
            description=description,
            url="https://mcp.example.com/mcp",
            requires_token=False,
            website_url=None,
            version=None,
            source="registry",
        )

    servers = McpCatalogServerRepository(async_db)
    # The filter drops a NUL before it gets here; this is the store's own answer to one that slips through.
    refused = await servers.upsert(
        [entry("com.example/good", "fine"), entry("com.example/bad", "nul \x00 here"), entry("com.example/also", "ok")],
        seen_at=datetime(2026, 11, 10, 9, 0, tzinfo=UTC),
    )
    await async_db.commit()

    assert refused == 1
    assert await _names(async_db) == ["com.example/also", "com.example/good"]


@pytest.mark.asyncio
async def test_a_run_taken_for_dead_stops_without_touching_the_run_that_replaced_it(async_db: AsyncSession) -> None:
    clock = _Clock()
    slow = _service(async_db, _config(), clock)
    await slow.request_sync()
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
    await service.request_sync()

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
    await service.request_sync()
    resumed = await service.sync_if_due(registry.fetch_page)
    assert resumed is not None and resumed.outcome == "complete"
    assert registry.asked == [_PAGE_2_CURSOR]


@pytest.mark.asyncio
async def test_a_request_made_before_the_sync_was_switched_off_is_not_reported_as_pending(
    async_db: AsyncSession,
) -> None:
    await _service(async_db, _config(), _Clock()).request_sync()

    status = await _service(async_db, _config(mcp_catalog_sync="off"), _Clock()).sync_status()

    assert (status.mode, status.requested) == ("off", False)


@pytest.mark.asyncio
async def test_two_requests_featuring_one_server_at_once_leave_one_entry(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    organization = await _organization(async_db)
    featured = McpCatalogFeaturedRepository(async_db)
    copy = {"title": "Snag", "description": None, "url": "https://mcp.snag.ac/mcp"}

    first = await featured.add(organization.id, "ac.snag/snag", copy)
    # The second insert is what a request sees when it read the shelf before the first one committed.
    second = await featured.add(organization.id, "ac.snag/snag", copy)
    await async_db.commit()

    assert first is not None and second is None
    assert await featured.count_for_organization(organization.id) == 1
    admin = await _member(async_db, organization, role="admin")
    again = await service.feature(
        user=admin, request=McpCatalogFeatureRequest(server_id=await _server_id(async_db, "ac.snag/snag"))
    )
    assert again.id == first.id


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
async def test_an_admin_features_a_server_and_every_member_sees_it(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="admin")
    member = await _member(async_db, organization, role="member")
    server_id = await _server_id(async_db, "ae.easyvoice/tts")

    featured = await service.feature(user=admin, request=McpCatalogFeatureRequest(server_id=server_id))
    shelf = await service.list_featured(user=member)

    assert (featured.name, featured.url, featured.requires_token) == (
        "ae.easyvoice/tts",
        "https://easyvoice.ae/api/mcp",
        True,
    )
    assert (featured.catalog_status, featured.catalog_server_id) == ("current", server_id)
    assert [entry.id for entry in shelf.data] == [featured.id]
    assert shelf.count == 1


@pytest.mark.asyncio
async def test_a_plain_member_cannot_change_the_shelf(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="admin")
    member = await _member(async_db, organization, role="member")
    server_id = await _server_id(async_db, "ac.snag/snag")
    featured = await service.feature(user=admin, request=McpCatalogFeatureRequest(server_id=server_id))

    with pytest.raises(NotAuthorizedError):
        await service.feature(user=member, request=McpCatalogFeatureRequest(server_id=server_id))
    with pytest.raises(NotAuthorizedError):
        await service.unfeature(user=member, featured_id=featured.id)

    assert (await service.list_featured(user=member)).count == 1


@pytest.mark.asyncio
async def test_one_organizations_shelf_is_invisible_to_another(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    acme_admin = await _member(async_db, await _organization(async_db), role="owner")
    other_admin = await _member(async_db, await _organization(async_db, slug="globex"), role="owner")
    server_id = await _server_id(async_db, "ac.snag/snag")
    featured = await service.feature(user=acme_admin, request=McpCatalogFeatureRequest(server_id=server_id))

    assert (await service.list_featured(user=other_admin)).data == []
    # The same server on two shelves is two entries.
    theirs = await service.feature(user=other_admin, request=McpCatalogFeatureRequest(server_id=server_id))
    assert theirs.id != featured.id
    assert (await service.list_featured(user=acme_admin)).count == 1
    with pytest.raises(McpCatalogFeaturedNotFoundError):
        await service.unfeature(user=other_admin, featured_id=featured.id)


@pytest.mark.asyncio
async def test_a_sync_never_changes_or_removes_a_featured_entry(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    admin = await _member(async_db, await _organization(async_db), role="admin")
    kept = await service.feature(
        user=admin, request=McpCatalogFeatureRequest(server_id=await _server_id(async_db, "ac.inference.sh/mcp"))
    )
    gone = await service.feature(
        user=admin, request=McpCatalogFeatureRequest(server_id=await _server_id(async_db, "africa.lomi/mcp"))
    )
    clock = _Clock()
    clock.advance(hours=1)
    service = _service(async_db, _config(), clock)
    page = _recorded("page_1")
    moved = json.loads(json.dumps(page.items[0]))
    moved["server"]["remotes"][0]["url"] = "https://new.inference.sh/mcp"
    moved["server"]["title"] = "Renamed"
    await service.request_sync()
    await service.sync_if_due(
        _Registry({None: RegistryPage(items=[moved, *page.items[1:]], next_cursor=None)}).fetch_page
    )

    shelf = {entry.name: entry for entry in (await service.list_featured(user=admin)).data}

    assert (shelf[kept.name].url, shelf[kept.name].title) == ("https://api.inference.sh/mcp", "inference.sh")
    assert shelf[kept.name].catalog_status == "changed"
    assert shelf[gone.name].url == "https://mcp.lomi.africa/mcp"
    assert (shelf[gone.name].catalog_status, shelf[gone.name].catalog_server_id) == ("removed", None)
    assert shelf[kept.name].catalog_server_id == await _server_id(async_db, "ac.inference.sh/mcp")

    # Featuring it again is how an admin accepts what the catalog lists now.
    again = await service.feature(
        user=admin, request=McpCatalogFeatureRequest(server_id=await _server_id(async_db, "ac.inference.sh/mcp"))
    )
    assert (again.id, again.url, again.title, again.catalog_status) == (
        kept.id,
        "https://new.inference.sh/mcp",
        "Renamed",
        "current",
    )
    assert (await service.list_featured(user=admin)).count == 2


@pytest.mark.asyncio
async def test_unfeaturing_removes_the_entry_and_an_unknown_one_is_not_found(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    admin = await _member(async_db, await _organization(async_db), role="admin")
    featured = await service.feature(
        user=admin, request=McpCatalogFeatureRequest(server_id=await _server_id(async_db, "ac.snag/snag"))
    )

    await service.unfeature(user=admin, featured_id=featured.id)

    assert (await service.list_featured(user=admin)).data == []
    # The catalog server itself is untouched.
    assert await _names(async_db) == _KEPT
    with pytest.raises(McpCatalogFeaturedNotFoundError):
        await service.unfeature(user=admin, featured_id=featured.id)


@pytest.mark.asyncio
async def test_featuring_a_server_the_catalog_does_not_hold_is_not_found(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    admin = await _member(async_db, await _organization(async_db), role="admin")

    with pytest.raises(McpCatalogServerNotFoundError):
        await service.feature(user=admin, request=McpCatalogFeatureRequest(server_id=uuid.uuid4()))


@pytest.mark.asyncio
async def test_a_shelf_is_capped(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog_service, "MAX_FEATURED_PER_ORGANIZATION", 2)
    service = await _synced(async_db)
    admin = await _member(async_db, await _organization(async_db), role="admin")
    ids = [await _server_id(async_db, name) for name in _KEPT[:3]]
    for server_id in ids[:2]:
        await service.feature(user=admin, request=McpCatalogFeatureRequest(server_id=server_id))

    # Taking an existing entry's copy again is not one more.
    await service.feature(user=admin, request=McpCatalogFeatureRequest(server_id=ids[0]))
    with pytest.raises(McpCatalogFeaturedLimitReachedError):
        await service.feature(user=admin, request=McpCatalogFeatureRequest(server_id=ids[2]))


@pytest.mark.asyncio
async def test_deleting_an_organization_takes_its_shelf(async_db: AsyncSession) -> None:
    service = await _synced(async_db)
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="admin")
    await service.feature(
        user=admin, request=McpCatalogFeatureRequest(server_id=await _server_id(async_db, "ac.snag/snag"))
    )

    # An identity always points at an organization, so the admin moves before theirs goes.
    admin.active_organization_id = (await _organization(async_db, slug="globex")).id
    await async_db.commit()
    await async_db.delete(await async_db.get(Organization, organization.id))
    await async_db.commit()

    assert (await async_db.execute(select(McpCatalogFeatured))).scalars().all() == []


@pytest.mark.asyncio
async def test_two_workers_polling_at_once_start_one_sync(async_db: AsyncSession) -> None:
    clock = _Clock()
    service = _service(async_db, _config(), clock)
    await service.request_sync()
    state = McpCatalogSyncStateRepository(async_db)

    claims = [
        await state.claim(clock.now, scheduled=False, stale_before=clock.now, next_sync_at=clock.now) for _ in range(2)
    ]
    await async_db.commit()

    assert claims == [True, False]
    row = (await async_db.execute(select(McpCatalogSyncState))).scalar_one()
    assert (row.running, row.sync_requested) == (True, False)


def test_the_catalog_is_empty_and_searchable_before_any_sync(client: TestClient) -> None:
    servers = client.get(f"{_CATALOG}/servers", headers=_MASTER)
    featured = client.get(_FEATURED, headers=_MASTER)
    status = client.get(f"{_CATALOG}/sync", headers=_MASTER)

    assert servers.status_code == 200
    assert servers.json() == {"data": [], "count": 0}
    assert featured.status_code == 200
    assert featured.json() == {"data": [], "count": 0}
    assert status.status_code == 200
    body = status.json()
    assert (body["mode"], body["running"], body["outcome"], body["server_count"]) == ("manual", False, None, 0)


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
        assert client.get(_FEATURED, headers=_MASTER).status_code == 404
        assert "mcp_catalog" not in client.get(f"{API_ROOT}/bootstrap").json()["surfaces"]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", f"{_CATALOG}/servers"),
        ("GET", f"{_CATALOG}/sync"),
        ("POST", f"{_CATALOG}/sync"),
        ("GET", _FEATURED),
        ("POST", _FEATURED),
        ("DELETE", f"{_FEATURED}/00000000-0000-4000-8000-000000000000"),
    ],
)
def test_a_caller_without_a_credential_is_refused(client: TestClient, method: str, path: str) -> None:
    assert client.request(method, path).status_code == 401


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "skip=-1", f"q={'x' * 201}"])
def test_a_search_out_of_bounds_is_refused(client: TestClient, query: str) -> None:
    assert client.get(f"{_CATALOG}/servers?{query}", headers=_MASTER).status_code == 422


def test_featuring_over_http_answers_not_found_and_validates(client: TestClient) -> None:
    unknown = client.post(_FEATURED, json={"server_id": str(uuid.uuid4())}, headers=_MASTER)
    malformed = client.post(_FEATURED, json={"server_id": "nope"}, headers=_MASTER)
    extra = client.post(_FEATURED, json={"server_id": str(uuid.uuid4()), "url": "https://x.example"}, headers=_MASTER)
    missing = client.delete(f"{_FEATURED}/{uuid.uuid4()}", headers=_MASTER)

    assert unknown.status_code == 404
    assert malformed.status_code == 422
    assert extra.status_code == 422
    assert missing.status_code == 404
