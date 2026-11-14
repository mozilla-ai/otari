"""Members' requests for a catalog server: who may ask, list, withdraw and decide, the bounds, and the mail.

At the service layer, as `test_workspace_mcp_servers.py` is: the API acts as the one bootstrap operator, so
a member refused and another organization's workspace invisible are reachable only with identities built
at the role a case needs. The routes are walked once as that operator at the end.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, event, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import NotAuthorizedError, WorkspaceNotFoundError
from gateway.exceptions.tools_exceptions import (
    McpCatalogServerNotListedError,
    McpServerAlreadyAddedError,
    McpServerAlreadyRequestedError,
    McpServerRequestDecidedError,
    McpServerRequestLimitError,
    McpServerRequestNotFoundError,
    McpServerRequestPendingError,
)
from gateway.models.mcp import ResolvedMcpServer
from gateway.models.mcp_catalog import McpCatalogServer
from gateway.models.tenancy import Organization, User, Workspace
from gateway.models.tools import McpServerRequest, McpServerRequestStatus, WorkspaceMcpServer
from gateway.repositories.mcp_catalog import McpCatalogServerRepository, McpCatalogSyncStateRepository
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.tools import McpServerRequestRepository, WorkspaceMcpServerRepository
from gateway.schemas.mcp_catalog import McpCatalogEntry
from gateway.schemas.tools import McpServerRequestCreate, McpServerRequestDecline
from gateway.services import mcp_stateless
from gateway.services.mail import Mailer, MailMessage
from gateway.services.mcp_catalog import McpCatalogService
from gateway.services.mcp_stateless import ServerCheck
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.workspace_mcp_server_service import WorkspaceMcpServerCreate, WorkspaceMcpServerService
from gateway.services.tools import (
    McpGrantMembershipListener,
    McpServerRequestService,
    WorkspaceMcpServerUrls,
    _mcp_server_requests,
)
from gateway.services.tools._mcp_server_requests import (
    MAX_PENDING_REQUESTS_PER_MEMBER,
    MAX_REQUEST_MAIL_RECIPIENTS,
    MAX_REQUESTS_PER_MEMBER_PER_DAY,
)

# Public IP literals, so the URL safety check never reaches a DNS resolver.
SENTRY = ("io.sentry/mcp", "Sentry", "https://93.184.216.34/mcp")
LINEAR = ("app.linear/mcp", "Linear", "https://93.184.216.35/mcp")


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _no_dialing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored server is checked when it is created; nothing here should reach the network."""

    async def check(server: ResolvedMcpServer) -> ServerCheck:
        return ServerCheck(tool_names=[])

    monkeypatch.setattr(mcp_stateless, "check_stored_server", check)


class _Mail:
    """Stands in for the mailer: records what it was asked to send, and can be switched off."""

    def __init__(self, *, ready: bool = True) -> None:
        self.can_send_links = ready
        self.asset_root = None
        self.sent: list[tuple[str, MailMessage]] = []

    def link(self, path: str) -> str:
        return f"https://otari.example{path}"

    async def send(self, *, to: str, message: MailMessage) -> None:
        self.sent.append((to, message))


class _Later:
    """Collects the work a service defers until after the response, for a test to run."""

    def __init__(self) -> None:
        self.work: list[Callable[[], Awaitable[None]]] = []

    def __call__(self, work: Callable[[], Awaitable[None]]) -> None:
        self.work.append(work)

    async def run(self) -> None:
        pending, self.work = self.work, []
        for work in pending:
            await work()


class _World:
    def __init__(self, db: AsyncSession, *, mail_ready: bool = True, clock: Callable[[], datetime] | None = None):
        self.db = db
        self.mail = _Mail(ready=mail_ready)
        self.later = _Later()
        uow = UnitOfWork(db)
        access = WorkspaceAccess(db, OrganizationService(db, membership_listener=None))
        catalog = McpCatalogService(
            uow,
            McpCatalogServerRepository(uow),
            McpCatalogSyncStateRepository(uow),
            access,
            GatewayConfig(mcp_catalog_sync="manual"),
            workspace_servers=WorkspaceMcpServerUrls(WorkspaceMcpServerRepository(uow)),
        )
        self.requests = McpServerRequestService(
            uow,
            McpServerRequestRepository(uow),
            WorkspaceMcpServerRepository(uow),
            access,
            catalog,
            cast(Mailer, self.mail),
            send_later=self.later,
            **({"clock": clock} if clock is not None else {}),
        )
        self.servers = WorkspaceMcpServerService(db, requests=self.requests, config=GatewayConfig())

    async def sent(self) -> list[tuple[str, str]]:
        """Each address mailed so far and the subject it got, once the deferred sends have run."""
        await self.later.run()
        return [(to, message.subject) for to, message in self.mail.sent]


async def _organization(db: AsyncSession, *, slug: str = "acme") -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _person(
    db: AsyncSession,
    organization: Organization,
    name: str,
    *,
    role: str = "member",
    workspace: Workspace | None = None,
    workspace_role: str = "member",
) -> User:
    user = await UserRepository(db).create_local_identity(full_name=name, active_organization_id=organization.id)
    user.email = f"{name.lower().replace(' ', '.')}@{organization.slug}.example"
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    if workspace is not None:
        await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=user.id, role=workspace_role)
    await db.commit()
    return user


async def _workspace(db: AsyncSession, organization: Organization, name: str = "Research") -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name, organization_id=organization.id, created_by_user_id=None
    )
    await db.commit()
    return workspace


async def _list_in_catalog(db: AsyncSession, *servers: tuple[str, str, str]) -> None:
    entries = [
        McpCatalogEntry(
            name=name,
            title=title,
            description=None,
            url=url,
            requires_token=False,
            website_url=None,
            version=None,
            source="registry",
        )
        for name, title, url in servers
    ]
    await McpCatalogServerRepository(db).upsert(entries, seen_at=datetime.now(UTC))
    await db.commit()


class _Team:
    """An organization with one workspace, its owner, a workspace admin, and two plain members."""

    organization: Organization
    workspace: Workspace
    owner: User
    admin: User
    ada: User
    grace: User


async def _team(db: AsyncSession) -> _Team:
    team = _Team()
    team.organization = await _organization(db)
    team.workspace = await _workspace(db, team.organization)
    team.owner = await _person(db, team.organization, "Olive Owner", role="owner")
    team.admin = await _person(db, team.organization, "Adam Admin", workspace=team.workspace, workspace_role="admin")
    team.ada = await _person(db, team.organization, "Ada Member", workspace=team.workspace)
    team.grace = await _person(db, team.organization, "Grace Viewer", workspace=team.workspace, workspace_role="viewer")
    await _list_in_catalog(db, SENTRY, LINEAR)
    return team


async def _reload(db: AsyncSession, team: _Team) -> None:
    """Read the team's rows again. A refusal rolls the session back, which expires everything it loaded."""
    for row in (team.organization, team.workspace, team.owner, team.admin, team.ada, team.grace):
        await db.refresh(row)


def _ask(name: str = SENTRY[0], note: str | None = None) -> McpServerRequestCreate:
    return McpServerRequestCreate(catalog_name=name, note=note)


@pytest.mark.asyncio
async def test_a_member_asks_and_the_managers_are_mailed_once_each(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)

    made = await world.requests.create(
        user=team.ada, workspace_id=team.workspace.id, request=_ask(note="  Triage production errors  ")
    )

    assert (made.status, made.title, made.url) == ("pending", "Sentry", SENTRY[2])
    assert made.note == "Triage production errors"
    assert made.catalog_server is not None and made.catalog_server.name == SENTRY[0]
    # The workspace's own admin first, then the organization's owner; never the requester or another member.
    assert await world.sent() == [
        ("adam.admin@acme.example", "Ada Member asked for Sentry in Research"),
        ("olive.owner@acme.example", "Ada Member asked for Sentry in Research"),
    ]
    text = world.mail.sent[0][1].text
    assert "Triage production errors" in text
    assert "https://otari.example/#/tools/mcp-servers" in text


@pytest.mark.asyncio
async def test_without_mail_the_request_is_stored_and_nothing_is_sent(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db, mail_ready=False)

    await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    listed = await world.requests.pending(user=team.admin, workspace_id=team.workspace.id)

    assert listed.count == 1
    assert await world.sent() == []


@pytest.mark.asyncio
async def test_a_viewer_may_ask(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)

    made = await world.requests.create(user=team.grace, workspace_id=team.workspace.id, request=_ask())

    assert made.status == "pending"


@pytest.mark.asyncio
async def test_a_request_names_a_server_the_catalog_lists(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)

    with pytest.raises(McpCatalogServerNotListedError):
        await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask("com.nobody/mcp"))


@pytest.mark.asyncio
async def test_one_pending_request_per_member_and_server_while_others_may_ask_too(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask(note="Errors"))

    with pytest.raises(McpServerAlreadyRequestedError):
        await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    await _reload(async_db, team)
    await world.requests.create(user=team.grace, workspace_id=team.workspace.id, request=_ask(note="Dashboards"))

    listed = await world.requests.pending(user=team.owner, workspace_id=team.workspace.id)
    assert listed.count == 2
    assert [group.catalog_name for group in listed.data] == [SENTRY[0]]
    group = listed.data[0]
    assert [(who.full_name, who.note) for who in group.requesters] == [
        ("Ada Member", "Errors"),
        ("Grace Viewer", "Dashboards"),
    ]
    assert group.catalog_server is not None and group.catalog_server.url == SENTRY[2]


@pytest.mark.asyncio
async def test_a_server_the_workspace_has_cannot_be_requested(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    # Stored with a trailing slash the catalog's URL does not have: still the same endpoint.
    await world.servers.create_server(
        user=team.owner,
        workspace_id=team.workspace.id,
        request=WorkspaceMcpServerCreate(name="Sentry", url=f"{SENTRY[2]}/"),
    )

    with pytest.raises(McpServerAlreadyAddedError):
        await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())


@pytest.mark.asyncio
async def test_a_member_holds_a_bounded_number_of_waiting_requests(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    servers = [(f"com.example/s{i}", f"S{i}", f"https://93.184.216.40/mcp/{i}") for i in range(12)]
    await _list_in_catalog(async_db, *servers)
    world = _World(async_db)
    assert MAX_PENDING_REQUESTS_PER_MEMBER == MAX_REQUESTS_PER_MEMBER_PER_DAY == 10
    for name, _, _ in servers[:10]:
        await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask(name))

    with pytest.raises(McpServerRequestLimitError, match="10 requests waiting") as refused:
        await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask(servers[10][0]))

    assert refused.value.status_code == 429
    await _reload(async_db, team)
    # The bound is each member's own.
    await world.requests.create(user=team.grace, workspace_id=team.workspace.id, request=_ask(servers[10][0]))


@pytest.mark.asyncio
async def test_withdrawing_and_asking_again_still_counts_toward_the_day(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    now = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
    world = _World(async_db, clock=lambda: now)
    for _ in range(10):
        made = await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
        await world.requests.withdraw(user=team.ada, workspace_id=team.workspace.id, request_id=made.id)

    with pytest.raises(McpServerRequestLimitError, match="in the last day"):
        await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    await _reload(async_db, team)
    # Ten requests, ten mails to each of the two managers, and no more.
    assert len(await world.sent()) == 20

    # The rows were stamped by the real clock, so the day is aged out by moving them before the service's.
    for row in (await async_db.execute(select(McpServerRequest))).scalars():
        row.created_at = now - timedelta(days=2)
    await async_db.commit()
    assert (await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())).id


@pytest.mark.asyncio
async def test_one_request_is_mailed_to_a_bounded_number_of_managers(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    for index in range(MAX_REQUEST_MAIL_RECIPIENTS + 5):
        await _person(async_db, team.organization, f"Extra Admin {index}", role="admin")
    world = _World(async_db)

    await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())

    assert len(await world.sent()) == MAX_REQUEST_MAIL_RECIPIENTS


@pytest.mark.asyncio
async def test_a_member_lists_and_withdraws_only_their_own(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    adas = await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    graces = await world.requests.create(user=team.grace, workspace_id=team.workspace.id, request=_ask(LINEAR[0]))

    mine = await world.requests.mine(user=team.ada, workspace_id=team.workspace.id)
    assert [entry.id for entry in mine.data] == [adas.id]
    # Someone else's request is answered as one that does not exist, also for a manager.
    for caller in (team.ada, team.owner):
        with pytest.raises(McpServerRequestNotFoundError):
            await world.requests.withdraw(user=caller, workspace_id=team.workspace.id, request_id=graces.id)
        await _reload(async_db, team)
    # Only managers see everyone's.
    for member in (team.ada, team.grace):
        with pytest.raises(NotAuthorizedError):
            await world.requests.pending(user=member, workspace_id=team.workspace.id)
        with pytest.raises(NotAuthorizedError):
            await world.requests.decline(
                user=member, workspace_id=team.workspace.id, request=McpServerRequestDecline(catalog_name=LINEAR[0])
            )

    await world.requests.withdraw(user=team.ada, workspace_id=team.workspace.id, request_id=adas.id)
    assert (await world.requests.mine(user=team.ada, workspace_id=team.workspace.id)).data == []
    with pytest.raises(McpServerRequestDecidedError):
        await world.requests.withdraw(user=team.ada, workspace_id=team.workspace.id, request_id=adas.id)
    await _reload(async_db, team)
    # Withdrawn, so the server can be asked for again.
    assert (await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())).id != adas.id


@pytest.mark.asyncio
async def test_both_lists_page_and_a_managers_page_keeps_each_servers_requests_together(
    async_db: AsyncSession,
) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    for who, name in ((team.ada, SENTRY[0]), (team.ada, LINEAR[0]), (team.grace, SENTRY[0])):
        await world.requests.create(user=who, workspace_id=team.workspace.id, request=_ask(name))

    first = await world.requests.pending(user=team.owner, workspace_id=team.workspace.id, limit=1)
    second = await world.requests.pending(user=team.owner, workspace_id=team.workspace.id, skip=1, limit=1)
    past = await world.requests.pending(user=team.owner, workspace_id=team.workspace.id, skip=2, limit=1)

    assert [(group.catalog_name, len(group.requesters)) for group in first.data] == [(SENTRY[0], 2)]
    assert [(group.catalog_name, len(group.requesters)) for group in second.data] == [(LINEAR[0], 1)]
    assert past.data == []
    assert first.count == second.count == past.count == 3
    # `count` totals the requests; the servers they ask for are what a page of `data` holds.
    assert first.server_count == second.server_count == past.server_count == 2

    newest = await world.requests.mine(user=team.ada, workspace_id=team.workspace.id, limit=1)
    older = await world.requests.mine(user=team.ada, workspace_id=team.workspace.id, skip=1, limit=1)
    assert [entry.catalog_name for entry in newest.data + older.data] == [LINEAR[0], SENTRY[0]]


@pytest.mark.asyncio
async def test_a_waiting_request_stays_on_the_first_page_behind_any_number_of_answers(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    waiting = await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    # Written directly, past the daily cap: more answers than one default page, all newer and unacknowledged.
    later = datetime.now(UTC) + timedelta(minutes=1)
    for n in range(51):
        async_db.add(
            McpServerRequest(
                workspace_id=team.workspace.id,
                catalog_name=LINEAR[0],
                title=LINEAR[1],
                url=LINEAR[2],
                requester_id=team.ada.id,
                status=McpServerRequestStatus.DECLINED,
                decided_at=later,
                created_at=later + timedelta(seconds=n),
            )
        )
    await async_db.commit()

    first = (await world.requests.mine(user=team.ada, workspace_id=team.workspace.id)).data

    assert len(first) == 50
    assert first[0].id == waiting.id
    assert [entry.status for entry in first[1:]] == ["declined"] * 49
    # The answers behind it stay newest first, so what the page cuts is the oldest of them.
    assert first[1].created_at > first[-1].created_at


async def _waiting(db: AsyncSession, team: _Team, entry: tuple[str, str, str], people: int) -> None:
    """``people`` members of their own asking for one catalog server, written directly: each asks once."""
    start = datetime.now(UTC)
    for n in range(people):
        person = await UserRepository(db).create_local_identity(
            full_name=f"Asker {n}", active_organization_id=team.organization.id
        )
        db.add(
            McpServerRequest(
                workspace_id=team.workspace.id,
                catalog_name=entry[0],
                title=entry[1],
                url=entry[2],
                requester_id=person.id,
                note=f"note {n}",
                created_at=start + timedelta(microseconds=n),
            )
        )
    await db.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [f"{SENTRY[2]}\n", f"\t{SENTRY[2]}/", f" {SENTRY[2]}/\r\n"])
async def test_a_stored_url_with_whitespace_around_it_still_counts_as_stored(
    async_db: AsyncSession, stored: str
) -> None:
    team = await _team(async_db)
    async_db.add(WorkspaceMcpServer(workspace_id=team.workspace.id, name="sentry", url=stored))
    await async_db.commit()

    async with UnitOfWork(async_db) as uow:
        assert await WorkspaceMcpServerRepository(uow).has_url(team.workspace.id, SENTRY[2])


@pytest.mark.asyncio
async def test_one_busy_server_cannot_crowd_another_out_of_the_page_and_says_how_many_asked(
    async_db: AsyncSession,
) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    await _waiting(async_db, team, SENTRY, _mcp_server_requests.MAX_REQUESTERS_LISTED + 3)
    await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask(LINEAR[0]))

    listed = await world.requests.pending(user=team.owner, workspace_id=team.workspace.id)

    cap = _mcp_server_requests.MAX_REQUESTERS_LISTED
    assert [(group.catalog_name, len(group.requesters), group.requester_count) for group in listed.data] == [
        (SENTRY[0], cap, cap + 3),
        (LINEAR[0], 1, 1),
    ]
    assert listed.count == cap + 4
    assert listed.server_count == 2


@pytest.mark.asyncio
async def test_every_requester_of_a_server_past_any_cap_can_be_read_a_page_at_a_time(
    async_db: AsyncSession,
) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    await _waiting(async_db, team, SENTRY, 501)

    seen: list[str | None] = []
    skip = 0
    while True:
        page = await world.requests.requesters(
            user=team.owner, workspace_id=team.workspace.id, catalog_name=SENTRY[0], skip=skip
        )
        assert len(page.data) <= _mcp_server_requests.MAX_REQUESTERS_PAGE
        assert page.count == 501
        if not page.data:
            break
        seen.extend(requester.note for requester in page.data)
        skip += len(page.data)

    assert seen == [f"note {n}" for n in range(501)]
    with pytest.raises(NotAuthorizedError):
        await world.requests.requesters(user=team.ada, workspace_id=team.workspace.id, catalog_name=SENTRY[0])


@pytest.mark.asyncio
async def test_a_withdrawal_locks_the_request_an_approval_would_change(async_db: AsyncSession) -> None:
    """The withdrawal reads the row as the approval does, locked, so the later of the two sees the other's answer."""
    team = await _team(async_db)
    world = _World(async_db)
    made = await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    statements: list[str] = []

    def record(_conn: object, _cursor: object, statement: str, *_args: object) -> None:
        statements.append(statement)

    event.listen(Engine, "before_cursor_execute", record)
    try:
        await world.requests.withdraw(user=team.ada, workspace_id=team.workspace.id, request_id=made.id)
    finally:
        event.remove(Engine, "before_cursor_execute", record)

    reads = [statement for statement in statements if statement.lstrip().startswith("SELECT mcp_server_requests")]
    assert reads and all("FOR UPDATE" in statement for statement in reads)


def test_the_request_lists_take_a_bounded_page(client: TestClient) -> None:
    workspace_id = client.get(f"{API_ROOT}/workspaces", headers=_MASTER).json()["data"][0]["id"]
    base = f"{API_ROOT}/workspaces/{workspace_id}/mcp-server-requests"

    assert client.get(f"{base}?limit=501", headers=_MASTER).status_code == 422
    assert client.get(f"{base}/mine?limit=101", headers=_MASTER).status_code == 422
    assert client.get(f"{base}?skip=-1", headers=_MASTER).status_code == 422
    assert client.get(f"{base}?skip=5&limit=5", headers=_MASTER).json() == {"data": [], "count": 0, "server_count": 0}


@pytest.mark.asyncio
async def test_another_organizations_workspace_and_requests_are_invisible(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    made = await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    elsewhere = await _organization(async_db, slug="globex")
    their_workspace = await _workspace(async_db, elsewhere, "Theirs")
    outsider = await _person(async_db, elsewhere, "Otto Outsider", role="owner")

    with pytest.raises(WorkspaceNotFoundError):
        await world.requests.pending(user=outsider, workspace_id=team.workspace.id)
    with pytest.raises(WorkspaceNotFoundError):
        await world.requests.create(user=outsider, workspace_id=team.workspace.id, request=_ask())
    with pytest.raises(WorkspaceNotFoundError):
        await world.requests.withdraw(user=outsider, workspace_id=team.workspace.id, request_id=made.id)
    # A request's id is looked up inside the workspace named, so one from elsewhere is not found there.
    with pytest.raises(McpServerRequestNotFoundError):
        await world.requests.withdraw(user=outsider, workspace_id=their_workspace.id, request_id=made.id)
    await _reload(async_db, team)
    # A member of the organization who is not in the workspace cannot see it either.
    stranger = await _person(async_db, team.organization, "Sam Stranger")
    with pytest.raises(WorkspaceNotFoundError):
        await world.requests.mine(user=stranger, workspace_id=team.workspace.id)


@pytest.mark.asyncio
async def test_declining_answers_everyone_waiting_and_tells_them_why(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    await world.requests.create(user=team.grace, workspace_id=team.workspace.id, request=_ask())
    await world.requests.create(user=team.grace, workspace_id=team.workspace.id, request=_ask(LINEAR[0]))
    await world.sent()
    world.mail.sent.clear()

    answer = await world.requests.decline(
        user=team.admin,
        workspace_id=team.workspace.id,
        request=McpServerRequestDecline(catalog_name=SENTRY[0], reason="We use Datadog for this"),
    )

    assert answer.declined == 2
    left = await world.requests.pending(user=team.admin, workspace_id=team.workspace.id)
    assert [group.catalog_name for group in left.data] == [LINEAR[0]]
    mine = (await world.requests.mine(user=team.ada, workspace_id=team.workspace.id)).data
    assert [(entry.status, entry.decision_reason) for entry in mine] == [("declined", "We use Datadog for this")]
    assert await world.sent() == [
        ("ada.member@acme.example", "Your request for Sentry was declined"),
        ("grace.viewer@acme.example", "Your request for Sentry was declined"),
    ]
    assert "We use Datadog for this" in world.mail.sent[0][1].text
    with pytest.raises(McpServerRequestNotFoundError):
        await world.requests.decline(
            user=team.admin, workspace_id=team.workspace.id, request=McpServerRequestDecline(catalog_name=SENTRY[0])
        )


@pytest.mark.asyncio
async def test_adding_the_server_approves_everyone_waiting_for_it(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    await world.requests.create(user=team.grace, workspace_id=team.workspace.id, request=_ask())
    await world.requests.create(user=team.grace, workspace_id=team.workspace.id, request=_ask(LINEAR[0]))
    await world.sent()
    world.mail.sent.clear()

    # By its URL, as any add stores it: the request is answered however the server arrives.
    server = await world.servers.create_server(
        user=team.owner,
        workspace_id=team.workspace.id,
        request=WorkspaceMcpServerCreate(name="Sentry", url=f"{SENTRY[2]}/"),
    )

    rows = (await async_db.execute(select(McpServerRequest).order_by(McpServerRequest.created_at))).scalars().all()
    assert [(row.catalog_name, row.status, row.server_id, row.decided_by) for row in rows] == [
        (SENTRY[0], McpServerRequestStatus.APPROVED, server.id, team.owner.id),
        (SENTRY[0], McpServerRequestStatus.APPROVED, server.id, team.owner.id),
        (LINEAR[0], McpServerRequestStatus.PENDING, None, None),
    ]
    assert await world.sent() == [
        ("ada.member@acme.example", "Sentry was added to Research"),
        ("grace.viewer@acme.example", "Sentry was added to Research"),
    ]
    # The answer stays in the member's list until they acknowledge it.
    mine = (await world.requests.mine(user=team.ada, workspace_id=team.workspace.id)).data
    assert [(entry.status, entry.server_id) for entry in mine] == [("approved", server.id)]
    await world.requests.acknowledge(user=team.ada, workspace_id=team.workspace.id, request_id=mine[0].id)
    assert (await world.requests.mine(user=team.ada, workspace_id=team.workspace.id)).data == []
    # Grace has not, and once the server is deleted there is nothing left to tell her about.
    assert len((await world.requests.mine(user=team.grace, workspace_id=team.workspace.id)).data) == 2
    await world.servers.delete_server(user=team.owner, workspace_id=team.workspace.id, server_id=server.id)
    await _reload(async_db, team)
    left = (await world.requests.mine(user=team.grace, workspace_id=team.workspace.id)).data
    assert [entry.status for entry in left] == ["pending"]


@pytest.mark.asyncio
async def test_an_answer_is_acknowledged_only_by_who_asked_and_only_once_given(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    asked = await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())

    with pytest.raises(McpServerRequestPendingError):
        await world.requests.acknowledge(user=team.ada, workspace_id=team.workspace.id, request_id=asked.id)
    await _reload(async_db, team)
    await world.requests.decline(
        user=team.admin, workspace_id=team.workspace.id, request=McpServerRequestDecline(catalog_name=SENTRY[0])
    )
    # Someone else's answer is one that does not exist, also for a manager.
    for caller in (team.grace, team.owner):
        with pytest.raises(McpServerRequestNotFoundError):
            await world.requests.acknowledge(user=caller, workspace_id=team.workspace.id, request_id=asked.id)
        await _reload(async_db, team)
    assert len((await world.requests.mine(user=team.ada, workspace_id=team.workspace.id)).data) == 1

    for _ in range(2):
        await world.requests.acknowledge(user=team.ada, workspace_id=team.workspace.id, request_id=asked.id)
    assert (await world.requests.mine(user=team.ada, workspace_id=team.workspace.id)).data == []
    # The row is history, not removed.
    row = (await async_db.execute(select(McpServerRequest))).scalar_one()
    assert (row.status, row.acknowledged_at is not None) == (McpServerRequestStatus.DECLINED, True)


@pytest.mark.asyncio
async def test_asking_again_replaces_the_decline_and_adding_the_server_clears_every_decline(
    async_db: AsyncSession,
) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    for member in (team.ada, team.grace):
        await world.requests.create(user=member, workspace_id=team.workspace.id, request=_ask())
    await world.requests.decline(
        user=team.admin, workspace_id=team.workspace.id, request=McpServerRequestDecline(catalog_name=SENTRY[0])
    )

    again = await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    # Only the new request is Ada's to see; Grace, who did not ask again, still sees her decline.
    adas = (await world.requests.mine(user=team.ada, workspace_id=team.workspace.id)).data
    assert [(entry.id, entry.status) for entry in adas] == [(again.id, "pending")]
    graces = (await world.requests.mine(user=team.grace, workspace_id=team.workspace.id)).data
    assert [entry.status for entry in graces] == ["declined"]

    await world.sent()
    world.mail.sent.clear()
    await world.servers.create_server(
        user=team.owner, workspace_id=team.workspace.id, request=WorkspaceMcpServerCreate(name="Sentry", url=SENTRY[2])
    )
    adas = (await world.requests.mine(user=team.ada, workspace_id=team.workspace.id)).data
    assert [(entry.id, entry.status) for entry in adas] == [(again.id, "approved")]
    assert (await world.requests.mine(user=team.grace, workspace_id=team.workspace.id)).data == []
    # One mail, to the one member whose request the server answered.
    assert await world.sent() == [("ada.member@acme.example", "Sentry was added to Research")]
    assert (await async_db.execute(select(func.count()).select_from(McpServerRequest))).scalar_one() == 3


@pytest.mark.asyncio
async def test_a_refused_add_approves_nothing(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())

    with pytest.raises(NotAuthorizedError):
        await world.servers.create_server(
            user=team.grace,
            workspace_id=team.workspace.id,
            request=WorkspaceMcpServerCreate(name="Sentry", url=SENTRY[2]),
        )

    assert (await world.requests.pending(user=team.owner, workspace_id=team.workspace.id)).count == 1
    assert (await async_db.execute(select(WorkspaceMcpServer))).scalars().all() == []


@pytest.mark.asyncio
async def test_a_request_outlives_its_catalog_entry(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    row = (await async_db.execute(select(McpCatalogServer).where(McpCatalogServer.name == SENTRY[0]))).scalar_one()
    await async_db.delete(row)
    await async_db.commit()

    group = (await world.requests.pending(user=team.owner, workspace_id=team.workspace.id)).data[0]

    assert (group.title, group.url, group.catalog_server) == ("Sentry", SENTRY[2], None)


@pytest.mark.asyncio
async def test_a_member_who_leaves_the_workspace_has_their_waiting_requests_withdrawn(async_db: AsyncSession) -> None:
    team = await _team(async_db)
    world = _World(async_db)
    await world.requests.create(user=team.ada, workspace_id=team.workspace.id, request=_ask())
    await world.requests.create(user=team.grace, workspace_id=team.workspace.id, request=_ask())

    uow = UnitOfWork(async_db)
    async with uow:
        await McpGrantMembershipListener(uow).member_lost_workspaces(team.ada.id, [team.workspace.id])

    group = (await world.requests.pending(user=team.owner, workspace_id=team.workspace.id)).data[0]
    assert [who.full_name for who in group.requesters] == ["Grace Viewer"]


_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}


def test_the_routes_as_the_operator(client: TestClient, db_session: Session) -> None:
    db_session.add(
        McpCatalogServer(
            name=SENTRY[0], title=SENTRY[1], url=SENTRY[2], source="registry", last_seen_at=datetime.now(UTC)
        )
    )
    db_session.commit()
    workspace_id = client.get(f"{API_ROOT}/workspaces", headers=_MASTER).json()["data"][0]["id"]
    base = f"{API_ROOT}/workspaces/{workspace_id}/mcp-server-requests"

    made = client.post(base, headers=_MASTER, json={"catalog_name": SENTRY[0], "note": "For triage"})
    assert made.status_code == 201, made.text
    request_id = made.json()["id"]
    assert client.post(base, headers=_MASTER, json={"catalog_name": SENTRY[0]}).status_code == 409
    assert client.post(base, headers=_MASTER, json={"catalog_name": "com.nobody/mcp"}).status_code == 404
    assert client.post(base, headers=_MASTER, json={"catalog_name": SENTRY[0], "note": "x" * 501}).status_code == 422

    listed = client.get(base, headers=_MASTER).json()
    assert listed["count"] == 1
    assert listed["data"][0]["requesters"][0]["note"] == "For triage"
    assert [entry["id"] for entry in client.get(f"{base}/mine", headers=_MASTER).json()["data"]] == [request_id]

    assert client.delete(f"{base}/{uuid.uuid4()}", headers=_MASTER).status_code == 404
    assert client.delete(f"{base}/{request_id}", headers=_MASTER).status_code == 204
    assert client.delete(f"{base}/{request_id}", headers=_MASTER).status_code == 409
    assert client.post(f"{base}/decline", headers=_MASTER, json={"catalog_name": SENTRY[0]}).status_code == 404

    client.post(base, headers=_MASTER, json={"catalog_name": SENTRY[0]})
    again = client.get(f"{base}/mine", headers=_MASTER).json()["data"][0]["id"]
    assert client.post(f"{base}/{again}/acknowledge", headers=_MASTER).status_code == 409
    declined = client.post(f"{base}/decline", headers=_MASTER, json={"catalog_name": SENTRY[0], "reason": "No"})
    assert declined.json() == {"declined": 1}
    assert client.post(f"{base}/{uuid.uuid4()}/acknowledge", headers=_MASTER).status_code == 404
    assert client.post(f"{base}/{again}/acknowledge", headers=_MASTER).status_code == 204
    assert client.get(f"{base}/mine", headers=_MASTER).json()["data"] == []
    foreign = f"{API_ROOT}/workspaces/{uuid.uuid4()}/mcp-server-requests"
    assert client.get(foreign, headers=_MASTER).status_code == 404
    assert client.get(f"{foreign}/mine", headers=_MASTER).status_code == 404
