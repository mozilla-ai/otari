"""Managing an organization's own MCP servers: its owners and admins add, change, remove and limit them.

Exercised at the service layer, as ``test_workspace_mcp_servers.py`` is, because the rules that matter (a plain member
refused, a workspace admin refused, another organization's workspace not found) need identities at roles the API's one
operator identity never has.
"""

import ipaddress
import uuid

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions import TenancyError
from gateway.exceptions.organizations_exceptions import NotAuthorizedError, WorkspaceNotFoundError
from gateway.exceptions.tools_exceptions import (
    McpServerNameTakenError,
    WorkspaceMcpServerNotFoundError,
    WorkspaceMcpServerUnsafeUrlError,
)
from gateway.models.tenancy import OrganizationMember, User, Workspace
from gateway.models.tools import McpSharedConnection, WorkspaceMcpServer, WorkspaceMcpServerOverride
from gateway.repositories.tenancy import OrganizationRepository, WorkspaceMemberRepository, WorkspaceRepository
from gateway.repositories.tools import WorkspaceMcpServerOverrideRepository, WorkspaceMcpServerRepository
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.workspace_mcp_server_service import (
    WorkspaceMcpServerCreate,
    WorkspaceMcpServers,
    WorkspaceMcpServerUpdate,
)
from gateway.services.tools import (
    AttachedMcpServerIndex,
    OrganizationMcpServerLimit,
    OrganizationMcpServerWorkspaceService,
)

from . import test_mcp_user_connections as base
from .test_mcp_user_connections import PUBLIC_URL, _member, _service, _state, _World, _world

# The scaffolding's fixtures, taken by assignment: pytest finds a fixture by its name in this module.
wire = base.wire
config = base.config
checks = base.checks
_secret_key = base._secret_key
attached_index_on_async_db = base.attached_index_on_async_db

pytestmark = pytest.mark.asyncio


def _workspaces(db: AsyncSession) -> OrganizationMcpServerWorkspaceService:
    uow = UnitOfWork(db)
    return OrganizationMcpServerWorkspaceService(
        uow,
        servers=WorkspaceMcpServerRepository(uow),
        overrides=WorkspaceMcpServerOverrideRepository(uow),
        organizations=OrganizationService(db, membership_listener=None),
        lock_workspace=WorkspaceRepository(db).lock,
    )


async def _sales(db: AsyncSession, world: _World) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Sales", organization_id=world.workspace.organization_id, created_by_user_id=world.owner.id
    )
    await db.commit()
    return workspace


async def _add(db: AsyncSession, config: GatewayConfig, user: User, **fields: object) -> uuid.UUID:
    values: dict[str, object] = {"name": "jira", "url": PUBLIC_URL, "authorization_token": "org-token"}
    values.update(fields)
    created = await _service(db, config).create_server(
        user=user,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(**values),  # type: ignore[arg-type]
    )
    return created.id


async def test_an_owner_adds_changes_and_removes_an_organization_server(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    service = _service(async_db, config)

    jira = await _add(async_db, config, world.owner)
    listed = await service.list_organization_servers(user=world.owner)
    renamed = await service.update_server(
        user=world.owner, workspace_id=None, server_id=jira, request=WorkspaceMcpServerUpdate(name="tracker")
    )

    assert [(server.id, server.scope, server.workspace_id) for server in listed.data] == [(jira, "organization", None)]
    assert renamed.name == "tracker"
    workspace_list = await service.list_servers(user=world.owner, workspace_id=world.workspace.id)
    assert jira not in [server.id for server in workspace_list.data], "the workspace's page lists its own: PR 4"
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.update_server(
            user=world.owner, workspace_id=world.workspace.id, server_id=jira, request=WorkspaceMcpServerUpdate()
        )
    await service.delete_server(user=world.owner, workspace_id=None, server_id=jira)
    assert (await service.list_organization_servers(user=world.owner)).data == []


async def test_only_the_organization_s_owners_and_admins_manage_its_servers(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _add(async_db, config, world.owner)
    # An admin of the workspace who is a plain member of the organization.
    workspace_admin = await _member(async_db, await _organization_of(async_db, world), role="member", name="Lead")
    await WorkspaceMemberRepository(async_db).create(
        workspace_id=world.workspace.id, user_id=workspace_admin.id, role="admin"
    )
    await async_db.commit()
    service = _service(async_db, config)

    for user in (world.member, workspace_admin):
        with pytest.raises(TenancyError):
            await service.list_organization_servers(user=user)
        with pytest.raises(TenancyError):
            await service.update_server(
                user=user, workspace_id=None, server_id=jira, request=WorkspaceMcpServerUpdate(name="mine")
            )
        with pytest.raises(TenancyError):
            await _workspaces(async_db).workspaces(user=user, server_id=jira)


async def test_a_create_that_waited_behind_a_demotion_is_refused(
    async_db: AsyncSession, config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The role is read as stored once the lock is held, not from the session's copy of the membership.

    The caller's membership is already in the session when the create starts. A demotion holding the
    organization's lock commits just before the create is granted it (simulated on a session of its own),
    and a read answered from the session's copy would still see an admin.
    """
    world = await _world(async_db, config)
    admin = await _member(async_db, await _organization_of(async_db, world), role="admin", name="Admin")
    await async_db.commit()
    admin_id, organization_id = admin.id, world.workspace.organization_id
    # As the route's own permission check leaves it: the membership loaded in this session.
    loaded = await OrganizationService(async_db, membership_listener=None).members.get_active_by_organization_and_user(
        organization_id, admin_id
    )
    assert loaded is not None and loaded.role == "admin"
    original = OrganizationRepository.lock_settings

    async def demoted_while_waiting(self: OrganizationRepository, organization_id: uuid.UUID) -> None:
        async with AsyncSession(async_db.bind, expire_on_commit=False) as other:
            await other.execute(
                update(OrganizationMember).where(col(OrganizationMember.user_id) == admin_id).values(role="member")
            )
            await other.commit()
        await original(self, organization_id)

    monkeypatch.setattr(OrganizationRepository, "lock_settings", demoted_while_waiting)
    with pytest.raises(NotAuthorizedError):
        await _add(async_db, config, admin)
    await async_db.rollback()

    stored = await async_db.execute(select(WorkspaceMcpServer.id).where(col(WorkspaceMcpServer.workspace_id).is_(None)))
    assert stored.scalars().all() == [], "nothing stored"


async def _organization_of(db: AsyncSession, world: _World):  # type: ignore[no-untyped-def]
    organization = await OrganizationRepository(db).get(world.workspace.organization_id)
    assert organization is not None
    return organization


async def test_an_organization_server_cannot_take_a_name_any_of_its_workspaces_uses(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)

    await _add(async_db, config, world.owner, name="jira")

    with pytest.raises(McpServerNameTakenError):
        await _add(async_db, config, world.owner, name="drive")
    with pytest.raises(McpServerNameTakenError):
        await _add(async_db, config, world.owner, name="jira")


async def test_the_organization_limits_one_workspace_and_lifting_it_leaves_no_row(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    sales = await _sales(async_db, world)
    jira = await _add(async_db, config, world.owner)
    service = _workspaces(async_db)

    before = await service.workspaces(user=world.owner, server_id=jira)
    limited = await service.set_limit(
        user=world.owner, server_id=jira, workspace_id=sales.id, limit=OrganizationMcpServerLimit(tools=["@read"])
    )
    off = await service.set_limit(
        user=world.owner, server_id=jira, workspace_id=world.workspace.id, limit=OrganizationMcpServerLimit(tools=[])
    )
    after = await service.workspaces(user=world.owner, server_id=jira)

    assert [(row.name, row.reached, row.organization_tools) for row in before.data] == [
        ("Default", True, None),
        ("Sales", True, None),
    ]
    assert (limited.reached, limited.organization_tools) == (True, ["@read"])
    assert (off.reached, off.organization_tools) == (False, [])
    assert [(row.name, row.reached) for row in after.data] == [("Default", False), ("Sales", True)]

    cleared = await service.clear_limit(user=world.owner, server_id=jira, workspace_id=sales.id)
    rows = (await async_db.execute(select(WorkspaceMcpServerOverride.workspace_id))).scalars().all()
    assert (cleared.reached, cleared.organization_tools) == (True, None)
    assert rows == [world.workspace.id]


async def test_lifting_the_limit_keeps_what_the_workspace_set_itself(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _add(async_db, config, world.owner)
    async_db.add(
        WorkspaceMcpServerOverride(
            workspace_id=world.workspace.id,
            organization_id=world.workspace.organization_id,
            server_id=jira,
            organization_tools=["search*"],
            everyone_tools=["search_issues"],
        )
    )
    await async_db.commit()

    cleared = await _workspaces(async_db).clear_limit(user=world.owner, server_id=jira, workspace_id=world.workspace.id)

    assert (cleared.organization_tools, cleared.narrowed) == (None, True)
    rows = (await async_db.execute(select(WorkspaceMcpServerOverride.everyone_tools))).scalars().all()
    assert rows == [["search_issues"]]


async def test_a_limit_names_a_workspace_of_the_organization_and_one_of_its_own_servers(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _add(async_db, config, world.owner)
    other = await OrganizationRepository(async_db).create_organization(
        name="Globex", slug="globex", created_by_user_id=None
    )
    foreign = await WorkspaceRepository(async_db).create_workspace(
        name="Default", organization_id=other.id, created_by_user_id=None
    )
    await async_db.commit()
    service = _workspaces(async_db)

    with pytest.raises(WorkspaceNotFoundError):
        await service.set_limit(
            user=world.owner, server_id=jira, workspace_id=foreign.id, limit=OrganizationMcpServerLimit(tools=[])
        )
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.workspaces(user=world.owner, server_id=world.server.id)


async def test_an_owner_signs_in_for_everyone_on_an_organization_server(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _add(async_db, config, world.owner, authorization_token=None)
    service = _service(async_db, config)

    started = await service.start_shared_sign_in(user=world.owner, workspace_id=None, server_id=jira, return_to=None)
    done = await service.complete_connection(user=world.owner, state=_state(started.authorization_url), code="code")

    assert (done.server_id, done.workspace_id, done.shared) == (jira, None, True)
    accounts = (await async_db.execute(select(McpSharedConnection.server_id))).scalars().all()
    assert accounts == [jira]


@pytest.mark.usefixtures("attached_index_on_async_db")
async def test_every_organization_server_write_reaches_the_attached_server_index(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    AttachedMcpServerIndex.reset()
    world = await _world(async_db, config)
    service = _service(async_db, config)
    servers = WorkspaceMcpServers(async_db)
    organization_id = world.workspace.organization_id
    # The scaffolding's own workspace server attaches nothing.
    await service.update_server(
        user=world.owner,
        workspace_id=world.workspace.id,
        server_id=world.server.id,
        request=WorkspaceMcpServerUpdate(attach_by_default=False),
    )

    async def attached() -> bool:
        return await servers.any_attached(world.workspace.id, organization_id)

    jira = await _add(async_db, config, world.owner, attach_by_default=True)
    assert await attached(), "added attached to every request"
    ids = {"user": world.owner, "workspace_id": None, "server_id": jira}
    await service.update_server(**ids, request=WorkspaceMcpServerUpdate(enabled=False))  # type: ignore[arg-type]
    assert not await attached(), "turned off"
    await service.update_server(**ids, request=WorkspaceMcpServerUpdate(enabled=True))  # type: ignore[arg-type]
    assert await attached(), "turned back on"
    await service.update_server(**ids, request=WorkspaceMcpServerUpdate(attach_by_default=False))  # type: ignore[arg-type]
    assert not await attached(), "no longer attached"
    await service.update_server(**ids, request=WorkspaceMcpServerUpdate(attach_by_default=True))  # type: ignore[arg-type]
    assert await attached()
    await service.delete_server(user=world.owner, workspace_id=None, server_id=jira)
    assert not await attached(), "deleted"


@pytest.mark.parametrize(
    ("url", "resolves_to"),
    [
        ("http://127.0.0.1:8000/mcp", None),
        ("http://[::1]:8000/mcp", None),
        ("http://localhost:8000/mcp", "127.0.0.1"),
        ("https://mcp.rebound.example/mcp", "127.0.0.1"),
    ],
)
async def test_an_organization_server_on_an_internal_address_is_refused_when_saved(
    async_db: AsyncSession, config: GatewayConfig, monkeypatch: pytest.MonkeyPatch, url: str, resolves_to: str | None
) -> None:
    """An organization's URL is a tenant's, as a workspace's is, so it must be public."""
    if resolves_to is not None:

        async def resolve(_host: str) -> list[object]:
            return [ipaddress.ip_address(resolves_to)]

        monkeypatch.setattr("gateway.services.url_safety._resolve_all_async", resolve)
    world = await _world(async_db, config)

    with pytest.raises(WorkspaceMcpServerUnsafeUrlError):
        await _add(async_db, config, world.owner, url=url)
    jira = await _add(async_db, config, world.owner)
    with pytest.raises(WorkspaceMcpServerUnsafeUrlError):
        await _service(async_db, config).update_server(
            user=world.owner, workspace_id=None, server_id=jira, request=WorkspaceMcpServerUpdate(url=url)
        )
