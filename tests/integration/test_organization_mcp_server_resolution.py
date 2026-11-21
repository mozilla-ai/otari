"""An organization's MCP servers as its workspaces' requests, sign-ins and memberships reach them.

No route makes an organization server yet, so each one here is written directly.
"""

import uuid

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.tools_exceptions import (
    McpConnectionFlowError,
    McpServerNameTakenError,
    WorkspaceMcpServerAccessDeniedError,
    WorkspaceMcpServerNotFoundError,
)
from gateway.models.tenancy import Organization, OrganizationMember, User, Workspace
from gateway.models.tools import (
    McpCredentialMode,
    McpOAuthFlow,
    McpServerAlias,
    McpUserConnection,
    WorkspaceMcpServer,
    WorkspaceMcpServerGrant,
    WorkspaceMcpServerOverride,
)
from gateway.repositories.tenancy import OrganizationRepository, WorkspaceMemberRepository, WorkspaceRepository
from gateway.repositories.tools import (
    McpServerAliasRepository,
    WorkspaceMcpServerOverrideRepository,
    WorkspaceMcpServerRepository,
)
from gateway.repositories.users_repository import provision_attribution_users
from gateway.services.model_access import McpAccess, resolve_request_mcp_access
from gateway.services.secret_box import encrypt_secret
from gateway.services.tenancy.workspace_mcp_server_service import (
    WorkspaceMcpServerCreate,
    WorkspaceMcpServers,
    WorkspaceMcpServerUpdate,
    resolve_workspace_mcp_servers,
)
from gateway.services.tools import (
    AttachedMcpServerIndex,
    McpConnectionService,
    McpGrantMembershipListener,
    McpWorkspaceServers,
)

from . import test_mcp_user_connections as base
from .test_mcp_user_connections import PUBLIC_URL, _connections, _reach, _service, _state, _World, _world

# The scaffolding's fixtures, taken by assignment: pytest finds a fixture by its name in this module.
wire = base.wire
config = base.config
checks = base.checks
_secret_key = base._secret_key

pytestmark = pytest.mark.asyncio


async def _shared(db: AsyncSession, organization_id: uuid.UUID, **fields: object) -> uuid.UUID:
    """An organization server named ``jira``, sending a token, unless ``fields`` say otherwise."""
    values: dict[str, object] = {"name": "jira", "url": PUBLIC_URL, "encrypted_token": encrypt_secret("org-token")}
    values.update(fields)
    server = WorkspaceMcpServer(organization_id=organization_id, workspace_id=None, **values)
    db.add(server)
    await db.commit()
    return server.id


async def _override(db: AsyncSession, workspace: Workspace, server_id: uuid.UUID, **fields: object) -> None:
    db.add(
        WorkspaceMcpServerOverride(
            workspace_id=workspace.id, organization_id=workspace.organization_id, server_id=server_id, **fields
        )
    )
    await db.commit()


async def _another_workspace(db: AsyncSession, world: _World, *members: User) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Sales", organization_id=world.workspace.organization_id, created_by_user_id=world.owner.id
    )
    for member in members:
        await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=member.id, role="member")
    await db.commit()
    return workspace


async def _foreign_workspace(db: AsyncSession) -> Workspace:
    other: Organization = await OrganizationRepository(db).create_organization(
        name="Globex", slug="globex", created_by_user_id=None
    )
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Default", organization_id=other.id, created_by_user_id=None
    )
    await db.commit()
    return workspace


async def test_a_workspace_reaches_its_organization_s_server_and_no_other_organization_does(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    foreign = await _foreign_workspace(async_db)

    resolved = await resolve_workspace_mcp_servers(
        async_db, workspace_id=world.workspace.id, server_ids=[jira], access=McpAccess()
    )

    assert [(config.name, config.authorization_token) for config in resolved] == [("jira", "org-token")]
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await resolve_workspace_mcp_servers(async_db, workspace_id=foreign.id, server_ids=[jira], access=McpAccess())


async def test_the_organization_s_limit_clips_every_rule_set_in_the_workspace(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id, read_tools=["list_issues"])
    await _override(async_db, world.workspace, jira, organization_tools=["@read", "search*"], everyone_tools=["*"])
    granted = McpAccess(user_id="ada", key_grants={jira: ("search_issues", "delete_issue", "list_issues")})

    [everyone] = await resolve_workspace_mcp_servers(
        async_db, workspace_id=world.workspace.id, server_ids=[jira], access=McpAccess(user_id="ada")
    )
    [keyed] = await resolve_workspace_mcp_servers(
        async_db, workspace_id=world.workspace.id, server_ids=[jira], access=granted
    )
    resolved = await WorkspaceMcpServers(async_db).resolve_one(world.workspace.id, jira, McpAccess(user_id="ada"))

    assert everyone.caller_tool_patterns == ["search*", "list_issues"]
    assert keyed.caller_tool_patterns == ["search_issues", "list_issues"]
    assert resolved is not None
    assert resolved.tool_limit == ["search*", "list_issues"]
    assert resolved.everyone_tools == ["*"]


@pytest.mark.parametrize(
    ("server_fields", "override_fields"),
    [
        ({"enabled": False}, None),
        ({}, {"enabled": False}),
        ({}, {"organization_tools": []}),
    ],
    ids=["organization switched it off", "workspace switched it off", "organization limit is nothing"],
)
async def test_a_server_turned_off_for_a_workspace_is_unknown_there(
    async_db: AsyncSession,
    config: GatewayConfig,
    server_fields: dict[str, bool],
    override_fields: dict[str, object] | None,
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id, **server_fields)
    if override_fields is not None:
        await _override(async_db, world.workspace, jira, **override_fields)

    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await resolve_workspace_mcp_servers(
            async_db, workspace_id=world.workspace.id, server_ids=[jira], access=McpAccess()
        )
    assert await WorkspaceMcpServers(async_db).resolve_one(world.workspace.id, jira, McpAccess()) is None


async def test_a_workspace_s_own_setting_decides_whether_it_attaches_an_organization_server(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    attached_by_organization = await _shared(async_db, world.workspace.organization_id, attach_by_default=True)
    attached_here = await _shared(async_db, world.workspace.organization_id, name="sentry")
    servers = WorkspaceMcpServers(async_db)

    async def attached() -> list[str]:
        await _refresh_attached(async_db)
        assert await servers.any_attached(world.workspace.id), "the organization attaches one, so the request reads"
        reached = await servers.resolve_for_request(world.workspace.id, [], McpAccess(), attach=True)
        return [server.name for server in reached.servers]

    assert await attached() == ["jira"]
    await _override(async_db, world.workspace, attached_by_organization, attach_by_default=False)
    assert await attached() == []
    await _override(async_db, world.workspace, attached_here, attach_by_default=True)
    assert await attached() == ["sentry"]


async def test_an_old_id_names_the_server_it_was_folded_into_in_its_own_workspace_only(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    sales = await _another_workspace(async_db, world)
    jira = await _shared(async_db, world.workspace.organization_id)
    old_id = uuid.uuid4()
    async_db.add(
        McpServerAlias(
            old_id=old_id,
            organization_id=world.workspace.organization_id,
            workspace_id=world.workspace.id,
            server_id=jira,
        )
    )
    await async_db.commit()

    resolved = await resolve_workspace_mcp_servers(
        async_db, workspace_id=world.workspace.id, server_ids=[old_id], access=McpAccess()
    )

    assert [config.stored_server_id for config in resolved] == [str(jira)]
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await resolve_workspace_mcp_servers(async_db, workspace_id=sales.id, server_ids=[old_id], access=McpAccess())


async def test_a_grant_on_an_organization_server_applies_in_its_own_workspace_only(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    sales = await _another_workspace(async_db, world, world.member)
    jira = await _shared(async_db, world.workspace.organization_id)
    member = str(world.member.id)
    await provision_attribution_users(async_db, [member])
    async_db.add(WorkspaceMcpServerGrant(server_id=jira, workspace_id=world.workspace.id, user_id=member, tools=[]))
    await async_db.commit()

    here = await resolve_request_mcp_access(async_db, None, world.workspace.id, session_user_id=member)
    there = await resolve_request_mcp_access(async_db, None, sales.id, session_user_id=member)

    assert here.user_grants == {jira: ()}
    assert there.user_grants == {}


async def test_a_workspace_server_cannot_take_a_name_its_organization_s_server_has(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    await _shared(async_db, world.workspace.organization_id)
    service = _service(async_db, config)

    with pytest.raises(McpServerNameTakenError):
        await service.update_server(
            user=world.owner,
            workspace_id=world.workspace.id,
            server_id=world.server.id,
            request=WorkspaceMcpServerUpdate(name="jira"),
        )
    with pytest.raises(McpServerNameTakenError):
        await service.create_server(
            user=world.owner,
            workspace_id=world.workspace.id,
            request=WorkspaceMcpServerCreate(name="jira", url=PUBLIC_URL),
        )


async def _per_user_shared(db: AsyncSession, world: _World) -> uuid.UUID:
    return await _shared(
        db, world.workspace.organization_id, encrypted_token=None, credential_mode=McpCredentialMode.PER_USER
    )


async def _start(
    db: AsyncSession, config: GatewayConfig, server_id: uuid.UUID, identity_id: uuid.UUID, workspace_id: uuid.UUID
) -> str:
    await provision_attribution_users(db, [str(identity_id)])
    await db.commit()
    return await _connections(db, config).start(
        server_id=server_id, user_id=str(identity_id), identity_id=identity_id, workspace_id=workspace_id
    )


async def test_a_sign_in_to_an_organization_server_starts_only_from_a_workspace_that_reaches_it(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _per_user_shared(async_db, world)
    sales = await _another_workspace(async_db, world, world.member)
    await _override(async_db, sales, jira, enabled=False)
    foreign = await _foreign_workspace(async_db)

    member_id, workspace_id = world.member.id, world.workspace.id
    url = await _start(async_db, config, jira, member_id, workspace_id)

    flows = (await async_db.execute(select(McpOAuthFlow.workspace_id))).scalars().all()
    assert flows == [workspace_id]
    for elsewhere in (sales.id, foreign.id):
        with pytest.raises(WorkspaceMcpServerNotFoundError):
            await _start(async_db, config, jira, member_id, elsewhere)
    member = await async_db.get(User, member_id)
    assert member is not None
    done = await _service(async_db, config).complete_connection(user=member, state=_state(url), code="code")
    assert (done.workspace_id, done.server_scope) == (workspace_id, "organization"), (
        "the workspace it started from, and the server named as the organization's, which has no page there"
    )
    stored = (await async_db.execute(select(McpUserConnection.server_id))).scalars().all()
    assert stored == [jira]


async def test_a_sign_in_to_an_organization_server_lasts_while_any_workspace_still_reaches_it(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _per_user_shared(async_db, world)
    sales = await _another_workspace(async_db, world, world.member)
    url = await _start(async_db, config, jira, world.member.id, world.workspace.id)
    await _service(async_db, config).complete_connection(user=world.member, state=_state(url), code="code")
    member = str(world.member.id)
    for workspace in (world.workspace, sales):
        async_db.add(WorkspaceMcpServerGrant(server_id=jira, workspace_id=workspace.id, user_id=member, tools=["*"]))
    await async_db.commit()

    async def leave(workspace: Workspace) -> None:
        """Remove the member from the workspace as the workspace route does: the listener, then the membership."""
        async with UnitOfWork(async_db) as uow:
            members = WorkspaceMemberRepository(uow)
            membership = await members.get_by_workspace_and_user(workspace.id, world.member.id)
            assert membership is not None
            await McpGrantMembershipListener(uow, _reach(uow)).member_removed(membership)
            await members.delete(membership)

    async def held() -> tuple[list[uuid.UUID], list[uuid.UUID]]:
        grants = (await async_db.execute(select(WorkspaceMcpServerGrant.workspace_id))).scalars().all()
        connections = (await async_db.execute(select(McpUserConnection.server_id))).scalars().all()
        return list(grants), list(connections)

    await leave(world.workspace)
    assert await held() == ([sales.id], [jira])
    await leave(sales)
    assert await held() == ([], [])


async def test_a_hidden_server_an_old_id_names_is_refused_under_the_id_the_caller_sent(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    await _override(async_db, world.workspace, jira, enabled=False)
    old_id = uuid.uuid4()
    async_db.add(
        McpServerAlias(
            old_id=old_id,
            organization_id=world.workspace.organization_id,
            workspace_id=world.workspace.id,
            server_id=jira,
        )
    )
    await async_db.commit()

    with pytest.raises(WorkspaceMcpServerNotFoundError) as refused:
        await resolve_workspace_mcp_servers(
            async_db, workspace_id=world.workspace.id, server_ids=[old_id], access=McpAccess()
        )

    assert str(old_id) in str(refused.value)
    assert str(jira) not in str(refused.value)


async def test_a_refusal_the_organization_s_limit_causes_is_the_organization_s(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    await _override(async_db, world.workspace, jira, organization_tools=["list_issues"])
    granted = McpAccess(user_id="ada", key_grants={jira: ("delete_issue",)})
    denied = McpAccess(user_id="ada", key_grants={jira: ()})
    servers = WorkspaceMcpServers(async_db)

    with pytest.raises(WorkspaceMcpServerAccessDeniedError) as by_limit:
        await servers.resolve_one(world.workspace.id, jira, granted)
    with pytest.raises(WorkspaceMcpServerAccessDeniedError) as by_key:
        await servers.resolve_one(world.workspace.id, jira, denied)

    assert (by_limit.value.decided_by, by_key.value.decided_by) == ("mcp_organization", "mcp_key")


async def test_a_link_s_page_hides_a_server_its_workspace_no_longer_reaches(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _per_user_shared(async_db, world)
    member = str(world.member.id)
    await provision_attribution_users(async_db, [member])
    await async_db.commit()
    link = McpConnectionService.sign_in_link(config, server_id=jira, user_id=member, workspace_id=world.workspace.id)
    ticket = link.split("ticket=")[1]
    connections = _connections(async_db, config)
    assert (await connections.read_link(ticket)).server_id == jira

    await _override(async_db, world.workspace, jira, enabled=False)

    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await connections.read_link(ticket)


async def test_a_sign_in_started_from_one_workspace_leaves_another_s_pending(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _per_user_shared(async_db, world)
    sales = await _another_workspace(async_db, world, world.member)
    member = str(world.member.id)
    await provision_attribution_users(async_db, [member])
    await async_db.commit()
    connections = _connections(async_db, config)

    for n, workspace_id in enumerate((world.workspace.id, sales.id, sales.id)):
        await connections.start(server_id=jira, user_id=member, browser_token=f"browser-{n}", workspace_id=workspace_id)

    flows = (await async_db.execute(select(McpOAuthFlow.workspace_id))).scalars().all()
    assert sorted(map(str, flows)) == sorted(map(str, [world.workspace.id, sales.id])), "one per workspace"


async def test_deleting_a_member_s_last_workspace_ends_their_organization_server_sign_in(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _per_user_shared(async_db, world)
    url = await _start(async_db, config, jira, world.member.id, world.workspace.id)
    await _service(async_db, config).complete_connection(user=world.member, state=_state(url), code="code")
    workspace_id = world.workspace.id
    member_ids = await WorkspaceMemberRepository(async_db).ids_for_workspace(workspace_id)

    async with UnitOfWork(async_db) as uow:
        await McpGrantMembershipListener(uow, _reach(uow)).workspace_deleted(workspace_id, member_ids)

    connections = (await async_db.execute(select(McpUserConnection.user_id))).scalars().all()
    assert connections == [], "the member reached the organization only through that workspace"


async def test_a_member_removes_their_own_connection_to_an_organization_server(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _per_user_shared(async_db, world)
    member_id, workspace_id = world.member.id, world.workspace.id
    url = await _start(async_db, config, jira, member_id, workspace_id)
    member = await async_db.get(User, member_id)
    assert member is not None
    service = _service(async_db, config)
    await service.complete_connection(user=member, state=_state(url), code="code")
    foreign = await _foreign_workspace(async_db)
    stranger_organization = await OrganizationRepository(async_db).get(foreign.organization_id)
    assert stranger_organization is not None
    stranger = await base._member(async_db, stranger_organization, role="owner", name="Stranger")
    await WorkspaceMemberRepository(async_db).create(workspace_id=foreign.id, user_id=stranger.id, role="owner")
    await async_db.commit()

    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.disconnect(user=stranger, workspace_id=foreign.id, server_id=jira)
    assert (await async_db.execute(select(McpUserConnection.server_id))).scalars().all() == [jira]

    await service.disconnect(user=member, workspace_id=workspace_id, server_id=jira)

    assert (await async_db.execute(select(McpUserConnection.id))).scalars().all() == []


async def test_a_sign_in_finishing_after_its_workspace_turned_the_server_off_stores_nothing(
    async_db: AsyncSession, config: GatewayConfig, wire: base._Wire
) -> None:
    world = await _world(async_db, config)
    jira = await _per_user_shared(async_db, world)
    member_id = world.member.id
    url = await _start(async_db, config, jira, member_id, world.workspace.id)
    await _override(async_db, world.workspace, jira, enabled=False)
    member = await async_db.get(User, member_id)
    assert member is not None

    with pytest.raises(McpConnectionFlowError, match="no longer available"):
        await _service(async_db, config).complete_connection(user=member, state=_state(url), code="code")

    assert (await async_db.execute(select(McpUserConnection.id))).scalars().all() == []
    await base._drain_revocations()
    assert wire.revoked == ["refresh-1"], "the token the exchange issued is ended upstream"


async def test_a_sign_in_finishing_after_its_member_was_suspended_from_the_organization_stores_nothing(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _per_user_shared(async_db, world)
    member_id = world.member.id
    url = await _start(async_db, config, jira, member_id, world.workspace.id)
    # Suspension leaves the workspace's own membership row as it was.
    await async_db.execute(
        update(OrganizationMember).where(col(OrganizationMember.user_id) == member_id).values(status="suspended")
    )
    await async_db.commit()
    member = await async_db.get(User, member_id)
    assert member is not None

    with pytest.raises(McpConnectionFlowError, match="no longer have access"):
        await _service(async_db, config).complete_connection(user=member, state=_state(url), code="code")

    assert (await async_db.execute(select(McpUserConnection.id))).scalars().all() == []


async def _refresh_attached(db: AsyncSession) -> None:
    await AttachedMcpServerIndex.refresh(WorkspaceMcpServerRepository(db), WorkspaceMcpServerOverrideRepository(db))


async def test_an_organization_server_attached_to_every_request_counts_in_its_workspaces(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    AttachedMcpServerIndex.reset()
    world = await _world(async_db, config)
    await async_db.execute(update(WorkspaceMcpServer).values(attach_by_default=False))
    organization_id = world.workspace.organization_id
    jira = await _shared(async_db, organization_id)
    sales = await _another_workspace(async_db, world)
    servers = WorkspaceMcpServers(async_db)

    async def attached(workspace: Workspace) -> bool:
        await _refresh_attached(async_db)
        return await servers.any_attached(workspace.id, organization_id)

    assert not await attached(world.workspace)
    await async_db.execute(
        update(WorkspaceMcpServer).where(col(WorkspaceMcpServer.id) == jira).values(attach_by_default=True)
    )
    await async_db.commit()
    assert await attached(world.workspace) and await attached(sales), "the organization's server, in every workspace"
    assert await servers.any_attached(sales.id), "the organization is read when the caller has none"
    await async_db.execute(update(WorkspaceMcpServer).where(col(WorkspaceMcpServer.id) == jira).values(enabled=False))
    await async_db.commit()
    assert not await attached(world.workspace), "turned off for the organization"


async def test_a_workspace_s_own_setting_attaches_an_organization_server_there_alone(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    AttachedMcpServerIndex.reset()
    world = await _world(async_db, config)
    await async_db.execute(update(WorkspaceMcpServer).values(attach_by_default=False))
    organization_id = world.workspace.organization_id
    jira = await _shared(async_db, organization_id)
    sales = await _another_workspace(async_db, world)
    await _override(async_db, sales, jira, attach_by_default=True)
    await _refresh_attached(async_db)
    servers = WorkspaceMcpServers(async_db)

    assert await servers.any_attached(sales.id, organization_id)
    assert not await servers.any_attached(world.workspace.id, organization_id)
    # The request then reads the server as the workspace sees it, and attaches it.
    views, _ = await McpWorkspaceServers(
        WorkspaceMcpServerRepository(async_db),
        WorkspaceMcpServerOverrideRepository(async_db),
        McpServerAliasRepository(async_db),
    ).for_request(sales.id, organization_id, [], attached=True)
    assert [view.row.id for view in views if view.attach_by_default] == [jira]
