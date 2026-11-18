"""A workspace's side of its organization's MCP servers: listing them, its own settings, grants, and sign-ins.

Exercised at the service layer, as ``test_workspace_mcp_servers.py`` is. Organization servers are written directly.
"""

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps.organizations import get_user_access
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.tools_exceptions import McpServerIsOrganizationsError, WorkspaceMcpServerNotFoundError
from gateway.models.api_keys import APIKey
from gateway.models.tools import McpUserConnection, WorkspaceMcpServerOverride
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.services.api_keys import ApiKeyService
from gateway.services.model_access import McpAccess
from gateway.services.tenancy.workspace_mcp_server_service import (
    WorkspaceMcpServerGrantsCreate,
    WorkspaceMcpServers,
    WorkspaceMcpServerService,
    WorkspaceMcpServerSettingsUpdate,
    WorkspaceMcpServerUpdate,
    resolve_workspace_mcp_servers,
)
from gateway.services.tools import AttachedMcpServerIndex

from . import test_mcp_user_connections as base
from .test_mcp_user_connections import _connections, _state, _World, _world
from .test_organization_mcp_server_resolution import _another_workspace, _foreign_workspace, _override, _shared
from .test_workspace_mcp_servers import _key

# The scaffolding's fixtures, taken by assignment: pytest finds a fixture by its name in this module.
wire = base.wire
config = base.config
checks = base.checks
_secret_key = base._secret_key
attached_index_on_async_db = base.attached_index_on_async_db

pytestmark = pytest.mark.asyncio


def _service(db: AsyncSession, config: GatewayConfig) -> WorkspaceMcpServerService:
    uow = UnitOfWork(db)
    keys = ApiKeyService(uow, ApiKeyRepositories.on(uow), users=get_user_access(uow))
    return WorkspaceMcpServerService(db, keys, connections=_connections(db, config), config=config)


async def _member_key(db: AsyncSession, world: _World) -> APIKey:
    key = await _key(db, world.workspace.id, user_id=str(world.member.id))
    await db.commit()
    return key


async def test_a_workspace_lists_its_organization_s_servers_beside_its_own(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    hidden = await _shared(async_db, world.workspace.organization_id, name="sentry")
    await _override(async_db, world.workspace, hidden, enabled=False)
    service = _service(async_db, config)

    managed = await service.list_servers(user=world.owner, workspace_id=world.workspace.id)
    seen = await service.list_servers(user=world.member, workspace_id=world.workspace.id)

    assert [(server.name, server.scope) for server in managed.data] == [
        ("drive", "workspace"),
        ("jira", "organization"),
        ("sentry", "organization"),
    ]
    assert [server.name for server in seen.data] == ["drive", "jira"], "a member sees only what the workspace reaches"
    sentry = managed.data[2]
    assert sentry.in_workspace is not None and (sentry.in_workspace.reached, sentry.in_workspace.enabled) == (
        False,
        False,
    )
    assert managed.data[0].in_workspace is None
    assert jira in [server.id for server in seen.data]


async def test_a_workspace_changes_only_its_own_settings_on_its_organization_s_server(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id, attach_by_default=True)
    await _override(async_db, world.workspace, jira, organization_tools=["list_*"])
    service = _service(async_db, config)
    ids = {"workspace_id": world.workspace.id, "server_id": jira}

    with pytest.raises(McpServerIsOrganizationsError):
        await service.update_server(user=world.owner, request=WorkspaceMcpServerUpdate(name="mine"), **ids)
    narrowed = await service.set_workspace_settings(
        user=world.owner,
        request=WorkspaceMcpServerSettingsUpdate(everyone_tools=["*"], attach_by_default=False),
        **ids,
    )
    [resolved] = await resolve_workspace_mcp_servers(
        async_db, workspace_id=world.workspace.id, server_ids=[jira], access=McpAccess(user_id="ada")
    )
    assert narrowed.in_workspace is not None
    assert (narrowed.in_workspace.everyone_tools, narrowed.in_workspace.attach_by_default) == (["*"], False)
    assert resolved.caller_tool_patterns == ["list_*"], "a workspace rule of every tool stays under the limit"

    off = await service.set_workspace_settings(
        user=world.owner, request=WorkspaceMcpServerSettingsUpdate(enabled=False), **ids
    )
    assert off.in_workspace is not None and off.in_workspace.reached is False
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await resolve_workspace_mcp_servers(
            async_db, workspace_id=world.workspace.id, server_ids=[jira], access=McpAccess()
        )

    cleared = await service.clear_workspace_settings(user=world.owner, **ids)
    assert cleared.in_workspace is not None
    assert (cleared.in_workspace.enabled, cleared.in_workspace.organization_tools) == (True, ["list_*"])
    rows = (await async_db.execute(select(WorkspaceMcpServerOverride.organization_tools))).scalars().all()
    assert rows == [["list_*"]], "the organization's limit stays"
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.set_workspace_settings(
            user=world.owner,
            workspace_id=world.workspace.id,
            server_id=world.server.id,
            request=WorkspaceMcpServerSettingsUpdate(),
        )


async def test_a_grant_on_an_organization_server_is_one_workspace_s(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    sales = await _another_workspace(async_db, world, world.member)
    jira = await _shared(async_db, world.workspace.organization_id)
    key = await _member_key(async_db, world)
    service = _service(async_db, config)
    member = str(world.member.id)

    here = await service.add_grants(
        user=world.owner,
        workspace_id=world.workspace.id,
        server_id=jira,
        request=WorkspaceMcpServerGrantsCreate(user_ids=[member], tools=["list_issues"]),
    )
    there = await service.access(user=world.owner, workspace_id=sales.id, server_id=jira)
    check = await service.check_access(
        user=world.owner, workspace_id=world.workspace.id, server_id=jira, api_key_id=key.id, tool="list_issues"
    )

    assert [(grant.user_id, grant.tools) for grant in here.grants] == [(member, ["list_issues"])]
    assert there.grants == [], "a grant made in one workspace is not another's"
    assert (check.allowed, check.decided_by) == (True, "user")


async def test_the_organization_s_limit_decides_what_a_workspace_rule_cannot_reach(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    await _override(async_db, world.workspace, jira, organization_tools=["list_issues"])
    key = await _member_key(async_db, world)
    service = _service(async_db, config)

    check = await service.check_access(
        user=world.owner, workspace_id=world.workspace.id, server_id=jira, api_key_id=key.id, tool="delete_issue"
    )
    access = await service.key_access(user=world.owner, workspace_id=world.workspace.id, api_key_id=key.id)

    assert (check.allowed, check.decided_by) == (False, "organization")
    assert {row.name: row.organization_tools for row in access} == {"drive": None, "jira": ["list_issues"]}


async def test_two_admins_changing_different_settings_keep_both_changes(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    """Each PATCH sends only what its control changed, so the second, from a view that missed the first, keeps it."""
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    ids = {"workspace_id": world.workspace.id, "server_id": jira}
    first = _service(async_db, config)
    async with AsyncSession(async_db.bind, expire_on_commit=False) as other:
        second = _service(other, config)
        stale = await second.get_server(user=world.owner, **ids)
        assert stale.in_workspace is not None and stale.in_workspace.enabled is True

        await first.set_workspace_settings(
            user=world.owner, request=WorkspaceMcpServerSettingsUpdate.model_validate({"enabled": False}), **ids
        )
        policy = await second.set_workspace_settings(
            user=world.owner, request=WorkspaceMcpServerSettingsUpdate.model_validate({"everyone_tools": []}), **ids
        )

    assert policy.in_workspace is not None
    assert (policy.in_workspace.enabled, policy.in_workspace.everyone_tools) == (False, []), "Off stays off"


async def test_a_change_names_only_what_it_sets_and_null_restores_the_server_s_own(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    await _override(async_db, world.workspace, jira, everyone_tools=["@read"], attach_by_default=True)
    service = _service(async_db, config)
    ids = {"workspace_id": world.workspace.id, "server_id": jira}

    inherited = await service.set_workspace_settings(
        user=world.owner, request=WorkspaceMcpServerSettingsUpdate.model_validate({"everyone_tools": None}), **ids
    )

    assert inherited.in_workspace is not None
    assert (inherited.in_workspace.everyone_tools, inherited.in_workspace.attach_by_default) == (None, True)
    with pytest.raises(ValidationError, match="enabled cannot be null"):
        WorkspaceMcpServerSettingsUpdate.model_validate({"enabled": None})


async def test_an_empty_organization_limit_is_the_organization_s_refusal(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    await _override(async_db, world.workspace, jira, organization_tools=[])
    key = await _member_key(async_db, world)
    service = _service(async_db, config)

    check = await service.check_access(
        user=world.owner, workspace_id=world.workspace.id, server_id=jira, api_key_id=key.id, tool="list_issues"
    )

    assert (check.allowed, check.decided_by) == (False, "organization")


async def test_the_caller_s_access_is_read_as_the_workspace_reaches_the_server(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    await _shared(async_db, world.workspace.organization_id, name="sentry")
    await _override(async_db, world.workspace, jira, organization_tools=["list_issues"])
    service = _service(async_db, config)
    ids = {"workspace_id": world.workspace.id, "server_id": jira}

    listed = await service.list_servers(user=world.member, workspace_id=world.workspace.id)
    one = await service.get_server(user=world.member, **ids)

    access = {server.name: server.caller_access for server in listed.data}
    assert (access["jira"], one.caller_access) == ("some", "some"), "every tool, under a limit of one"
    assert access["sentry"] == "all"

    await service.set_workspace_settings(
        user=world.owner, request=WorkspaceMcpServerSettingsUpdate(enabled=False), **ids
    )
    off = await service.get_server(user=world.owner, **ids)
    assert off.caller_access == "none", "the workspace turned it off"


async def test_the_caller_s_access_is_none_where_the_two_ceilings_do_not_meet(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id, allowed_tools=["list_issues"])
    await _override(async_db, world.workspace, jira, organization_tools=["delete_issue"])
    service = _service(async_db, config)

    one = await service.get_server(user=world.member, workspace_id=world.workspace.id, server_id=jira)

    assert one.caller_access == "none"


async def test_another_organization_s_workspace_reaches_nothing_of_it(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    foreign = await _foreign_workspace(async_db)
    service = _service(async_db, config)

    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service._reached(foreign.id, jira, all_states=True)


async def test_a_member_signs_in_to_an_organization_server_from_their_workspace(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id, encrypted_token=None, credential_mode="per_user")
    service = _service(async_db, config)
    ids = {"workspace_id": world.workspace.id, "server_id": jira}

    started = await service.start_connection(user=world.member, return_to=None, **ids)
    await service.complete_connection(user=world.member, state=_state(started.authorization_url), code="code")
    mine = await service.my_connections(user=world.member, workspace_id=world.workspace.id)
    await service.disconnect(user=world.member, **ids)

    assert [connection.server_id for connection in mine.data] == [jira]
    remaining = (await async_db.execute(select(McpUserConnection.id))).scalars().all()
    assert remaining == []


async def test_a_workspace_s_own_page_reads_an_organization_server_it_turned_off_only_for_its_admins(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    await _override(async_db, world.workspace, jira, enabled=False)
    service = _service(async_db, config)

    read = await service.get_server(user=world.owner, workspace_id=world.workspace.id, server_id=jira)
    assert read.scope == "organization"
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.get_server(user=world.member, workspace_id=world.workspace.id, server_id=jira)


@pytest.mark.usefixtures("attached_index_on_async_db")
async def test_a_workspace_attaching_an_organization_server_reaches_the_attached_server_index(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    AttachedMcpServerIndex.reset()
    world = await _world(async_db, config)
    jira = await _shared(async_db, world.workspace.organization_id)
    sales = await _another_workspace(async_db, world)
    service = _service(async_db, config)
    servers = WorkspaceMcpServers(async_db)
    organization_id = world.workspace.organization_id
    await AttachedMcpServerIndex.refresh_after_write()
    assert not await servers.any_attached(sales.id, organization_id)

    await service.set_workspace_settings(
        user=world.owner,
        workspace_id=sales.id,
        server_id=jira,
        request=WorkspaceMcpServerSettingsUpdate(attach_by_default=True),
    )
    assert await servers.any_attached(sales.id, organization_id), "attached there by its own setting"
    assert not await servers.any_attached(world.workspace.id, organization_id), "and nowhere else"

    await service.clear_workspace_settings(user=world.owner, workspace_id=sales.id, server_id=jira)
    assert not await servers.any_attached(sales.id, organization_id), "back to the organization's"
