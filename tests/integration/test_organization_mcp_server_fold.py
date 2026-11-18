"""Folding workspaces' own copies of a server into their organization's, and promoting one in place.

Exercised at the service layer, with sign-ins made through the scaffolding's OAuth wire.
"""

import uuid

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.exceptions.tools_exceptions import (
    McpServerFoldError,
    McpServerNameTakenError,
    McpSignInRequiredError,
    WorkspaceMcpServerNotFoundError,
)
from gateway.models.tenancy import User, Workspace
from gateway.models.tools import (
    McpServerAlias,
    McpUserConnection,
    WorkspaceMcpServer,
    WorkspaceMcpServerGrant,
    WorkspaceMcpServerOverride,
)
from gateway.repositories.tenancy import WorkspaceMemberRepository, WorkspaceRepository
from gateway.services.model_access import McpAccess
from gateway.services.tenancy.workspace_mcp_server_service import (
    McpServerPromote,
    WorkspaceMcpServerCreate,
    resolve_workspace_mcp_servers,
)
from gateway.services.tools import McpServerFoldCandidate

from . import test_mcp_user_connections as base
from .test_mcp_user_connections import PUBLIC_URL, _service, _sign_in, _World, _world

# The scaffolding's fixtures, taken by assignment: pytest finds a fixture by its name in this module.
wire = base.wire
config = base.config
checks = base.checks
_secret_key = base._secret_key

pytestmark = pytest.mark.asyncio


async def _sales(db: AsyncSession, config: GatewayConfig, world: _World, *members: User) -> tuple[Workspace, uuid.UUID]:
    """A second workspace of the organization, with its own copy of the world's server."""
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Sales", organization_id=world.workspace.organization_id, created_by_user_id=world.owner.id
    )
    for member in members:
        await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=member.id, role="member")
    await db.commit()
    copy = await _service(db, config).create_server(
        user=world.owner,
        workspace_id=workspace.id,
        request=WorkspaceMcpServerCreate(name="drive-sales", url=f"{PUBLIC_URL}/", credential_mode="per_user"),
    )
    return workspace, copy.id


async def _candidates(
    db: AsyncSession, config: GatewayConfig, world: _World, **query: object
) -> dict[uuid.UUID, McpServerFoldCandidate]:
    weighed = await _service(db, config).fold_candidates(user=world.owner, url=PUBLIC_URL, **query)  # type: ignore[arg-type]
    return {candidate.server_id: candidate for candidate in weighed.data}


async def test_a_fold_step_counts_who_must_sign_in_again(async_db: AsyncSession, config: GatewayConfig) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    _, sales_copy = await _sales(async_db, config, world)

    weighed = await _candidates(async_db, config, world, credential_mode="per_user")

    home = weighed[world.server.id]
    assert (home.signed_in, home.keep_sign_in, home.sign_in_again, home.same_credential) == (1, 0, 1, True)
    assert weighed[sales_copy].signed_in == 0
    assert weighed[sales_copy].workspace_name == "Sales"


async def test_folding_a_copy_moves_its_rules_keeps_its_id_and_ends_its_sign_ins(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    copy = world.server.id
    await _sign_in(async_db, config, world, world.member)
    member = str(world.member.id)
    await async_db.execute(
        update(WorkspaceMcpServer)
        .where(WorkspaceMcpServer.id == copy)
        .values(read_tools=["list_files"], everyone_tools=["@read"])
    )
    async_db.add(
        WorkspaceMcpServerGrant(server_id=copy, workspace_id=world.workspace.id, user_id=member, tools=["@read"])
    )
    await async_db.commit()

    created = await _service(async_db, config).create_server(
        user=world.owner,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(
            name="drive", url=PUBLIC_URL, credential_mode="per_user", fold_server_ids=[copy]
        ),
    )

    assert await async_db.get(WorkspaceMcpServer, copy) is None
    alias = (await async_db.execute(select(McpServerAlias.server_id))).scalars().all()
    assert alias == [created.id]
    grants = (await async_db.execute(select(WorkspaceMcpServerGrant.server_id, WorkspaceMcpServerGrant.tools))).all()
    assert [(row.server_id, row.tools) for row in grants] == [(created.id, ["list_files"])], "@read written out"
    overrides = (await async_db.execute(select(WorkspaceMcpServerOverride.everyone_tools))).scalars().all()
    assert overrides == [["list_files"]]
    assert (await async_db.execute(select(McpUserConnection.id))).scalars().all() == [], "its sign-in ended"
    with pytest.raises(McpSignInRequiredError) as needed:
        await resolve_workspace_mcp_servers(
            async_db, workspace_id=world.workspace.id, server_ids=[copy], access=McpAccess(user_id=member)
        )
    assert needed.value.mcp_server_id == created.id, "the old id names the new server, which needs a sign-in"


async def test_promoting_keeps_its_sign_ins_and_carries_a_copy_s_made_through_the_same_client(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    sales, sales_copy = await _sales(async_db, config, world, world.member, world.owner)
    await _sign_in(async_db, config, world, world.member)  # on the home copy, through its own client
    await _sign_in(async_db, config, world, world.owner)
    sales_world = _World(sales, world.owner, world.member, world.server.model_copy(update={"id": sales_copy}))
    await _sign_in(async_db, config, sales_world, world.owner)  # the owner holds one on both
    promoted = await async_db.get(WorkspaceMcpServer, sales_copy)
    assert promoted is not None and promoted.oauth_client is not None
    # The home copy's sign-ins were made through the same client, as the gateway's metadata document makes them.
    await async_db.execute(
        update(McpUserConnection)
        .where(McpUserConnection.server_id == world.server.id)
        .values(client_id=promoted.oauth_client["client_id"])
    )
    await async_db.commit()

    weighed = await _candidates(async_db, config, world, server_id=sales_copy)
    home = weighed[world.server.id]
    assert (home.signed_in, home.keep_sign_in, home.sign_in_again) == (2, 1, 1)

    done = await _service(async_db, config).promote(
        user=world.owner,
        workspace_id=sales.id,
        server_id=sales_copy,
        request=McpServerPromote(fold_server_ids=[world.server.id]),
    )

    assert (done.id, done.scope) == (sales_copy, "organization")
    held = (await async_db.execute(select(McpUserConnection.server_id, McpUserConnection.user_id))).all()
    assert sorted((row.server_id, row.user_id) for row in held) == sorted(
        [(sales_copy, str(world.owner.id)), (sales_copy, str(world.member.id))]
    ), "the owner keeps their sign-in on the promoted server, the member's is carried"


async def test_a_new_organization_server_may_take_a_name_only_the_copies_it_folds_have(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    service = _service(async_db, config)

    with pytest.raises(McpServerNameTakenError):
        await service.create_server(
            user=world.owner, workspace_id=None, request=WorkspaceMcpServerCreate(name="drive", url=PUBLIC_URL)
        )
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.create_server(
            user=world.owner,
            workspace_id=None,
            request=WorkspaceMcpServerCreate(
                name="other", url="https://93.184.216.99/mcp", fold_server_ids=[world.server.id]
            ),
        )
    created = await service.create_server(
        user=world.owner,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(name="drive", url=PUBLIC_URL, fold_server_ids=[world.server.id]),
    )
    assert created.name == "drive"


async def test_a_workspace_cannot_fold_anything_into_its_own_server(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    with pytest.raises(McpServerFoldError):
        await _service(async_db, config).create_server(
            user=world.owner,
            workspace_id=world.workspace.id,
            request=WorkspaceMcpServerCreate(name="other", url=PUBLIC_URL, fold_server_ids=[world.server.id]),
        )
