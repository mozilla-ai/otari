"""Folding workspaces' own copies of a server into their organization's, and promoting one in place.

Exercised at the service layer, with sign-ins made through the scaffolding's OAuth wire.
"""

import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.exceptions.tools_exceptions import (
    McpServerFoldError,
    McpServerIsOrganizationsError,
    McpServerNameTakenError,
    McpSignInRequiredError,
    WorkspaceMcpServerLimitReachedError,
    WorkspaceMcpServerNotFoundError,
)
from gateway.models.tenancy import User, Workspace
from gateway.models.tools import (
    McpCredentialMode,
    McpServerAlias,
    McpUserConnection,
    WorkspaceMcpServer,
    WorkspaceMcpServerGrant,
    WorkspaceMcpServerOverride,
)
from gateway.repositories.tenancy import WorkspaceMemberRepository, WorkspaceRepository
from gateway.services.model_access import McpAccess
from gateway.services.secret_box import encrypt_secret
from gateway.services.tenancy.workspace_mcp_server_service import (
    MAX_MCP_SERVERS_PER_WORKSPACE,
    McpServerPromote,
    WorkspaceMcpServerCreate,
    WorkspaceMcpServers,
    WorkspaceMcpServerUpdate,
    resolve_workspace_mcp_servers,
)
from gateway.services.tools import AttachedMcpServerIndex, McpServerFoldCandidate, McpUserTokens

from . import test_mcp_user_connections as base
from .test_mcp_user_connections import PUBLIC_URL, _expire, _service, _sign_in, _World, _world

# The scaffolding's fixtures, taken by assignment: pytest finds a fixture by its name in this module.
wire = base.wire
config = base.config
checks = base.checks
tokens = base.tokens
_secret_key = base._secret_key
attached_index_on_async_db = base.attached_index_on_async_db

pytestmark = pytest.mark.asyncio


async def _sales(
    db: AsyncSession, config: GatewayConfig, world: _World, *members: User, url: str = f"{PUBLIC_URL}/"
) -> tuple[Workspace, uuid.UUID]:
    """A second workspace of the organization, with its own copy of the world's server at ``url``."""
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Sales", organization_id=world.workspace.organization_id, created_by_user_id=world.owner.id
    )
    for member in members:
        await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=member.id, role="member")
    await db.commit()
    copy = await _service(db, config).create_server(
        user=world.owner,
        workspace_id=workspace.id,
        request=WorkspaceMcpServerCreate(name="drive-sales", url=url, credential_mode=McpCredentialMode.PER_USER),
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

    weighed = await _candidates(async_db, config, world, credential_mode=McpCredentialMode.PER_USER)

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
            name="drive", url=PUBLIC_URL, credential_mode=McpCredentialMode.PER_USER, fold_server_ids=[copy]
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


async def _promotable(
    db: AsyncSession, config: GatewayConfig, world: _World, *, url: str
) -> tuple[Workspace, uuid.UUID]:
    """Sales's copy at ``url``, the owner signed in to it, and both signed in to the home copy through its client."""
    sales, sales_copy = await _sales(db, config, world, world.member, world.owner, url=url)
    await _sign_in(db, config, world, world.member)  # on the home copy, through its own client
    await _sign_in(db, config, world, world.owner)
    sales_world = _World(sales, world.owner, world.member, world.server.model_copy(update={"id": sales_copy}))
    await _sign_in(db, config, sales_world, world.owner)  # the owner holds one on both
    promoted = await db.get(WorkspaceMcpServer, sales_copy)
    assert promoted is not None and promoted.oauth_client is not None
    # The home copy's sign-ins were made through the same client, as the gateway's metadata document makes them.
    await db.execute(
        update(McpUserConnection)
        .where(McpUserConnection.server_id == world.server.id)
        .values(client_id=promoted.oauth_client["client_id"])
    )
    await db.commit()
    return sales, sales_copy


async def test_promoting_keeps_its_sign_ins_and_carries_a_copy_s_made_through_the_same_client(
    async_db: AsyncSession, config: GatewayConfig, tokens: McpUserTokens
) -> None:
    world = await _world(async_db, config)
    sales, sales_copy = await _promotable(async_db, config, world, url=PUBLIC_URL)

    weighed = await _candidates(async_db, config, world, server_id=sales_copy)
    home = weighed[world.server.id]
    # The owner already holds a sign-in on the promoted server, and goes on through it.
    assert (home.signed_in, home.keep_sign_in, home.sign_in_again, home.switch_account) == (2, 1, 0, 1)

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
    # Carried, it still serves the member, and renews through the promoted server's client once it expires.
    await _expire(async_db, sales_copy)
    promoted = await async_db.get(WorkspaceMcpServer, sales_copy, populate_existing=True)
    assert promoted is not None
    assert (await tokens.token_for(promoted, str(world.member.id))).startswith("renewed-")


async def test_a_sign_in_made_at_another_spelling_of_the_url_is_not_carried(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    """The token reader sends a token only to the URL its sign-in was made for, so a trailing slash ends it."""
    world = await _world(async_db, config)
    sales, sales_copy = await _promotable(async_db, config, world, url=f"{PUBLIC_URL}/")

    weighed = await _candidates(async_db, config, world, server_id=sales_copy)
    home = weighed[world.server.id]
    assert (home.signed_in, home.keep_sign_in, home.sign_in_again, home.switch_account) == (2, 0, 1, 1)

    await _service(async_db, config).promote(
        user=world.owner,
        workspace_id=sales.id,
        server_id=sales_copy,
        request=McpServerPromote(fold_server_ids=[world.server.id]),
    )

    held = (await async_db.execute(select(McpUserConnection.server_id, McpUserConnection.user_id))).all()
    assert [(row.server_id, row.user_id) for row in held] == [(sales_copy, str(world.owner.id))]


@pytest.mark.parametrize(
    ("mode", "secret", "carried"),
    [
        # A shared server never sends a person's own token.
        (McpCredentialMode.SHARED, None, 0),
        # A new server's client is pinned to no token endpoint yet, so its secret goes nowhere a renewal needs it.
        (McpCredentialMode.PER_USER, "my-secret", 0),
        (McpCredentialMode.PER_USER, None, 1),
    ],
)
async def test_a_sign_in_is_carried_only_where_the_new_server_can_renew_it(
    async_db: AsyncSession, config: GatewayConfig, mode: McpCredentialMode, secret: str | None, carried: int
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    client_id = (await async_db.execute(select(McpUserConnection.client_id))).scalar_one()

    created = await _service(async_db, config).create_server(
        user=world.owner,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(
            name="drive",
            url=PUBLIC_URL,
            credential_mode=mode,
            oauth_client_id=client_id,
            oauth_client_secret=secret,
            fold_server_ids=[world.server.id],
        ),
    )

    held = (await async_db.execute(select(McpUserConnection.server_id))).scalars().all()
    assert held == [created.id] * carried


async def test_a_sign_in_without_a_refresh_token_is_not_carried(async_db: AsyncSession, config: GatewayConfig) -> None:
    """It could not be renewed on the new server, and would end at its first expiry."""
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    client_id = (await async_db.execute(select(McpUserConnection.client_id))).scalar_one()
    await async_db.execute(update(McpUserConnection).values(encrypted_refresh_token=None))
    await async_db.commit()

    await _service(async_db, config).create_server(
        user=world.owner,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(
            name="drive",
            url=PUBLIC_URL,
            credential_mode=McpCredentialMode.PER_USER,
            oauth_client_id=client_id,
            fold_server_ids=[world.server.id],
        ),
    )

    assert (await async_db.execute(select(McpUserConnection.id))).scalars().all() == []


async def test_a_fold_revokes_the_token_a_renewal_stored_after_this_session_read_the_sign_in(
    async_db: AsyncSession, config: GatewayConfig, wire: base._Wire
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    # Held for the whole test, as the fold's own session would hold it.
    stale = (await async_db.execute(select(McpUserConnection))).scalar_one()
    await async_db.execute(
        update(McpUserConnection)
        .values(encrypted_refresh_token=encrypt_secret("renewed-refresh"))
        .execution_options(synchronize_session=False)
    )
    await async_db.commit()
    assert stale.encrypted_refresh_token is not None

    await _service(async_db, config).create_server(
        user=world.owner,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(name="drive", url=PUBLIC_URL, fold_server_ids=[world.server.id]),
    )

    await base._drain_revocations()
    assert wire.revoked == ["renewed-refresh"]


async def test_a_copy_s_settings_are_kept_even_where_the_new_server_s_match_them_today(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    """The organization's server can change later; the workspace keeps what its copy had. A name that reads as a
    pattern is no tool's name, and as a limit would reach other tools, so it goes."""
    world = await _world(async_db, config)
    await async_db.execute(
        update(WorkspaceMcpServer)
        .where(WorkspaceMcpServer.id == world.server.id)
        .values(allowed_tools=["list_files", "list*"])
    )
    await async_db.commit()

    created = await _service(async_db, config).create_server(
        user=world.owner,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(
            name="drive",
            url=PUBLIC_URL,
            credential_mode=McpCredentialMode.PER_USER,
            allowed_tools=["list_files"],
            fold_server_ids=[world.server.id],
        ),
    )

    stored = (
        await async_db.execute(
            select(WorkspaceMcpServerOverride.attach_by_default, WorkspaceMcpServerOverride.organization_tools).where(
                WorkspaceMcpServerOverride.server_id == created.id
            )
        )
    ).one()
    assert (stored.attach_by_default, stored.organization_tools) == (False, ["list_files"])


async def test_a_copy_s_allowed_tools_stay_the_most_its_workspace_reaches(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config, allowed_tools=["list_files"])
    sales, sales_copy = await _sales(async_db, config, world)

    created = await _service(async_db, config).create_server(
        user=world.owner,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(
            name="drive",
            url=PUBLIC_URL,
            credential_mode=McpCredentialMode.PER_USER,
            fold_server_ids=[world.server.id, sales_copy],
        ),
    )

    overrides = (
        await async_db.execute(
            select(
                WorkspaceMcpServerOverride.workspace_id,
                WorkspaceMcpServerOverride.organization_tools,
                WorkspaceMcpServerOverride.everyone_tools,
            ).where(WorkspaceMcpServerOverride.server_id == created.id)
        )
    ).all()
    # Each copy's own policy is stored, even where it equals the new server's default today.
    assert sorted((row.workspace_id, row.organization_tools, row.everyone_tools) for row in overrides) == sorted(
        [(world.workspace.id, ["list_files"], ["*"]), (sales.id, None, ["*"])]
    )


async def test_an_organization_server_that_is_off_takes_the_place_only_of_copies_that_are_off(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    service = _service(async_db, config)
    off = WorkspaceMcpServerCreate(name="drive", url=PUBLIC_URL, enabled=False, fold_server_ids=[world.server.id])

    with pytest.raises(McpServerFoldError):
        await service.create_server(user=world.owner, workspace_id=None, request=off)
    assert await async_db.get(WorkspaceMcpServer, world.server.id) is not None

    await service.update_server(
        user=world.owner,
        workspace_id=world.workspace.id,
        server_id=world.server.id,
        request=WorkspaceMcpServerUpdate(enabled=False),
    )
    created = await service.create_server(user=world.owner, workspace_id=None, request=off)
    assert not created.enabled


async def test_one_workspace_folds_one_copy_and_a_promotion_leaves_its_own_workspace_alone(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    service = _service(async_db, config)
    second = await service.create_server(
        user=world.owner,
        workspace_id=world.workspace.id,
        request=WorkspaceMcpServerCreate(
            name="drive-2", url=f"{PUBLIC_URL}/", credential_mode=McpCredentialMode.PER_USER
        ),
    )

    with pytest.raises(McpServerFoldError):
        await service.create_server(
            user=world.owner,
            workspace_id=None,
            request=WorkspaceMcpServerCreate(
                name="drive-org", url=PUBLIC_URL, fold_server_ids=[world.server.id, second.id]
            ),
        )

    assert second.id not in await _candidates(async_db, config, world, server_id=world.server.id)
    with pytest.raises(McpServerFoldError):
        await service.promote(
            user=world.owner,
            workspace_id=world.workspace.id,
            server_id=world.server.id,
            request=McpServerPromote(fold_server_ids=[second.id]),
        )


async def test_a_copy_whose_url_moved_since_it_was_read_is_not_folded(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    """The copies are matched as they are once locked, not as this session last read them."""
    world = await _world(async_db, config)
    # Held for the whole test: the session keeps a clean row only while something references it.
    stale = await async_db.get(WorkspaceMcpServer, world.server.id)
    assert stale is not None
    await async_db.execute(
        update(WorkspaceMcpServer)
        .where(WorkspaceMcpServer.id == world.server.id)
        .values(url=f"{PUBLIC_URL}?project=blue")
        .execution_options(synchronize_session=False)
    )
    assert stale.url == PUBLIC_URL, "this session still holds the URL as it read it"

    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await _service(async_db, config).create_server(
            user=world.owner,
            workspace_id=None,
            request=WorkspaceMcpServerCreate(name="drive-org", url=PUBLIC_URL, fold_server_ids=[world.server.id]),
        )


async def test_a_promotion_counts_toward_the_organization_s_servers(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    organization_id = world.workspace.organization_id
    async_db.add_all(
        WorkspaceMcpServer(organization_id=organization_id, workspace_id=None, name=f"org-{n}", url=f"{PUBLIC_URL}/{n}")
        for n in range(MAX_MCP_SERVERS_PER_WORKSPACE)
    )
    await async_db.commit()

    with pytest.raises(WorkspaceMcpServerLimitReachedError):
        await _service(async_db, config).promote(
            user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, request=McpServerPromote()
        )


async def test_a_folded_copy_s_id_opens_the_organization_s_server_in_its_workspace_only(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    sales, _ = await _sales(async_db, config, world, world.member)
    service = _service(async_db, config)
    created = await service.create_server(
        user=world.owner,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(
            name="drive", url=PUBLIC_URL, credential_mode=McpCredentialMode.PER_USER, fold_server_ids=[world.server.id]
        ),
    )

    opened = await service.get_server(user=world.member, workspace_id=world.workspace.id, server_id=world.server.id)
    assert opened.id == created.id
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.get_server(user=world.member, workspace_id=sales.id, server_id=world.server.id)


async def test_a_request_naming_a_folded_copy_that_was_off_still_skips_it(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    """As it skipped the copy before the fold. The organization's server, off for the workspace, stays unknown by
    its own id there."""
    world = await _world(async_db, config, enabled=False)
    member = str(world.member.id)
    created = await _service(async_db, config).create_server(
        user=world.owner,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(
            name="drive", url=PUBLIC_URL, credential_mode=McpCredentialMode.PER_USER, fold_server_ids=[world.server.id]
        ),
    )

    skipped = await resolve_workspace_mcp_servers(
        async_db, workspace_id=world.workspace.id, server_ids=[world.server.id], access=McpAccess(user_id=member)
    )
    assert skipped == []
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await resolve_workspace_mcp_servers(
            async_db, workspace_id=world.workspace.id, server_ids=[created.id], access=McpAccess(user_id=member)
        )


@pytest.mark.parametrize("removal", ["shared_account", "connection"])
async def test_a_workspace_manager_s_removal_that_finds_the_server_promoted_once_locked_is_refused(
    async_db: AsyncSession, config: GatewayConfig, removal: str
) -> None:
    """The sign-ins of the organization's server serve every workspace now, which a workspace's manager does not run."""
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    # Read now: the refusal rolls the session back, which expires what it had loaded.
    member = str(world.member.id)
    service = _service(async_db, config)
    read_before = await async_db.get(WorkspaceMcpServer, world.server.id)
    await async_db.execute(
        update(WorkspaceMcpServer)
        .where(WorkspaceMcpServer.id == world.server.id)
        .values(workspace_id=None)
        .execution_options(synchronize_session=False)
    )
    await async_db.commit()
    service._server_in = AsyncMock(return_value=read_before)  # type: ignore[method-assign]

    ids = {"user": world.owner, "workspace_id": world.workspace.id, "server_id": world.server.id}
    with pytest.raises(McpServerIsOrganizationsError):
        if removal == "connection":
            await service.remove_connection(**ids, user_id=member)  # type: ignore[arg-type]
        else:
            await service.disconnect_shared(**ids)  # type: ignore[arg-type]
    held = (await async_db.execute(select(McpUserConnection.user_id))).scalars().all()
    assert held == [member]


@pytest.mark.parametrize("write", ["update", "delete"])
async def test_a_workspace_write_that_finds_the_server_promoted_once_locked_is_refused(
    async_db: AsyncSession, config: GatewayConfig, write: str
) -> None:
    """The write read the server as its workspace's own; a promotion committed before it locked the row."""
    world = await _world(async_db, config)
    service = _service(async_db, config)
    read_before = await async_db.get(WorkspaceMcpServer, world.server.id)
    await async_db.execute(
        update(WorkspaceMcpServer)
        .where(WorkspaceMcpServer.id == world.server.id)
        .values(workspace_id=None)
        .execution_options(synchronize_session=False)
    )
    await async_db.commit()
    service._server_in = AsyncMock(return_value=read_before)  # type: ignore[method-assign]

    ids = {"user": world.owner, "workspace_id": world.workspace.id, "server_id": world.server.id}
    with pytest.raises(McpServerIsOrganizationsError):
        if write == "update":
            await service.update_server(**ids, request=WorkspaceMcpServerUpdate(enabled=False))  # type: ignore[arg-type]
        else:
            await service.delete_server(**ids)  # type: ignore[arg-type]
    row = await async_db.get(WorkspaceMcpServer, world.server.id, populate_existing=True)
    assert row is not None and row.enabled


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


@pytest.mark.usefixtures("attached_index_on_async_db")
async def test_promoting_an_attached_server_attaches_it_in_every_workspace_at_once(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    AttachedMcpServerIndex.reset()
    world = await _world(async_db, config)
    sales, sales_copy = await _sales(async_db, config, world, url="https://93.184.216.99/other")
    service = _service(async_db, config)
    await service.update_server(
        user=world.owner,
        workspace_id=sales.id,
        server_id=sales_copy,
        request=WorkspaceMcpServerUpdate(attach_by_default=True),
    )
    servers = WorkspaceMcpServers(async_db)
    organization_id = world.workspace.organization_id
    assert not await servers.any_attached(world.workspace.id, organization_id)

    await service.promote(user=world.owner, workspace_id=sales.id, server_id=sales_copy, request=McpServerPromote())

    assert await servers.any_attached(world.workspace.id, organization_id), "the organization's now"


@pytest.mark.usefixtures("attached_index_on_async_db")
async def test_a_folded_copy_attached_to_every_request_stays_attached_in_its_workspace(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    AttachedMcpServerIndex.reset()
    world = await _world(async_db, config)
    service = _service(async_db, config)
    await service.update_server(
        user=world.owner,
        workspace_id=world.workspace.id,
        server_id=world.server.id,
        request=WorkspaceMcpServerUpdate(attach_by_default=True),
    )
    servers = WorkspaceMcpServers(async_db)
    organization_id = world.workspace.organization_id

    await service.create_server(
        user=world.owner,
        workspace_id=None,
        request=WorkspaceMcpServerCreate(
            name="drive",
            url=PUBLIC_URL,
            credential_mode=McpCredentialMode.PER_USER,
            fold_server_ids=[world.server.id],
        ),
    )

    assert await servers.any_attached(world.workspace.id, organization_id), "its workspace's own setting, now"
    stored = (await async_db.execute(select(WorkspaceMcpServerOverride.attach_by_default))).scalars().all()
    assert stored == [True], "the copy's attach, kept as its workspace's setting"
