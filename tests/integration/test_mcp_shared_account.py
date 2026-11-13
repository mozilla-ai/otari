"""A shared MCP server's account: the sign-in a manager makes for everyone, and what a request then carries.

At the service layer, on the scaffolding of `test_mcp_user_connections.py`, whose stand-in for the
authorization server these cases share.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.exceptions.tools_exceptions import (
    McpConnectionFlowError,
    McpConnectionModeError,
    McpConnectionNotFoundError,
    McpConnectionSetupError,
    McpConnectionUnavailableError,
    WorkspaceMcpServerCheckOriginError,
    WorkspaceMcpServerUnsafeUrlError,
)
from gateway.models.mcp import ResolvedMcpServer
from gateway.models.tenancy import OrganizationMember, WorkspaceMember
from gateway.models.tools import McpCredentialMode, McpOAuthFlow, McpSharedConnection, McpUserConnection
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import CheckFailure, ServerCheck
from gateway.services.model_access import McpAccess
from gateway.services.secret_box import decrypt_secret
from gateway.services.tenancy.workspace_mcp_server_service import (
    WorkspaceMcpServerCheckRequest,
    WorkspaceMcpServerPublic,
    WorkspaceMcpServers,
    WorkspaceMcpServerUpdate,
)
from gateway.services.tools import McpGrantMembershipListener, McpUserTokens
from gateway.services.tools._mcp_oauth import McpOAuthTokenError, McpOAuthTokens

from . import test_mcp_user_connections as base
from .test_mcp_user_connections import _row, _service, _sign_in, _state, _Wire, _World, _world

# The scaffolding's fixtures, taken by assignment: pytest finds a fixture by its name in this module.
wire = base.wire
config = base.config
tokens = base.tokens
_secret_key = base._secret_key

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def asks_for_an_account(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """A server that answers 401 until it is sent a credential, as most catalog servers do."""
    sent: list[str | None] = []

    async def check(server: ResolvedMcpServer) -> ServerCheck:
        sent.append(server.authorization_token)
        if server.authorization_token is None:
            return ServerCheck(tool_names=[], failure=CheckFailure.UNAUTHORIZED, http_status=401)
        return ServerCheck(tool_names=["list_issues"])

    monkeypatch.setattr(mcp_stateless, "check_stored_server", check)
    return sent


async def _shared_world(db: AsyncSession, config: GatewayConfig, **server: object) -> _World:
    return await _world(db, config, name="github", credential_mode="shared", **server)


async def _sign_in_for_everyone(db: AsyncSession, config: GatewayConfig, world: _World) -> WorkspaceMcpServerPublic:
    service = _service(db, config)
    ids = {"workspace_id": world.workspace.id, "server_id": world.server.id}
    started = await service.start_shared_sign_in(user=world.owner, return_to=None, **ids)
    done = await service.complete_connection(user=world.owner, state=_state(started.authorization_url), code="code")
    assert done.shared is True and done.checked is False
    return await service.get_server(user=world.owner, **ids)


async def _account(db: AsyncSession, world: _World) -> McpSharedConnection | None:
    return (
        await db.execute(
            select(McpSharedConnection)
            .where(McpSharedConnection.server_id == world.server.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


async def test_a_shared_server_that_asks_for_an_account_is_not_set_up_and_requests_skip_it(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _shared_world(async_db, config, attach_by_default=True)
    servers = WorkspaceMcpServers(async_db)
    member = McpAccess(user_id=str(world.member.id))

    assert world.server.needs_setup is True
    assert (world.server.has_token, world.server.shared_account) == (False, None)
    # Skipped as a disabled server is: named or attached, the request goes on without it.
    assert await servers.resolve_many(world.workspace.id, [world.server.id], member) == []
    assert (await servers.resolve_for_request(world.workspace.id, [], member, attach=True)).servers == []
    assert await servers.resolve_one(world.workspace.id, world.server.id, member) is None
    assert await servers.list_reachable(world.workspace.id, member) == []


async def test_a_shared_server_that_takes_no_credential_is_never_held_back(
    async_db: AsyncSession, config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def open_server(server: ResolvedMcpServer) -> ServerCheck:
        return ServerCheck(tool_names=["search_docs"])

    monkeypatch.setattr(mcp_stateless, "check_stored_server", open_server)
    world = await _shared_world(async_db, config)
    servers = WorkspaceMcpServers(async_db)

    assert world.server.needs_setup is False
    [resolved] = await servers.resolve_many(world.workspace.id, [world.server.id], McpAccess(user_id="anyone"))
    assert resolved.authorization_token is None


async def test_a_manager_signs_in_for_everyone_and_every_caller_then_carries_that_account(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, asks_for_an_account: list[str | None]
) -> None:
    world = await _shared_world(async_db, config)

    public = await _sign_in_for_everyone(async_db, config, world)

    assert public.needs_setup is False
    assert public.shared_account is not None
    assert public.shared_account.signed_in_by == "Owner"
    assert public.shared_account.renewable is True
    # The sign-in itself listed the server once, as a check does.
    assert asks_for_an_account == [None, "access-1"]
    assert public.health is not None and public.health.status == "healthy"
    account = await _account(async_db, world)
    assert account is not None and account.connected_by == world.owner.id
    assert account.encrypted_access_token != "access-1", "only ciphertext is stored"
    assert decrypt_secret(account.encrypted_access_token) == "access-1"
    # Nobody's own connection was made, and the sign-in is spent.
    assert (await async_db.execute(select(McpUserConnection))).first() is None
    assert (await async_db.execute(select(McpOAuthFlow))).first() is None

    servers = WorkspaceMcpServers(async_db)
    for access in (McpAccess(user_id=None), McpAccess(user_id=str(world.member.id)), McpAccess(unrestricted=True)):
        [resolved] = await servers.resolve_many(world.workspace.id, [world.server.id], access)
        assert resolved.authorization_token == "access-1"
        assert resolved.renew_token is not None
        [listed] = await servers.list_reachable(world.workspace.id, access)
        assert listed.sign_in_required is False
    one = await servers.resolve_one(world.workspace.id, world.server.id, McpAccess(user_id="anyone"))
    assert one is not None and one.authorization_token == "access-1"
    # A member sees who the workspace acts as, and never a token.
    seen = await _service(async_db, config).get_server(
        user=world.member, workspace_id=world.workspace.id, server_id=world.server.id
    )
    assert seen.shared_account is not None and seen.shared_account.signed_in_by == "Owner"


async def test_the_shared_account_is_only_tested_against_the_servers_own_origin(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, asks_for_an_account: list[str | None]
) -> None:
    """A test of unsaved details cannot carry the shared account's token to an address a manager types."""
    world = await _shared_world(async_db, config)
    await _sign_in_for_everyone(async_db, config, world)
    service = _service(async_db, config)
    sent = len(asks_for_an_account)

    with pytest.raises(WorkspaceMcpServerCheckOriginError):
        await service.check_connection(
            user=world.owner,
            workspace_id=world.workspace.id,
            request=WorkspaceMcpServerCheckRequest(url=base.OTHER_PUBLIC_URL, server_id=world.server.id),
        )
    assert len(asks_for_an_account) == sent, "nothing was sent"

    await service.check_connection(
        user=world.owner,
        workspace_id=world.workspace.id,
        request=WorkspaceMcpServerCheckRequest(url=f"{base.PUBLIC_URL}?tab=1", server_id=world.server.id),
    )
    assert asks_for_an_account[sent:] == ["access-1"]


async def test_only_a_manager_starts_or_ends_the_shared_account_and_only_on_a_shared_server(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _shared_world(async_db, config)
    service = _service(async_db, config)
    ids = {"workspace_id": world.workspace.id, "server_id": world.server.id}

    async def reloaded() -> None:
        # A refusal rolls the session back, which expires the identities these calls read.
        await async_db.refresh(world.member)
        await async_db.refresh(world.owner)

    with pytest.raises(NotAuthorizedError):
        await service.start_shared_sign_in(user=world.member, return_to=None, **ids)
    await reloaded()
    with pytest.raises(NotAuthorizedError):
        await service.disconnect_shared(user=world.member, **ids)
    await reloaded()
    # A member's own sign-in has nothing to do with a shared server.
    with pytest.raises(McpConnectionModeError, match="one shared account"):
        await service.start_connection(user=world.member, return_to=None, **ids)
    await reloaded()
    with pytest.raises(McpConnectionNotFoundError):
        await service.disconnect_shared(user=world.owner, **ids)
    await reloaded()

    await service.update_server(
        user=world.owner, request=WorkspaceMcpServerUpdate(credential_mode=McpCredentialMode.PER_USER), **ids
    )
    with pytest.raises(McpConnectionModeError, match="each member's own account"):
        await service.start_shared_sign_in(user=world.owner, return_to=None, **ids)


async def test_a_shared_sign_in_needs_https_since_its_token_travels_to_the_server(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _shared_world(async_db, config, url="http://93.184.216.34/mcp")

    with pytest.raises(WorkspaceMcpServerUnsafeUrlError):
        await _service(async_db, config).start_shared_sign_in(
            user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
        )


async def test_a_manager_who_lost_the_workspace_while_signing_in_stores_no_account_and_the_tokens_are_revoked(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire
) -> None:
    world = await _shared_world(async_db, config)
    service = _service(async_db, config)
    started = await service.start_shared_sign_in(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )
    await async_db.execute(
        update(WorkspaceMember).where(WorkspaceMember.user_id == world.owner.id).values(role="member")  # type: ignore[arg-type]
    )
    await async_db.execute(
        update(OrganizationMember).where(OrganizationMember.user_id == world.owner.id).values(role="member")  # type: ignore[arg-type]
    )
    await async_db.commit()

    with pytest.raises(McpConnectionFlowError, match="no longer manage"):
        await service.complete_connection(user=world.owner, state=_state(started.authorization_url), code="code")

    assert await _account(async_db, world) is None
    await asyncio.sleep(0.05)
    assert wire.revoked == ["refresh-1"], "tokens nothing will hold are ended at the authorization server"


async def test_another_persons_shared_sign_in_cannot_be_finished(async_db: AsyncSession, config: GatewayConfig) -> None:
    world = await _shared_world(async_db, config)
    service = _service(async_db, config)
    started = await service.start_shared_sign_in(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )

    with pytest.raises(McpConnectionFlowError, match="another person"):
        await service.complete_connection(user=world.member, state=_state(started.authorization_url), code="code")
    assert await _account(async_db, world) is None


async def test_signing_in_for_everyone_takes_the_place_of_a_stored_token_and_a_token_takes_its_place_back(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire
) -> None:
    world = await _shared_world(async_db, config, authorization_token="pasted-token")
    service = _service(async_db, config)
    ids = {"workspace_id": world.workspace.id, "server_id": world.server.id}
    servers = WorkspaceMcpServers(async_db)
    anyone = McpAccess(user_id="anyone")

    public = await _sign_in_for_everyone(async_db, config, world)
    assert (public.has_token, public.shared_account is not None) == (False, True)
    assert (await _row(async_db, world.server.id)).encrypted_token is None

    # A second sign-in replaces the account: one per server.
    wire.issue = McpOAuthTokens(access_token="access-2", refresh_token="refresh-2", expires_in=3600)
    await _sign_in_for_everyone(async_db, config, world)
    [resolved] = await servers.resolve_many(world.workspace.id, [world.server.id], anyone)
    assert resolved.authorization_token == "access-2"

    updated = await service.update_server(
        user=world.owner, request=WorkspaceMcpServerUpdate(authorization_token="new-token"), **ids
    )
    assert (updated.has_token, updated.shared_account) == (True, None)
    assert await _account(async_db, world) is None
    [resolved] = await servers.resolve_many(world.workspace.id, [world.server.id], anyone)
    assert (resolved.authorization_token, resolved.renew_token) == ("new-token", None)


@pytest.mark.parametrize("change", [{"credential_mode": "per_user"}, {"url": "https://93.184.216.35/mcp"}])
async def test_a_change_the_account_no_longer_fits_ends_it(
    async_db: AsyncSession, config: GatewayConfig, change: dict[str, str]
) -> None:
    world = await _shared_world(async_db, config)
    await _sign_in_for_everyone(async_db, config, world)

    updated = await _service(async_db, config).update_server(
        user=world.owner,
        workspace_id=world.workspace.id,
        server_id=world.server.id,
        request=WorkspaceMcpServerUpdate(**change),  # type: ignore[arg-type]
    )

    assert updated.shared_account is None
    assert await _account(async_db, world) is None


async def test_switching_a_personal_server_to_shared_ends_every_connection_and_waits_for_an_account(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)

    updated = await _service(async_db, config).update_server(
        user=world.owner,
        workspace_id=world.workspace.id,
        server_id=world.server.id,
        request=WorkspaceMcpServerUpdate(credential_mode=McpCredentialMode.SHARED),
    )

    assert (await async_db.execute(select(McpUserConnection))).first() is None
    assert (updated.credential_mode, updated.needs_setup, updated.shared_account) == ("shared", True, None)


async def test_disconnecting_the_shared_account_revokes_it_and_leaves_the_server_not_set_up(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire
) -> None:
    world = await _shared_world(async_db, config)
    await _sign_in_for_everyone(async_db, config, world)

    public = await _service(async_db, config).disconnect_shared(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id
    )

    assert (public.shared_account, public.needs_setup) == (None, True)
    assert await _account(async_db, world) is None
    assert wire.revoked == ["refresh-1"]


async def test_the_shared_accounts_token_is_renewed_before_it_expires_and_after_a_refusal(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, tokens: McpUserTokens
) -> None:
    world = await _shared_world(async_db, config)
    await _sign_in_for_everyone(async_db, config, world)
    now = datetime.now(UTC)
    await async_db.execute(
        update(McpSharedConnection).values(expires_at=now - timedelta(minutes=1), updated_at=now - timedelta(hours=1))
    )
    await async_db.commit()
    servers = WorkspaceMcpServers(async_db, tokens)

    [resolved] = await servers.resolve_many(world.workspace.id, [world.server.id], McpAccess(user_id="anyone"))

    assert resolved.authorization_token == "renewed-1"
    assert [refresh for _, refresh in wire.refreshes] == ["refresh-1"]
    account = await _account(async_db, world)
    assert account is not None and decrypt_secret(account.encrypted_refresh_token or "") == "rotated-1"
    # The server refuses the token mid-request: one renewal, whatever its recorded expiry.
    assert resolved.renew_token is not None
    assert await resolved.renew_token("renewed-1") == "renewed-2"


async def test_a_shared_account_whose_grant_ended_is_removed_and_the_server_goes_back_to_waiting(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, tokens: McpUserTokens
) -> None:
    world = await _shared_world(async_db, config)
    await _sign_in_for_everyone(async_db, config, world)
    wire.refresh_error = McpOAuthTokenError("ended", status_code=400, error_code="invalid_grant")
    servers = WorkspaceMcpServers(async_db, tokens)
    [resolved] = await servers.resolve_many(world.workspace.id, [world.server.id], McpAccess(user_id="anyone"))
    assert resolved.renew_token is not None

    with pytest.raises(McpConnectionUnavailableError):
        await resolved.renew_token("access-1")

    assert await _account(async_db, world) is None


async def test_a_shared_server_takes_a_managers_client_for_a_sign_in_that_cannot_register(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire
) -> None:
    """GitHub's shape: no registration, so the manager supplies the OAuth app the sign-in goes through."""
    world = await _shared_world(async_db, config, oauth_client_id="github-app", oauth_client_secret="app-secret")

    await _sign_in_for_everyone(async_db, config, world)

    assert wire.registrations == []
    assert wire.exchanges[0]["client"].client_id == "github-app"  # type: ignore[attr-defined]


async def test_a_shared_sign_in_without_a_public_address_says_so(
    async_db: AsyncSession, config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = await _shared_world(async_db, config)
    monkeypatch.setattr(config, "public_base_url", None)

    with pytest.raises(McpConnectionSetupError, match="public_base_url"):
        await _service(async_db, config).start_shared_sign_in(
            user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
        )


async def test_deleting_the_workspace_revokes_its_shared_account_and_a_member_leaving_does_not(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire
) -> None:
    world = await _shared_world(async_db, config)
    await _sign_in_for_everyone(async_db, config, world)
    uow = UnitOfWork(async_db)

    async with uow:
        await McpGrantMembershipListener(uow).member_lost_workspaces(world.member.id, [world.workspace.id])
    await base._drain_revocations()
    assert wire.revoked == []
    assert await _account(async_db, world) is not None

    async with uow:
        await McpGrantMembershipListener(uow).workspace_deleted(world.workspace.id, [])
    await base._drain_revocations()
    assert wire.revoked == ["refresh-1"]
