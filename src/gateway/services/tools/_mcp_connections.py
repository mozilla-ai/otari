"""Each user's own sign-in to a per-user stored MCP server.

A stored server in ``per_user`` mode sends no shared token: a call carries the token of the user
the calling key bills to, which that user got by signing in to the server through the MCP
authorization flow. Two classes share the rows:

* :class:`McpConnectionService` runs the sign-in (start it, finish it) and ends a connection.
  It decides nothing about who may ask: its callers have settled that.
* :class:`McpUserTokens` is what a request reads a user's token through, renewing it when it is
  about to expire.

The wire half is in ``_mcp_oauth``. No token, code, verifier or state is logged or returned.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import uuid
import weakref
from collections.abc import Callable, Collection
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.exceptions.tools_exceptions import (
    McpConnectionFlowError,
    McpConnectionModeError,
    McpConnectionNotFoundError,
    McpConnectionSetupError,
    McpConnectionUnavailableError,
    McpConnectionUpstreamError,
    McpSignInRequiredError,
    WorkspaceMcpServerNotFoundError,
)
from gateway.log_config import logger
from gateway.models.tools import McpCredentialMode, McpUserConnection, WorkspaceMcpServer
from gateway.repositories.tools import (
    McpOAuthFlowRepository,
    McpUserConnectionRepository,
    WorkspaceMcpServerRepository,
)
from gateway.repositories.users_repository import DEFAULT_USER_ID
from gateway.services.secret_box import SecretBoxUnavailableError, decrypt_secret, encrypt_secret, secret_box_configured
from gateway.services.tools import _mcp_oauth
from gateway.services.tools._mcp_oauth import (
    McpOAuthClient,
    McpOAuthDiscoveryError,
    McpOAuthMetadata,
    McpOAuthRegistrationError,
    McpOAuthTokenError,
    McpOAuthTokens,
)

# How long a started sign-in may take to come back.
FLOW_LIFETIME = timedelta(minutes=10)
# A token is renewed this long before its recorded expiry. A request hands the token to a tool loop
# that may run for minutes, so the margin covers the request as well as the clock.
EXPIRY_SKEW = timedelta(minutes=5)
# The gateway's own path the authorization server sends the browser back to, under the API root.
CALLBACK_PATH = "/mcp-connections/callback"
# The dashboard route that finishes a sign-in a dashboard session started.
DASHBOARD_CALLBACK_ROUTE = "/#/mcp-connections/callback"
_MAX_RETURN_TO_LENGTH = 2048


class StoredMcpOAuthClient(BaseModel):
    """The shape of ``WorkspaceMcpServer.oauth_client``: the client a server's users sign in through.

    A manager's own, or one the gateway registered itself (``dynamic``). A registered one is tied to
    the redirect URI it was registered with, and is replaced when that moves.
    """

    client_id: str
    encrypted_secret: str | None = None
    # How a registered client authenticates; a manager's is decided at sign-in from what the server takes.
    auth_method: str | None = None
    dynamic: bool = False
    redirect_uri: str | None = None


def stored_oauth_client(server: WorkspaceMcpServer) -> StoredMcpOAuthClient | None:
    """A server's stored OAuth client, if it has one."""
    return StoredMcpOAuthClient.model_validate(server.oauth_client) if server.oauth_client else None


def checked_return_to(return_to: str | None) -> str | None:
    """A dashboard route to land on after a sign-in, or ``None``.

    Only a path inside this deployment's own dashboard is kept, so a sign-in link cannot be made
    to end on somebody else's site.

    Raises:
        McpConnectionSetupError: the value is not such a path.
    """
    if return_to is None or return_to == "":
        return None
    usable = (
        len(return_to) <= _MAX_RETURN_TO_LENGTH
        and return_to.startswith("/")
        and not return_to.startswith("//")
        and "\\" not in return_to
        and not any(ord(char) < 0x20 or ord(char) == 0x7F for char in return_to)
    )
    if not usable:
        raise McpConnectionSetupError("return_to must be a path inside the dashboard, such as /tools/mcp-servers")
    return return_to


def callback_url(config: GatewayConfig) -> str:
    """Where an authorization server sends the browser back to, derived from ``public_base_url`` and never supplied.

    Raises:
        McpConnectionSetupError: ``public_base_url`` is unset, so the gateway cannot name itself.
    """
    base = (config.public_base_url or "").strip().rstrip("/")
    if not base:
        raise McpConnectionSetupError(
            "Signing in to an MCP server needs public_base_url set, because the address the "
            "authorization server returns to is derived from it"
        )
    return f"{base}{API_ROOT}{CALLBACK_PATH}"


def require_per_user_support(config: GatewayConfig) -> None:
    """Refuse per-user mode on a deployment that could not run a sign-in or keep its tokens.

    Raises:
        McpConnectionSetupError: ``public_base_url`` is unset.
        SecretBoxUnavailableTenancyError: ``OTARI_SECRET_KEY`` is unset.
    """
    callback_url(config)
    if not secret_box_configured():
        raise SecretBoxUnavailableTenancyError("MCP connection tokens")


def _state_hash(state: str) -> str:
    return hashlib.sha256(state.encode()).hexdigest()


def _expiry(tokens: McpOAuthTokens, now: datetime) -> datetime | None:
    return now + timedelta(seconds=tokens.expires_in) if tokens.expires_in is not None else None


def _needs_refresh(expires_at: datetime | None, now: datetime) -> bool:
    return expires_at is not None and expires_at <= now + EXPIRY_SKEW


def _is_expired(expires_at: datetime | None, now: datetime) -> bool:
    return expires_at is not None and expires_at <= now


def _encrypted(value: str) -> str:
    try:
        return encrypt_secret(value)
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError("MCP connection tokens") from None


def _client_for(server: WorkspaceMcpServer, *, client_id: str, auth_method: str) -> McpOAuthClient | None:
    """The client a grant was issued to, as the server still holds it; ``None`` once the server holds another.

    Raises:
        SecretDecryptionError: the stored client secret will not decrypt.
        SecretBoxUnavailableError: ``OTARI_SECRET_KEY`` is not configured.
    """
    stored = stored_oauth_client(server)
    if stored is None or stored.client_id != client_id:
        return None
    secret = decrypt_secret(stored.encrypted_secret) if stored.encrypted_secret else None
    return McpOAuthClient(client_id=client_id, client_secret=secret, auth_method=auth_method)


@dataclass(frozen=True, slots=True)
class FlowLanding:
    """What a callback's state names: nothing, or a pending sign-in and how it finishes."""

    known: bool
    # A dashboard session started it, so that session's dashboard finishes it.
    needs_dashboard: bool = False


@dataclass(frozen=True, slots=True)
class CompletedConnection:
    """A sign-in that finished: which server it connected, and where to land."""

    server_id: uuid.UUID
    workspace_id: uuid.UUID
    server_name: str
    return_to: str | None


@dataclass(frozen=True, slots=True)
class _Pending:
    """A consumed sign-in, copied off its row."""

    server_id: uuid.UUID
    user_id: str
    identity_id: uuid.UUID | None
    encrypted_code_verifier: str
    client_id: str
    client_auth_method: str
    token_endpoint: str
    revocation_endpoint: str | None
    resource: str | None
    redirect_uri: str
    return_to: str | None
    expires_at: datetime


class McpConnectionService:
    """Runs a user's sign-in to a per-user MCP server, and ends a connection."""

    def __init__(
        self,
        uow: UnitOfWork,
        servers: WorkspaceMcpServerRepository,
        connections: McpUserConnectionRepository,
        flows: McpOAuthFlowRepository,
        config: GatewayConfig,
    ) -> None:
        self._uow = uow
        self._servers = servers
        self._connections = connections
        self._flows = flows
        self._config = config

    async def start(
        self,
        *,
        server_id: uuid.UUID,
        user_id: str,
        return_to: str | None = None,
        identity_id: uuid.UUID | None = None,
    ) -> str:
        """Start a user's sign-in to a per-user server, and return the URL to send their browser to.

        ``user_id`` is the request-plane user the connection will belong to, whose row must exist,
        and the caller has already decided that this user may sign in to this server.
        ``return_to`` is a dashboard path to land on afterwards. ``identity_id`` is the dashboard
        identity asking, when one is: the sign-in then finishes only in that identity's session,
        so a link handed to somebody else connects nothing. Without it the sign-in finishes on the
        strength of its state alone, for a link minted outside a dashboard session.

        The state is single-use and expires after ``FLOW_LIFETIME``.

        Raises:
            WorkspaceMcpServerNotFoundError: no such server.
            McpConnectionModeError: the server uses a shared token.
            McpConnectionSetupError: ``public_base_url`` is unset, ``return_to`` is not a dashboard
                path, the user is the shared ``default`` one, or the authorization server registers
                no client and the server holds none.
            McpConnectionUpstreamError: the server's authorization setup could not be discovered.
        """
        if user_id == DEFAULT_USER_ID:
            raise McpConnectionSetupError("The shared default user stands for keys with no owner and cannot sign in")
        redirect_uri = callback_url(self._config)
        return_to = checked_return_to(return_to)
        async with self._uow:
            server = await self._per_user_server(server_id)
            url, stored = server.url, stored_oauth_client(server)
        try:
            metadata = await _mcp_oauth.discover(url)
            client = await self._client(server_id, url, stored, metadata, redirect_uri)
        except McpOAuthRegistrationError:
            raise McpConnectionSetupError(
                "This server's authorization server does not register clients on its own. "
                "Add an OAuth client ID to the server."
            ) from None
        except McpOAuthDiscoveryError as exc:
            raise McpConnectionUpstreamError(f"Could not start the sign-in: {exc}") from None

        state, verifier = secrets.token_urlsafe(32), _mcp_oauth.new_code_verifier()
        now = datetime.now(UTC)
        async with self._uow:
            await self._flows.purge_expired(now)
            await self._flows.add(
                state_hash=_state_hash(state),
                server_id=server_id,
                user_id=user_id,
                identity_id=identity_id,
                encrypted_code_verifier=_encrypted(verifier),
                client_id=client.client_id,
                client_auth_method=client.auth_method,
                token_endpoint=metadata.token_endpoint,
                revocation_endpoint=metadata.revocation_endpoint,
                resource=metadata.resource,
                redirect_uri=redirect_uri,
                scope=metadata.scope,
                return_to=return_to,
                expires_at=now + FLOW_LIFETIME,
            )
        logger.info("MCP sign-in started server_id=%s user_id=%s", server_id, user_id)
        return _mcp_oauth.authorization_url(
            metadata, client_id=client.client_id, redirect_uri=redirect_uri, state=state, verifier=verifier
        )

    async def _per_user_server(self, server_id: uuid.UUID, *, lock: bool = False) -> WorkspaceMcpServer:
        server = await self._servers.find(server_id, lock=lock)
        if server is None:
            raise WorkspaceMcpServerNotFoundError(server_id)
        if server.credential_mode != McpCredentialMode.PER_USER:
            raise McpConnectionModeError(server_id)
        return server

    async def _client(
        self,
        server_id: uuid.UUID,
        url: str,
        stored: StoredMcpOAuthClient | None,
        metadata: McpOAuthMetadata,
        redirect_uri: str,
    ) -> McpOAuthClient:
        """The client this sign-in presents: the server's stored one, or one registered now and stored."""
        if stored is not None and (not stored.dynamic or stored.redirect_uri == redirect_uri):
            secret = decrypt_secret(stored.encrypted_secret) if stored.encrypted_secret else None
            return McpOAuthClient(
                client_id=stored.client_id,
                client_secret=secret,
                auth_method=stored.auth_method or _mcp_oauth.auth_method_for(secret, metadata),
            )
        if metadata.registration_endpoint is None:
            raise McpOAuthRegistrationError("The authorization server publishes no registration endpoint")
        registered = await _mcp_oauth.register_client(metadata.registration_endpoint, redirect_uri)
        async with self._uow:
            # Two sign-ins can register at once. The row is locked and read again, and the client
            # another sign-in stored meanwhile is the one both use.
            server = await self._per_user_server(server_id, lock=True)
            if server.url != url:
                raise McpConnectionUpstreamError("The server's URL changed during the sign-in; start it again")
            current = stored_oauth_client(server)
            if current is not None and (not current.dynamic or current.redirect_uri == redirect_uri):
                secret = decrypt_secret(current.encrypted_secret) if current.encrypted_secret else None
                return McpOAuthClient(
                    client_id=current.client_id,
                    client_secret=secret,
                    auth_method=current.auth_method or _mcp_oauth.auth_method_for(secret, metadata),
                )
            server.oauth_client = StoredMcpOAuthClient(
                client_id=registered.client_id,
                encrypted_secret=_encrypted(registered.client_secret) if registered.client_secret else None,
                auth_method=registered.auth_method,
                dynamic=True,
                redirect_uri=redirect_uri,
            ).model_dump()
        logger.info("MCP OAuth client registered server_id=%s", server_id)
        return registered

    async def landing(self, state: str) -> FlowLanding:
        """What a callback's ``state`` names, without consuming it."""
        async with self._uow:
            flow = await self._flows.peek(_state_hash(state))
            if flow is None:
                return FlowLanding(known=False)
            return FlowLanding(known=True, needs_dashboard=flow.identity_id is not None)

    async def abandon(self, state: str) -> None:
        """Drop the pending sign-in a ``state`` names: its authorization server refused it."""
        async with self._uow:
            await self._flows.consume(_state_hash(state))

    async def complete(self, *, state: str, code: str, identity_id: uuid.UUID | None = None) -> CompletedConnection:
        """Finish a sign-in: trade the code for tokens and store them as the user's connection.

        The state is consumed whatever happens next, so it finishes at most one sign-in. One that a
        dashboard identity started finishes only for that identity.

        Raises:
            McpConnectionFlowError: the state is unknown, already used, expired or another person's,
                or the server changed since the sign-in started.
            McpConnectionUpstreamError: the authorization server refused the code.
        """
        async with self._uow:
            flow = await self._flows.consume(_state_hash(state))
            pending = (
                None
                if flow is None
                else _Pending(
                    server_id=flow.server_id,
                    user_id=flow.user_id,
                    identity_id=flow.identity_id,
                    encrypted_code_verifier=flow.encrypted_code_verifier,
                    client_id=flow.client_id,
                    client_auth_method=flow.client_auth_method,
                    token_endpoint=flow.token_endpoint,
                    revocation_endpoint=flow.revocation_endpoint,
                    resource=flow.resource,
                    redirect_uri=flow.redirect_uri,
                    return_to=flow.return_to,
                    expires_at=flow.expires_at,
                )
            )
            server = await self._servers.find(pending.server_id) if pending is not None else None
        if pending is None:
            raise McpConnectionFlowError("This sign-in is unknown or was already finished. Start it again.")
        if _is_expired(pending.expires_at, datetime.now(UTC)):
            raise McpConnectionFlowError("This sign-in took too long. Start it again.")
        if pending.identity_id is not None and pending.identity_id != identity_id:
            logger.warning("MCP sign-in refused reason=other_identity server_id=%s", pending.server_id)
            raise McpConnectionFlowError("This sign-in was started by another person. Start your own.")
        client = (
            _client_for(server, client_id=pending.client_id, auth_method=pending.client_auth_method)
            if server is not None and server.credential_mode == McpCredentialMode.PER_USER
            else None
        )
        if server is None or client is None:
            raise McpConnectionFlowError("The server changed while you were signing in. Start it again.")
        try:
            tokens = await _mcp_oauth.exchange_code(
                pending.token_endpoint,
                client,
                code=code,
                redirect_uri=pending.redirect_uri,
                verifier=decrypt_secret(pending.encrypted_code_verifier),
                resource=pending.resource,
            )
        except McpOAuthTokenError as exc:
            raise McpConnectionUpstreamError(f"Could not finish the sign-in: {exc}") from None
        async with self._uow:
            await self._connections.put(
                pending.server_id,
                pending.user_id,
                encrypted_access_token=_encrypted(tokens.access_token),
                encrypted_refresh_token=_encrypted(tokens.refresh_token) if tokens.refresh_token else None,
                expires_at=_expiry(tokens, datetime.now(UTC)),
                scope=tokens.scope,
                client_id=pending.client_id,
                client_auth_method=pending.client_auth_method,
                token_endpoint=pending.token_endpoint,
                revocation_endpoint=pending.revocation_endpoint,
                resource=pending.resource,
            )
        logger.info("MCP sign-in finished server_id=%s user_id=%s", pending.server_id, pending.user_id)
        return CompletedConnection(
            server_id=server.id, workspace_id=server.workspace_id, server_name=server.name, return_to=pending.return_to
        )

    async def of_user(self, user_id: str, server_ids: Collection[uuid.UUID]) -> list[McpUserConnection]:
        """A user's connections to these servers, oldest first."""
        async with self._uow:
            return await self._connections.for_user(user_id, server_ids)

    async def of_server(self, server_id: uuid.UUID) -> list[McpUserConnection]:
        """Every user's connection to a server, oldest first."""
        async with self._uow:
            return await self._connections.for_server(server_id)

    async def disconnect(self, *, server_id: uuid.UUID, user_id: str) -> None:
        """End a user's connection to a server: drop its tokens, and ask the authorization server to end them.

        The revocation is best effort. The tokens are gone from here whether or not the remote side
        confirms it.

        Raises:
            McpConnectionNotFoundError: the user has no connection to this server.
        """
        async with self._uow:
            connection = await self._connections.find(server_id, user_id)
            if connection is None:
                raise McpConnectionNotFoundError(server_id)
            server = await self._servers.find(server_id)
            revocation_endpoint = connection.revocation_endpoint
            encrypted_token = connection.encrypted_refresh_token or connection.encrypted_access_token
            client_id, auth_method = connection.client_id, connection.client_auth_method
            await self._connections.remove(connection)
            await self._flows.remove_for_user(user_id, [server_id])
        logger.info("MCP connection ended server_id=%s user_id=%s", server_id, user_id)
        if revocation_endpoint is None or server is None:
            return
        try:
            client = _client_for(server, client_id=client_id, auth_method=auth_method)
            if client is not None:
                revoked = await _mcp_oauth.revoke(revocation_endpoint, client, token=decrypt_secret(encrypted_token))
                logger.info("MCP connection revocation server_id=%s confirmed=%s", server_id, revoked)
        except (ValueError, SecretBoxUnavailableError):
            # A token or secret that will not decrypt cannot be revoked remotely; it is already gone here.
            logger.warning("MCP connection revocation skipped server_id=%s reason=unreadable", server_id)


@dataclass(frozen=True, slots=True)
class _Refreshed:
    """What a renewal under the row's lock came to."""

    token: str | None = None
    sign_in_required: bool = False


# One renewal at a time per connection inside this process. The row lock orders renewals across
# workers on PostgreSQL; this also orders them where the database takes no row lock (SQLite).
_refresh_locks: weakref.WeakValueDictionary[tuple[uuid.UUID, str], asyncio.Lock] = weakref.WeakValueDictionary()


class McpUserTokens:
    """Reads the token a user's calls to a per-user MCP server carry, renewing it when it is about to expire.

    The read runs on the session it was built on and commits nothing. A renewal runs in a Unit of
    Work of its own, so the request's transaction is neither committed nor held across the call to
    the authorization server.
    """

    def __init__(
        self,
        connections: McpUserConnectionRepository,
        *,
        unit_of_work: Callable[[], AbstractAsyncContextManager[UnitOfWork]] = create_unit_of_work,
    ) -> None:
        self._connections = connections
        self._unit_of_work = unit_of_work

    async def connected(self, user_id: str | None, server_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        """Which of these servers a user has signed in to; none for a caller with no user."""
        if user_id is None:
            return set()
        return await self._connections.connected_server_ids(user_id, server_ids)

    async def token_for(self, server: WorkspaceMcpServer, user_id: str) -> str:
        """The access token to send to a per-user server for one user.

        Raises:
            McpSignInRequiredError: the user has no connection, or its grant ended and it was removed.
            McpConnectionUnavailableError: the token has expired and could not be renewed right now.
            SecretDecryptionError: a stored token or client secret will not decrypt.
            SecretBoxUnavailableError: ``OTARI_SECRET_KEY`` is not configured.
        """
        connection = await self._connections.find(server.id, user_id)
        if connection is None:
            raise McpSignInRequiredError(server.id, name=server.name, user_id=user_id)
        now = datetime.now(UTC)
        if not _needs_refresh(connection.expires_at, now):
            return decrypt_secret(connection.encrypted_access_token)
        stale, expired = connection.encrypted_access_token, _is_expired(connection.expires_at, now)
        try:
            outcome = await self._renew(server, user_id, stale=stale)
        except DATABASE_ERRORS as exc:
            logger.warning(
                "MCP token renewal could not run server_id=%s error=%s", server.id, describe_database_error(exc)
            )
            outcome = _Refreshed()
        if outcome.sign_in_required:
            raise McpSignInRequiredError(server.id, name=server.name, user_id=user_id)
        if outcome.token is not None:
            return outcome.token
        if expired:
            raise McpConnectionUnavailableError(server.id)
        # The renewal failed for now and the token is still inside its lifetime, so it is used.
        return decrypt_secret(stale)

    async def _renew(self, server: WorkspaceMcpServer, user_id: str, *, stale: str) -> _Refreshed:
        """Renew a connection's tokens unless another caller already has.

        An authorization server may rotate refresh tokens and end the whole grant when a spent one
        is sent again, so renewals of one connection run one at a time: the row is locked, read
        again, and a token that is no longer the ``stale`` one is another caller's renewal and is
        used as it is. A compare-and-swap would not do, because the loser would already have sent
        the spent refresh token.
        """
        key = (server.id, user_id)
        lock = _refresh_locks.get(key)
        if lock is None:
            lock = _refresh_locks[key] = asyncio.Lock()
        async with lock, self._unit_of_work() as uow, uow:
            connections = McpUserConnectionRepository(uow)
            connection = await connections.find(server.id, user_id, lock=True)
            if connection is None:
                return _Refreshed(sign_in_required=True)
            now = datetime.now(UTC)
            if connection.encrypted_access_token != stale and not _needs_refresh(connection.expires_at, now):
                return _Refreshed(token=decrypt_secret(connection.encrypted_access_token))
            client = _client_for(server, client_id=connection.client_id, auth_method=connection.client_auth_method)
            if connection.encrypted_refresh_token is None or client is None:
                # Nothing to renew with. Once the token is past its expiry only a new sign-in helps.
                if not _is_expired(connection.expires_at, now):
                    return _Refreshed()
                await connections.remove(connection)
                logger.info("MCP connection ended server_id=%s user_id=%s reason=not_renewable", server.id, user_id)
                return _Refreshed(sign_in_required=True)
            try:
                tokens = await _mcp_oauth.refresh(
                    connection.token_endpoint,
                    client,
                    refresh_token=decrypt_secret(connection.encrypted_refresh_token),
                    resource=connection.resource,
                )
            except McpOAuthTokenError as exc:
                if not exc.is_permanent:
                    logger.warning(
                        "MCP token renewal failed server_id=%s status=%s error=%s",
                        server.id,
                        exc.status_code,
                        exc.error_code,
                    )
                    return _Refreshed()
                await connections.remove(connection)
                logger.info(
                    "MCP connection ended server_id=%s user_id=%s reason=%s",
                    server.id,
                    user_id,
                    exc.error_code or f"http_{exc.status_code}",
                )
                return _Refreshed(sign_in_required=True)
            await connections.store_tokens(
                connection,
                encrypted_access_token=_encrypted(tokens.access_token),
                encrypted_refresh_token=_encrypted(tokens.refresh_token) if tokens.refresh_token else None,
                expires_at=_expiry(tokens, now),
                # An omitted scope means the grant's scope is unchanged (RFC 6749 section 5.1).
                scope=tokens.scope if tokens.scope is not None else connection.scope,
            )
            logger.info("MCP token renewed server_id=%s user_id=%s", server.id, user_id)
            return _Refreshed(token=tokens.access_token)


__all__ = [
    "CALLBACK_PATH",
    "DASHBOARD_CALLBACK_ROUTE",
    "CompletedConnection",
    "FlowLanding",
    "McpConnectionService",
    "McpUserTokens",
    "StoredMcpOAuthClient",
    "callback_url",
    "checked_return_to",
    "require_per_user_support",
    "stored_oauth_client",
]
