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
import re
import secrets
import uuid
import weakref
from collections.abc import Awaitable, Callable, Collection
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

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
from gateway.services.mcp_stateless import CheckFailure
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
    secret_box_configured,
)
from gateway.services.tools import _mcp_oauth
from gateway.services.tools._mcp_oauth import (
    McpOAuthClient,
    McpOAuthDiscoveryError,
    McpOAuthMetadata,
    McpOAuthPkceUnsupportedError,
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
# How long one renewal may hold a connection's row while it waits on the authorization server.
RENEWAL_DEADLINE_SECONDS = 20.0
CALLBACK_PATH = "/mcp-connections/callback"
# The dashboard route that finishes a sign-in a dashboard session started.
DASHBOARD_CALLBACK_ROUTE = "/#/mcp-connections/callback"
_MAX_RETURN_TO_LENGTH = 2048
# An OAuth ``error`` code as RFC 6749 section 4.1.2.1 spells one, and short.
_SAFE_ERROR = re.compile(r"[a-z0-9_.-]{1,64}")


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
    # When a registered client's secret stops working, as an epoch second; None when it does not expire.
    secret_expires_at: int | None = None

    def serves(self, redirect_uri: str, now: datetime) -> bool:
        """Whether a sign-in can go through this client: a manager's always, a registered one while it holds."""
        if not self.dynamic:
            return True
        expired = self.secret_expires_at is not None and self.secret_expires_at <= now.timestamp()
        return self.redirect_uri == redirect_uri and not expired


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


def _needs_refresh(connection: McpUserConnection, now: datetime) -> bool:
    """Whether a connection's token is close enough to its expiry to renew before use.

    The margin is ``EXPIRY_SKEW``, and never more than a quarter of the token's own lifetime, so
    a token issued for five minutes is not renewed on every request.
    """
    if connection.expires_at is None:
        return False
    lifetime = max(connection.expires_at - connection.updated_at, timedelta(0))
    return connection.expires_at <= now + min(EXPIRY_SKEW, lifetime / 4)


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
    return _client_from(stored_oauth_client(server), client_id=client_id, auth_method=auth_method)


@dataclass(frozen=True, slots=True)
class McpServerRef:
    """What reading a user's token needs of their server, copied while the row is still loaded.

    A request's session may be released while a token is renewed, and a release whose commit fails
    rolls back, which expires every row it loaded; a copy is still readable after that.
    """

    @classmethod
    def of(cls, row: WorkspaceMcpServer) -> McpServerRef:
        return cls(row.id, row.name, stored_oauth_client(row))

    id: uuid.UUID
    name: str
    client: StoredMcpOAuthClient | None


def _client_from(stored: StoredMcpOAuthClient | None, *, client_id: str, auth_method: str) -> McpOAuthClient | None:
    """See :func:`_client_for`, from a client already read off its server."""
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
class CallbackOutcome:
    """Where an authorization server's redirect lands in the dashboard, as the callback page's query.

    One of: a sign-in the dashboard still has to finish (``state`` and ``code``), one finished here
    (``connected``, with ``return_to`` when it named one), or a refusal (``error``).
    """

    state: str | None = None
    code: str | None = None
    # The authorization server's ``iss``, passed on with the code so whoever spends it can check it.
    iss: str | None = None
    connected: uuid.UUID | None = None
    return_to: str | None = None
    error: str | None = None

    def query(self) -> dict[str, str]:
        """The outcome as query parameters, leaving out what it does not carry."""
        fields = {
            "state": self.state,
            "code": self.code,
            "iss": self.iss,
            "connected": str(self.connected) if self.connected else None,
            "return_to": self.return_to,
            "error": self.error,
        }
        return {name: value for name, value in fields.items() if value}


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
    issuer: str | None
    requires_iss: bool
    return_to: str | None
    expires_at: datetime


class McpConnectionEligibility(Protocol):
    """Whether a user may still hold a connection to a server of a workspace, as a sign-in finishes.

    Implemented where workspaces and their members are known, and run inside the caller's Unit of
    Work, taking the locks a removal takes, so a sign-in and a removal that race serialize.
    """

    async def hold(self, workspace_id: uuid.UUID) -> None:
        """Lock the workspace as a member's removal and a workspace's deletion do, before anything else is locked."""
        ...

    async def may_connect(self, workspace_id: uuid.UUID, user_id: str) -> bool:
        """Whether ``user_id`` may hold a connection to a server of ``workspace_id`` now, under ``hold``."""
        ...


@dataclass(frozen=True, slots=True)
class McpRevocation:
    """A token to ask an authorization server to end, read before its connection is deleted."""

    endpoint: str
    client: McpOAuthClient
    token: str


def revocations_for(server: WorkspaceMcpServer, connections: Collection[McpUserConnection]) -> list[McpRevocation]:
    """What revoking these connections to ``server`` sends: one token each, the refresh token where there is one.

    A connection whose server publishes no revocation endpoint, whose client the server no longer
    holds, or whose token will not decrypt is left out: its tokens are only deleted here.
    """
    revocations: list[McpRevocation] = []
    for connection in connections:
        if connection.revocation_endpoint is None:
            continue
        try:
            client = _client_for(server, client_id=connection.client_id, auth_method=connection.client_auth_method)
            if client is None:
                continue
            token = decrypt_secret(connection.encrypted_refresh_token or connection.encrypted_access_token)
        except (SecretDecryptionError, SecretBoxUnavailableError):
            continue
        revocations.append(McpRevocation(endpoint=connection.revocation_endpoint, client=client, token=token))
    return revocations


async def collect_revocations(
    connections: McpUserConnectionRepository,
    servers: WorkspaceMcpServerRepository,
    *,
    server_ids: Collection[uuid.UUID] | None = None,
    user_id: str | None = None,
) -> tuple[list[McpRevocation], int]:
    """What revoking the connections a removal is about to delete sends, and how many it leaves unread.

    Read before the rows go, while their servers still hold the clients the tokens were issued to.
    """
    rows, total = await connections.revocable(server_ids=server_ids, user_id=user_id)
    held = await servers.find_many({row.server_id for row in rows})
    revocations = [
        revocation
        for server in held
        for revocation in revocations_for(server, [row for row in rows if row.server_id == server.id])
    ]
    return revocations, total - len(rows)


# Bounds on revocations sent after a bulk removal, which nothing waits on.
_MAX_BACKGROUND_REVOCATIONS = 500
_BACKGROUND_REVOCATION_CONCURRENCY = 8
_REVOCATION_DEADLINE_SECONDS = 10.0
# Held so a running revocation task is not collected before it finishes.
_revocation_tasks: set[asyncio.Task[None]] = set()


def revoke_in_background(revocations: Collection[McpRevocation], *, unread: int = 0) -> None:
    """Ask authorization servers to end these tokens, without waiting and without failing anything.

    For a removal that has already committed. Each call has a deadline, a few run at once, and
    only counts are logged. Past ``_MAX_BACKGROUND_REVOCATIONS`` the rest are dropped, and they are
    counted as skipped together with ``unread``, the connections the removal ended without reading.
    """
    if unread:
        logger.info("MCP connections ended without revocation count=%d reason=over_page", unread)
    if not revocations:
        return
    batch = list(revocations)[:_MAX_BACKGROUND_REVOCATIONS]
    skipped = len(revocations) - len(batch) + unread
    task = asyncio.get_running_loop().create_task(_revoke_all(batch, skipped), name="mcp-revocations")
    _revocation_tasks.add(task)
    task.add_done_callback(_revocation_tasks.discard)


async def _revoke_all(revocations: list[McpRevocation], skipped: int) -> None:
    gate = asyncio.Semaphore(_BACKGROUND_REVOCATION_CONCURRENCY)

    async def one(revocation: McpRevocation) -> bool:
        async with gate:
            try:
                async with asyncio.timeout(_REVOCATION_DEADLINE_SECONDS):
                    return await _mcp_oauth.revoke(revocation.endpoint, revocation.client, token=revocation.token)
            except TimeoutError:
                return False

    outcomes = await asyncio.gather(*(one(revocation) for revocation in revocations), return_exceptions=True)
    confirmed = sum(1 for outcome in outcomes if outcome is True)
    logger.info("MCP connections revoked in bulk sent=%d confirmed=%d skipped=%d", len(revocations), confirmed, skipped)


async def sign_in_problem(server_url: str) -> CheckFailure | None:
    """Why users could not sign in to a server at this URL, found from its OAuth metadata, or ``None``.

    Run when a manager saves a per-user server, so a setup no sign-in can use is shown then
    rather than to the first user who tries. ``SIGN_IN_UNSUPPORTED`` is the authorization server
    not offering S256 PKCE; ``SIGN_IN_UNAVAILABLE`` is metadata that could not be found or used.
    """
    try:
        await _mcp_oauth.discover(server_url)
    except McpOAuthPkceUnsupportedError:
        return CheckFailure.SIGN_IN_UNSUPPORTED
    except McpOAuthDiscoveryError:
        return CheckFailure.SIGN_IN_UNAVAILABLE
    return None


class McpConnectionService:
    """Runs a user's sign-in to a per-user MCP server, and ends a connection."""

    def __init__(
        self,
        uow: UnitOfWork,
        servers: WorkspaceMcpServerRepository,
        connections: McpUserConnectionRepository,
        flows: McpOAuthFlowRepository,
        config: GatewayConfig,
        *,
        eligibility: McpConnectionEligibility,
    ) -> None:
        self._uow = uow
        self._servers = servers
        self._connections = connections
        self._flows = flows
        self._config = config
        self._eligibility = eligibility

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
        except McpOAuthPkceUnsupportedError as exc:
            raise McpConnectionSetupError(f"This server cannot be signed in to: {exc}") from None
        except McpOAuthDiscoveryError as exc:
            raise McpConnectionUpstreamError(f"Could not start the sign-in: {exc}") from None

        state, verifier = secrets.token_urlsafe(32), _mcp_oauth.new_code_verifier()
        now = datetime.now(UTC)
        async with self._uow:
            # Discovery took seconds. The row is locked and read again, so a sign-in found for the
            # server as it was cannot be stored against the server as a manager has since made it.
            server = await self._per_user_server(server_id, lock=True)
            current = stored_oauth_client(server)
            if server.url != url or current is None or current.client_id != client.client_id:
                raise McpConnectionFlowError("The server changed while the sign-in was starting. Start it again.")
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
                issuer=metadata.issuer,
                requires_iss=metadata.sends_iss,
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
        if stored is not None and stored.serves(redirect_uri, datetime.now(UTC)):
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
            if current is not None and current.serves(redirect_uri, datetime.now(UTC)):
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
                secret_expires_at=registered.secret_expires_at,
            ).model_dump()
        logger.info("MCP OAuth client registered server_id=%s", server_id)
        return registered

    async def receive_callback(
        self, *, state: str | None, code: str | None, error: str | None, iss: str | None = None
    ) -> CallbackOutcome:
        """Decide what an authorization server's redirect comes to. It carries no credential of ours.

        A sign-in a dashboard session started is finished by that session's own dashboard, so its
        code is passed on and nothing is spent here. One started outside a session is finished
        here, on its state alone. A redirect that reports a refusal drops the sign-in it names.
        """
        if not state:
            return CallbackOutcome(error="invalid_request")
        if error or not code:
            await self.abandon(state)
            # A short code the authorization server chose; anything else is not passed on.
            return CallbackOutcome(error=error if error and _SAFE_ERROR.fullmatch(error) else "invalid_request")
        landing = await self.landing(state)
        if not landing.known:
            return CallbackOutcome(error="unknown_state")
        if landing.needs_dashboard:
            return CallbackOutcome(state=state, code=code, iss=iss)
        try:
            done = await self.complete(state=state, code=code, iss=iss)
        except (McpConnectionFlowError, McpConnectionUpstreamError) as exc:
            logger.info("MCP sign-in callback refused reason=%s", type(exc).__name__)
            return CallbackOutcome(error="sign_in_failed")
        return CallbackOutcome(connected=done.server_id, return_to=done.return_to)

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

    async def complete(
        self, *, state: str, code: str, identity_id: uuid.UUID | None = None, iss: str | None = None
    ) -> CompletedConnection:
        """Finish a sign-in: trade the code for tokens and store them as the user's connection.

        The state is consumed whatever happens next, so it finishes at most one sign-in. One that a
        dashboard identity started finishes only for that identity. ``iss`` is the authorization
        response's own issuer (RFC 9207): one that names another server is refused, and so is a
        missing one where the server said it always sends it, since one redirect URI serves every
        server and only this tells their answers apart. The user must still be allowed a connection
        when the tokens are stored, or a removal that ran meanwhile would be undone.

        Raises:
            McpConnectionFlowError: the state is unknown, already used, expired or another person's,
                the answer names another issuer, the server changed since the sign-in started, or
                the user lost access meanwhile.
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
                    issuer=flow.issuer,
                    requires_iss=flow.requires_iss,
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
        if (iss is not None or pending.requires_iss) and (
            iss is None or pending.issuer is None or iss.rstrip("/") != pending.issuer.rstrip("/")
        ):
            logger.warning("MCP sign-in refused reason=issuer_mismatch server_id=%s", pending.server_id)
            raise McpConnectionFlowError("This answer did not come from the server you were signing in to.")
        try:
            client = (
                _client_for(server, client_id=pending.client_id, auth_method=pending.client_auth_method)
                if server is not None and server.credential_mode == McpCredentialMode.PER_USER
                else None
            )
            verifier = decrypt_secret(pending.encrypted_code_verifier)
        except (SecretDecryptionError, SecretBoxUnavailableError):
            # OTARI_SECRET_KEY changed or went away since the sign-in started. The state is spent, so it starts again.
            logger.warning("MCP sign-in secrets unreadable server_id=%s", pending.server_id)
            raise McpConnectionFlowError("This sign-in can no longer be finished. Start it again.") from None
        if server is None or client is None:
            raise McpConnectionFlowError("The server changed while you were signing in. Start it again.")
        # Copied now: the row is read again below into this same object.
        url_before, workspace_id = server.url, server.workspace_id
        try:
            tokens = await _mcp_oauth.exchange_code(
                pending.token_endpoint,
                client,
                code=code,
                redirect_uri=pending.redirect_uri,
                verifier=verifier,
                resource=pending.resource,
            )
        except McpOAuthTokenError as exc:
            if exc.refuses_client:
                await self._forget_registered_client(pending.server_id, pending.client_id)
            raise McpConnectionUpstreamError(f"Could not finish the sign-in: {exc}") from None
        refusal: str | None = None
        async with self._uow:
            # Locks in the order the removals take them: the workspace (a member's removal, the
            # workspace's deletion), then the server row (its edit or deletion), then the user
            # (their deletion, inside ``may_connect``). Whichever of the two commits second sees
            # the other.
            await self._eligibility.hold(workspace_id)
            # The exchange took seconds. Tokens issued for the server as it was are not stored
            # against the server as a manager has since made it.
            current = await self._servers.find(pending.server_id, lock=True)
            client_now = stored_oauth_client(current) if current is not None else None
            if (
                current is None
                or current.url != url_before
                or current.credential_mode != McpCredentialMode.PER_USER
                or client_now is None
                or client_now.client_id != pending.client_id
            ):
                refusal = "The server changed while you were signing in. Start it again."
            elif not await self._eligibility.may_connect(current.workspace_id, pending.user_id):
                logger.info("MCP sign-in refused reason=no_longer_eligible server_id=%s", pending.server_id)
                refusal = "You no longer have access to this server's workspace."
            else:
                # Under the locks above, so a removal waiting on them sees this connection and ends it.
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
        if refusal is not None:
            # The authorization server issued these tokens and nothing will hold them: ask it to end them.
            if pending.revocation_endpoint is not None:
                revoke_in_background(
                    [
                        McpRevocation(
                            endpoint=pending.revocation_endpoint,
                            client=client,
                            token=tokens.refresh_token or tokens.access_token,
                        )
                    ]
                )
            raise McpConnectionFlowError(refusal)
        logger.info("MCP sign-in finished server_id=%s user_id=%s", pending.server_id, pending.user_id)
        return CompletedConnection(
            server_id=server.id, workspace_id=server.workspace_id, server_name=server.name, return_to=pending.return_to
        )

    async def _forget_registered_client(self, server_id: uuid.UUID, client_id: str) -> None:
        """Drop a client Otari registered that its authorization server no longer accepts.

        The next sign-in registers a new one. A manager's own client is theirs to fix and is kept.
        """
        async with self._uow:
            server = await self._servers.find(server_id, lock=True)
            stored = stored_oauth_client(server) if server is not None else None
            if server is not None and stored is not None and stored.dynamic and stored.client_id == client_id:
                server.oauth_client = None
                logger.info("MCP OAuth client dropped server_id=%s reason=refused", server_id)

    async def disconnect(self, *, server_id: uuid.UUID, user_id: str) -> None:
        """End a user's connection to a server: drop its tokens, and ask the authorization server to end them.

        The revocation is best effort. The tokens are gone from here whether or not the remote side
        confirms it.

        Raises:
            McpConnectionNotFoundError: the user has no connection to this server.
        """
        async with self._uow:
            # Locked, so a renewal in flight finishes first and it is the token it stored that is revoked.
            connection = await self._connections.find(server_id, user_id, lock=True)
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
                # Bounded as the background revocations are: the tokens are already gone here, and the
                # person disconnecting should not wait on a slow authorization server.
                try:
                    async with asyncio.timeout(_REVOCATION_DEADLINE_SECONDS):
                        revoked = await _mcp_oauth.revoke(
                            revocation_endpoint, client, token=decrypt_secret(encrypted_token)
                        )
                except TimeoutError:
                    revoked = False
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

    The read runs on the session it was built on. A renewal runs in a Unit of Work of its own,
    which holds the row's lock across the call to the authorization server. ``release`` hands the
    caller's own pooled connection back first, so a request waiting on a renewal holds none: with
    it, a burst of requests for one user costs the pool one connection, not one each plus one.
    """

    def __init__(
        self,
        connections: McpUserConnectionRepository,
        *,
        unit_of_work: Callable[[], AbstractAsyncContextManager[UnitOfWork]] = create_unit_of_work,
        release: Callable[[], Awaitable[object]] | None = None,
    ) -> None:
        self._connections = connections
        self._unit_of_work = unit_of_work
        self._release = release

    async def connected(self, user_id: str | None, server_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        """Which of these servers a user has signed in to; none for a caller with no user."""
        if user_id is None:
            return set()
        return await self._connections.connected_server_ids(user_id, server_ids)

    async def token_for(self, row: WorkspaceMcpServer | McpServerRef, user_id: str) -> str:
        """The access token to send to a per-user server for one user.

        Raises:
            McpSignInRequiredError: the user has no connection, or its grant ended and it was removed.
            McpConnectionUnavailableError: the token has expired and could not be renewed right now.
            SecretDecryptionError: a stored token or client secret will not decrypt.
            SecretBoxUnavailableError: ``OTARI_SECRET_KEY`` is not configured.
        """
        # Copied before anything may release the session and expire the row.
        server = row if isinstance(row, McpServerRef) else McpServerRef.of(row)
        connection = await self._connections.find(server.id, user_id)
        if connection is None:
            raise McpSignInRequiredError(server.id, name=server.name, user_id=user_id)
        now = datetime.now(UTC)
        stale = decrypt_secret(connection.encrypted_access_token)
        if not _needs_refresh(connection, now):
            return stale
        outcome = await self._shielded_renew(server, user_id, stale=stale, rejected=False)
        if outcome.sign_in_required:
            raise McpSignInRequiredError(server.id, name=server.name, user_id=user_id)
        if outcome.token is not None:
            return outcome.token
        if _is_expired(connection.expires_at, now):
            raise McpConnectionUnavailableError(server.id)
        # The renewal failed for now and the token is still inside its lifetime, so it is used.
        return stale

    async def renew_rejected(self, row: WorkspaceMcpServer | McpServerRef, user_id: str, rejected: str) -> str:
        """A new token after the MCP server refused ``rejected`` with a 401, for one retry.

        A token can be revoked before its recorded expiry, or carry no expiry at all, so a refusal
        is the only sign it ended. Renewed once whatever its recorded expiry; with nothing to renew
        with, or a grant the authorization server ended, the connection is removed.

        Raises:
            McpSignInRequiredError: the connection is gone, or was removed now.
            McpConnectionUnavailableError: the renewal failed for now.
        """
        server = row if isinstance(row, McpServerRef) else McpServerRef.of(row)
        outcome = await self._shielded_renew(server, user_id, stale=rejected, rejected=True)
        if outcome.sign_in_required:
            raise McpSignInRequiredError(server.id, name=server.name, user_id=user_id)
        if outcome.token is None:
            raise McpConnectionUnavailableError(server.id)
        return outcome.token

    async def _shielded_renew(self, server: McpServerRef, user_id: str, *, stale: str, rejected: bool) -> _Refreshed:
        """``_renew``, carried through to its own commit even if the request waiting on it is cancelled.

        A renewal the authorization server answered has rotated the refresh token, and only storing
        its answer keeps the grant: a request that goes away must not take that store with it.
        """
        try:
            return await asyncio.shield(self._renew(server, user_id, stale=stale, rejected=rejected))
        except DATABASE_ERRORS as exc:
            logger.warning(
                "MCP token renewal could not run server_id=%s error=%s", server.id, describe_database_error(exc)
            )
            return _Refreshed()

    async def _renew(self, server: McpServerRef, user_id: str, *, stale: str, rejected: bool) -> _Refreshed:
        """Renew a connection's tokens unless another caller already has.

        An authorization server may rotate refresh tokens and end the whole grant when a spent one
        is sent again, so renewals of one connection run one at a time: the row is locked, read
        again, and a token that is no longer the ``stale`` one is another caller's renewal and is
        used as it is. A compare-and-swap would not do, because the loser would already have sent
        the spent refresh token. ``rejected`` says the MCP server refused ``stale``: it is then
        renewed whatever its recorded expiry, and with nothing to renew with the connection ends.
        """
        if self._release is not None:
            await self._release()
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
            current = decrypt_secret(connection.encrypted_access_token)
            if current != stale and (rejected or not _needs_refresh(connection, now)):
                return _Refreshed(token=current)
            client = _client_from(
                server.client, client_id=connection.client_id, auth_method=connection.client_auth_method
            )
            if connection.encrypted_refresh_token is None or client is None:
                # Nothing to renew with. Once the token is refused or past its expiry only a new sign-in helps.
                if not rejected and not _is_expired(connection.expires_at, now):
                    return _Refreshed()
                await connections.remove(connection)
                logger.info("MCP connection ended server_id=%s user_id=%s reason=not_renewable", server.id, user_id)
                return _Refreshed(sign_in_required=True)
            try:
                # A deadline on the exchange: its timeouts are per read, and the row stays locked meanwhile.
                # It covers sending and reading only, so an answer that arrived is always stored below.
                tokens = await _mcp_oauth.refresh(
                    connection.token_endpoint,
                    client,
                    refresh_token=decrypt_secret(connection.encrypted_refresh_token),
                    resource=connection.resource,
                    deadline=RENEWAL_DEADLINE_SECONDS,
                )
            except McpOAuthTokenError as exc:
                # A refused client is a setting to fix, which then renews every connection again.
                # Only a client Otari registered itself has nobody to fix it: it is replaced at the
                # next sign-in, so its grants end here.
                registered = server.client
                gone = exc.is_permanent or (exc.refuses_client and registered is not None and registered.dynamic)
                if not gone:
                    logger.warning(
                        "MCP token renewal failed server_id=%s status=%s error=%s",
                        server.id,
                        exc.status_code,
                        exc.error_code,
                    )
                    return _Refreshed()
                await connections.remove(connection)
                if not exc.is_permanent:
                    # The refused client goes too, or the next sign-in would be sent through it.
                    servers = WorkspaceMcpServerRepository(uow)
                    row = await servers.find(server.id, lock=True)
                    held = stored_oauth_client(row) if row is not None else None
                    if row is not None and held is not None and held.dynamic and held.client_id == client.client_id:
                        row.oauth_client = None
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
    "McpConnectionEligibility",
    "McpRevocation",
    "McpServerRef",
    "revocations_for",
    "collect_revocations",
    "revoke_in_background",
    "sign_in_problem",
    "CallbackOutcome",
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
