"""Workspace-scoped MCP server configuration: CRUD, and the request-path resolve.

Ported from otari-ai's `MCPServerService` (otari#658). An MCP server is stored
per workspace, its bearer token encrypted on save and decrypted when a request
names the server by id, and never returned in a public response.

**Where it plugs in.** A request names stored servers with `mcp_server_ids`.
They are resolved here, through
:func:`resolve_workspace_mcp_servers`, called at admission where the request's session is live and
`RequestContext.workspace_id` already names the workspace its key belongs to.
That is the seam otari#655 settled and otari#678 wrote down; MCP is the
exception that decision names, because there is no deployment-wide server list
for a workspace row to narrow, so a row here is the only source of the
configuration rather than a narrowing of one.

**Deliberate divergences from otari-ai's version:**

- Authorization is this repository's own workspace check
  (`services.tenancy.authorization`), so a workspace owner/admin qualifies and
  not only an organization owner/admin. Every *write* is management-gated,
  because these rows decide which external endpoints the gateway connects to
  and authenticates against on the workspace's behalf. The list read is not
  (otari-ai#1942): those endpoints act on every request a member sends through
  the workspace, so seeing them is a member's to do, and a reader who cannot
  manage the workspace gets any credential in the URL masked.
- URL safety is one check in this layer rather than the platform's split
  between a synchronous ingress validator and a service-layer SSRF pass. This
  repository's `validate_mcp_url` is a single async function that already does
  both, and these routes are async, so the reason for the split does not apply.
- No analytics events, and no `status` field on the public shape. The
  platform's is derived from configuration alone; this one publishes `health`
  instead, the outcome of actually connecting, which a server gets when it is
  stored, when its connection details change, and on request.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.exceptions.tools_exceptions import (
    WorkspaceMcpServerAlreadyExistsError,
    WorkspaceMcpServerCheckBusyError,
    WorkspaceMcpServerLimitReachedError,
    WorkspaceMcpServerNotFoundError,
    WorkspaceMcpServerUnsafeUrlError,
)
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig, ResolvedMcpServer
from gateway.models.tenancy import User
from gateway.models.tools import WorkspaceMcpServer
from gateway.repositories.tenancy import WorkspaceRepository

# The module as well as the names, so the check is looked up at call time and a
# test can stand in for the network.
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import CheckFailure, McpCapacityUnavailable, ServerCheck
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tenancy import authorization
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.url_safety import UnsafeURLError, redact_url_secrets, validate_mcp_url

# What a workspace may configure. A resolved request opens a session to every
# server it names, so this bounds the fan-out one workspace can ask a gateway
# process for. Same value the platform uses: generous for any real setup, low
# enough that the list is not a resource in itself.
MAX_MCP_SERVERS_PER_WORKSPACE = 50

# Bounds on the tool allow-list, which is arbitrary JSON stored on the row and
# read again on the completion path. Every other field here is bounded; without
# these two the list is the one way to put an unbounded value in the table.
MAX_ALLOWED_TOOLS = 200
MAX_TOOL_NAME_LENGTH = 256

_MAX_LIST_LIMIT = 1000

# Sentinel for "this PATCH did not mention the token", which is not the same as
# "this PATCH set it to null": see `WorkspaceMcpServerUpdate`.
_UNSET = object()


class WorkspaceMcpServerCreate(BaseModel):
    """Request body for registering a server.

    ``authorization_token`` is never stored as sent: it is encrypted with
    ``OTARI_SECRET_KEY`` and only the ciphertext is kept, the same convention
    `providers.ProviderCredential` and `OrgProviderKey` already use.
    """

    name: str = Field(min_length=1, max_length=128, description="Label for the server, unique within the workspace")
    url: str = Field(min_length=1, max_length=2048, description="Streamable HTTP MCP endpoint")
    authorization_token: str | None = Field(
        default=None,
        max_length=8192,
        description="Bearer token for the server; requires an https URL. Encrypted at rest, never returned",
    )
    purpose_hint: str | None = Field(
        default=None, max_length=2000, description="Hint prepended to the system message to help the model choose"
    )
    allowed_tools: list[Annotated[str, Field(max_length=MAX_TOOL_NAME_LENGTH)]] | None = Field(
        default=None,
        max_length=MAX_ALLOWED_TOOLS,
        description="Allow-list of tool names; null exposes every tool the server offers",
    )
    enabled: bool = Field(default=True, description="Whether a request naming this server actually reaches it")


class WorkspaceMcpServerUpdate(BaseModel):
    """Partial update. Only the fields the caller sets are applied.

    ``authorization_token`` has three states rather than two, which is what a
    write-only field needs: omit it to leave the stored token alone, send
    ``""`` to clear it, send a value to rotate it. An explicit ``null`` also
    leaves it alone, matching the platform: a client that serializes its whole
    form back, with an empty token box it never filled in, must not destroy a
    token it was never shown.
    """

    # ``SkipJsonSchema[None]`` on the three that back NOT NULL columns: ``None``
    # is this schema's "not sent" marker, and the validator below refuses it as a
    # value, so the published schema must not advertise ``null`` as accepted.
    # Without it the OpenAPI spec and the generated TypeScript client tell a
    # client that ``{"url": null}`` is valid and the server answers 422.
    name: Annotated[str, Field(min_length=1, max_length=128)] | SkipJsonSchema[None] = None
    url: Annotated[str, Field(min_length=1, max_length=2048)] | SkipJsonSchema[None] = None
    authorization_token: str | None = Field(default=None, max_length=8192)
    purpose_hint: str | None = Field(default=None, max_length=2000)
    allowed_tools: list[Annotated[str, Field(max_length=MAX_TOOL_NAME_LENGTH)]] | None = Field(
        default=None, max_length=MAX_ALLOWED_TOOLS
    )
    enabled: bool | SkipJsonSchema[None] = None

    @model_validator(mode="after")
    def _reject_explicit_nulls(self) -> WorkspaceMcpServerUpdate:
        """Refuse an explicit ``null`` for a column that cannot hold one.

        ``None`` is this schema's "not sent" marker, so a caller that means to
        clear a required field is expressing something the row cannot store.
        Caught here rather than at the flush: a ``NOT NULL`` violation arrives
        as the same ``IntegrityError`` the unique index raises, and the service
        would report a name collision that never happened.
        ``purpose_hint`` and ``allowed_tools`` are nullable and are deliberately
        clearable this way, and the token has its own three-state rule above.
        """
        nulled = [
            field
            for field in ("name", "url", "enabled")
            if field in self.model_fields_set and getattr(self, field) is None
        ]
        if nulled:
            raise ValueError(f"{', '.join(nulled)} cannot be null; omit the field to leave it unchanged")
        return self


class WorkspaceMcpServerHealth(BaseModel):
    """The last connection check: whether the gateway reached the server, and what it found.

    Describes the server as the gateway saw it at ``checked_at``, not now: a
    server that answered then can be down since, and the request path checks
    nothing in advance on the strength of this.
    """

    status: Literal["healthy", "failing"]
    checked_at: str
    error: CheckFailure | None = Field(default=None, description="Why the check failed; null when healthy")
    http_status: int | None = Field(default=None, description="The HTTP status the server refused with, if any")
    tool_count: int | None = Field(
        default=None, description="Live tools the allow-list admits; null when the check failed"
    )
    missing_allowed_tools: list[str] = Field(
        default_factory=list, description="Allow-listed names the server did not offer"
    )
    tools: list[str] = Field(
        default_factory=list,
        description=f"Every tool name the server offers, allowed or not, up to {MAX_ALLOWED_TOOLS}; empty on a failure",
    )


class WorkspaceMcpServerCheckRequest(BaseModel):
    """Connection details to check without storing them, for a form to test before it saves.

    ``server_id`` names a stored server whose token stands in when
    ``authorization_token`` is omitted, so an edit form can test a new URL with
    a token it was never shown. ``""`` checks with no token at all.
    """

    url: str = Field(min_length=1, max_length=2048, description="Streamable HTTP MCP endpoint")
    authorization_token: str | None = Field(
        default=None, max_length=8192, description="Bearer token; omit to use the stored server's, if any"
    )
    allowed_tools: list[Annotated[str, Field(max_length=MAX_TOOL_NAME_LENGTH)]] | None = Field(
        default=None, max_length=MAX_ALLOWED_TOOLS, description="The allow-list to report the catalog against"
    )
    server_id: uuid.UUID | None = Field(default=None, description="A stored server whose token to use")


class WorkspaceMcpServerPublic(BaseModel):
    """The API-facing shape. Never carries the token, only whether one is set.

    No ``last4``-style prefix either, unlike `OrgProviderKeyPublic`: a provider
    key's last four digits let an operator match a stored key against the one
    in their provider's console, and there is no equivalent workflow for an MCP
    bearer token.
    """

    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    url: str
    purpose_hint: str | None
    allowed_tools: list[str] | None
    enabled: bool
    has_token: bool
    health: WorkspaceMcpServerHealth | None = Field(description="The last connection check; null until one has run")
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, server: WorkspaceMcpServer, *, redact_url: bool = False) -> WorkspaceMcpServerPublic:
        """Shape one row, optionally masking any credential embedded in its URL.

        ``redact_url`` is what keeps the member read from being a wider
        disclosure than the management read it was split from: the token column
        was never returned either way, but nothing rejects a URL that carries a
        credential in its userinfo (``https://user:secret@host``) or a query
        parameter, so a caller who may read the row without managing it gets
        those components masked. A manager sees the URL as stored, because the
        edit form restates it and would otherwise save the mask back.
        """
        return cls(
            id=server.id,
            workspace_id=server.workspace_id,
            name=server.name,
            url=redact_url_secrets(server.url) if redact_url else server.url,
            purpose_hint=server.purpose_hint,
            allowed_tools=server.allowed_tools,
            enabled=server.enabled,
            has_token=server.encrypted_token is not None,
            health=WorkspaceMcpServerHealth.model_validate(server.health) if server.health is not None else None,
            created_at=server.created_at.isoformat(),
            updated_at=server.updated_at.isoformat(),
        )


class WorkspaceMcpServersPublic(BaseModel):
    data: list[WorkspaceMcpServerPublic]
    count: int


async def _validate_url(url: str, *, has_token: bool) -> None:
    """Run the same SSRF/TLS check a request-body MCP server faces, at write time.

    Checked here as well as on the request path (`services/tools/_mcp_admission.py`)
    rather than instead of it: this catches an operator's mistake at the moment
    they make it, and the request-path check is what still holds when DNS moves
    under a URL that was safe when it was stored.
    """
    try:
        await validate_mcp_url(url, has_authorization_token=has_token)
    except UnsafeURLError as exc:
        raise WorkspaceMcpServerUnsafeUrlError(str(exc)) from exc


def _encrypted(token: str) -> str:
    try:
        return encrypt_secret(token)
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError("MCP server authorization tokens") from None


def _stored_token(server: WorkspaceMcpServer) -> str | None:
    """The row's token in the clear.

    Raises:
        SecretDecryptionError: the stored token will not decrypt.
        SecretBoxUnavailableError: ``OTARI_SECRET_KEY`` is not configured.
    """
    return decrypt_secret(server.encrypted_token) if server.encrypted_token else None


def _connect(url: str, token: str | None, *, server_id: uuid.UUID | None = None) -> Awaitable[ServerCheck]:
    """The connection check itself, looked up at call time so a test can stand in for the network."""
    return mcp_stateless.check_stored_server(
        ResolvedMcpServer(id=server_id or uuid.uuid4(), name="check", url=url, authorization_token=token)
    )


async def _check(server: WorkspaceMcpServer) -> ServerCheck:
    """Run the connection check on a stored row, settling its URL and token first.

    The same order the request path takes: a token that will not decrypt and a
    URL that is no longer safe each stop the check before anything is sent.

    Raises:
        McpCapacityUnavailable: no check slot was free, so nothing was checked.
    """
    try:
        token = _stored_token(server)
    except (SecretDecryptionError, SecretBoxUnavailableError):
        return ServerCheck(tool_names=[], failure=CheckFailure.CREDENTIALS_UNAVAILABLE)
    try:
        await validate_mcp_url(server.url, has_authorization_token=token is not None)
    except UnsafeURLError:
        return ServerCheck(tool_names=[], failure=CheckFailure.UNSAFE_URL)
    return await _connect(server.url, token, server_id=server.id)


def _health(allowed_tools: list[str] | None, check: ServerCheck) -> WorkspaceMcpServerHealth:
    """Read a check against an allow-list: ``None`` admits every tool and ``[]`` none, as the tool loop reads it."""
    offered = list(dict.fromkeys(name for name in check.tool_names if len(name) <= MAX_TOOL_NAME_LENGTH))
    failed = check.failure is not None
    allowed = None if allowed_tools is None else list(dict.fromkeys(allowed_tools))
    return WorkspaceMcpServerHealth(
        status="failing" if failed else "healthy",
        checked_at=datetime.now(UTC).isoformat(),
        error=check.failure,
        http_status=check.http_status,
        tool_count=None if failed else len(offered) if allowed is None else len(set(allowed) & set(offered)),
        missing_allowed_tools=[] if failed or allowed is None else [name for name in allowed if name not in offered],
        tools=offered[:MAX_ALLOWED_TOOLS],
    )


async def resolve_workspace_mcp_servers(
    db: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    server_ids: list[uuid.UUID],
) -> list[McpServerConfig]:
    """Swap a request's ``mcp_server_ids`` for the workspace's stored configs.

    IDs are de-duplicated with their order preserved.
    An ID naming no server *in this workspace* raises :class:`WorkspaceMcpServerNotFoundError`.
    A disabled server is skipped rather than refused.

    No authorization check, and none is missing: ``workspace_id`` comes off the
    key that authenticated the request (`services/workspace_scope.py`), never
    off a header, so there is no caller-supplied scope here to verify. An id
    belonging to another workspace is simply not found, which is also what
    keeps this from being an existence oracle across tenants.

    Raises `secret_box.SecretDecryptionError` when a stored token will not
    decrypt. Dropping the token and connecting anyway would silently send an
    unauthenticated request to a server the workspace configured a credential
    for, so the request fails instead.
    """
    if not server_ids:
        return []

    ordered_ids = list(dict.fromkeys(server_ids))
    rows = (
        (
            await db.execute(
                select(WorkspaceMcpServer).where(
                    WorkspaceMcpServer.workspace_id == workspace_id,
                    WorkspaceMcpServer.id.in_(ordered_ids),
                )
            )
        )
        .scalars()
        .all()
    )
    by_id = {row.id: row for row in rows}

    missing = [server_id for server_id in ordered_ids if server_id not in by_id]
    if missing:
        raise WorkspaceMcpServerNotFoundError(missing[0])

    resolved: list[McpServerConfig] = []
    for server_id in ordered_ids:
        row = by_id[server_id]
        if not row.enabled:
            continue
        resolved.append(
            McpServerConfig(
                name=row.name,
                url=row.url,
                authorization_token=decrypt_secret(row.encrypted_token) if row.encrypted_token else None,
                purpose_hint=row.purpose_hint,
                allowed_tools=row.allowed_tools,
            )
        )
    return resolved


async def resolve_workspace_mcp_server(
    db: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
) -> ResolvedMcpServer | None:
    """Resolve one stored server for the caller-orchestrated MCP endpoints.

    The singular sibling of :func:`resolve_workspace_mcp_servers`. It differs
    from that one in the two ways the stored-server endpoints need. It reports a
    disabled server instead of skipping it, because a disabled server is a
    named 404 here rather than one entry quietly missing from a list; and it
    returns the id, so a revision can be derived over the configuration that
    was actually resolved.

    ``None`` means no such server *in this workspace*, which covers an id
    belonging to another one: the same non-oracle answer the plural resolver
    gives, since ``workspace_id`` comes off the authenticating key.

    Raises `secret_box.SecretDecryptionError` when a stored token will not
    decrypt, for the same reason the plural resolver does: connecting without a
    credential the workspace configured would send an unauthenticated request
    to a server that expects one.
    """
    row = (
        await db.execute(
            select(WorkspaceMcpServer).where(
                WorkspaceMcpServer.workspace_id == workspace_id,
                WorkspaceMcpServer.id == server_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    return ResolvedMcpServer(
        id=row.id,
        name=row.name,
        url=row.url,
        authorization_token=decrypt_secret(row.encrypted_token) if row.encrypted_token else None,
        enabled=row.enabled,
        purpose_hint=row.purpose_hint,
        allowed_tools=row.allowed_tools,
    )


class WorkspaceMcpServers:
    """The MCP servers a workspace may reach, as a request reads them.

    A caller connects to a returned server directly; nothing here proxies MCP traffic.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def resolve_many(self, workspace_id: uuid.UUID, server_ids: list[uuid.UUID]) -> list[McpServerConfig]:
        """See :func:`resolve_workspace_mcp_servers`."""
        return await resolve_workspace_mcp_servers(self._db, workspace_id=workspace_id, server_ids=server_ids)

    async def resolve_one(self, workspace_id: uuid.UUID, server_id: uuid.UUID) -> ResolvedMcpServer | None:
        """See :func:`resolve_workspace_mcp_server`."""
        return await resolve_workspace_mcp_server(self._db, workspace_id=workspace_id, server_id=server_id)


class WorkspaceMcpServerService:
    """CRUD for a workspace's MCP servers. Writes are management-gated; the list is not."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.organizations = OrganizationService(db, membership_listener=None)

    async def _require_management(self, user: User, workspace_id: uuid.UUID) -> uuid.UUID:
        """Resolve the workspace and confirm the caller may manage it.

        Writes only. The list used to sit behind this gate too, on the argument
        that a server's URL is not something every member needs; the roles
        matrix settled it the other way (otari-ai#1942): a member may *view*
        what shapes their own requests, and these servers already act on every
        request the member sends through the workspace. Tokens were never
        exposed either way (`WorkspaceMcpServerPublic` carries `has_token`
        alone), so the read gate is member visibility, in `list_servers`.
        """
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        await authorization.require_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )
        return workspace.id

    async def _get_or_404(self, workspace_id: uuid.UUID, server_id: uuid.UUID) -> WorkspaceMcpServer:
        server = await self.db.get(WorkspaceMcpServer, server_id)
        if server is None or server.workspace_id != workspace_id:
            raise WorkspaceMcpServerNotFoundError(server_id)
        return server

    async def _record_check(self, server: WorkspaceMcpServer) -> bool:
        """Check a stored server's connection and save the outcome as its health.

        The transaction is ended first, so no pooled connection is held while
        the gateway waits on a remote server for up to the discovery deadline.
        Returns false when no check slot was free, leaving the row's health as
        it was.

        An edit can land while the check waits. The row is re-read under a lock
        before the write, and a check of a URL, token or allow-list the row no
        longer holds is dropped: that edit cleared health and runs its own.
        """
        checked = (server.url, server.encrypted_token, server.allowed_tools)
        await self.db.commit()
        try:
            check = await _check(server)
        except McpCapacityUnavailable:
            logger.info("MCP server check skipped server_id=%s reason=capacity", server.id)
            return False
        await self.db.refresh(server, with_for_update=True)
        if (server.url, server.encrypted_token, server.allowed_tools) != checked:
            await self.db.commit()
            logger.info("MCP server check dropped server_id=%s reason=edited", server.id)
            return True
        server.health = _health(server.allowed_tools, check).model_dump(mode="json")
        await self.db.commit()
        await self.db.refresh(server)
        logger.info("MCP server checked server_id=%s outcome=%s", server.id, check.failure or "ok")
        return True

    async def list_servers(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> WorkspaceMcpServersPublic:
        """List a page of the workspace's servers, plus the total.

        Any member who can see the workspace may call this.
        A caller who cannot manage the workspace gets any credential in a server URL masked.
        """
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        workspace_id = workspace.id
        manages = await authorization.has_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )
        limit = min(limit, _MAX_LIST_LIMIT)

        count = (
            await self.db.execute(
                select(func.count())
                .select_from(WorkspaceMcpServer)
                .where(WorkspaceMcpServer.workspace_id == workspace_id)
            )
        ).scalar_one()
        servers = (
            (
                await self.db.execute(
                    select(WorkspaceMcpServer)
                    .where(WorkspaceMcpServer.workspace_id == workspace_id)
                    .order_by(WorkspaceMcpServer.created_at, WorkspaceMcpServer.id)
                    .offset(skip)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return WorkspaceMcpServersPublic(
            data=[WorkspaceMcpServerPublic.from_model(server, redact_url=not manages) for server in servers],
            count=count,
        )

    async def create_server(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        request: WorkspaceMcpServerCreate,
    ) -> WorkspaceMcpServerPublic:
        """Register a server, encrypting its token before it is stored."""
        resolved_workspace_id = await self._require_management(user, workspace_id)
        await _validate_url(request.url, has_token=bool(request.authorization_token))

        # The workspace lock stops a concurrent create from invalidating the count below before this insert.
        await WorkspaceRepository(self.db).lock(resolved_workspace_id)

        count = (
            await self.db.execute(
                select(func.count())
                .select_from(WorkspaceMcpServer)
                .where(WorkspaceMcpServer.workspace_id == resolved_workspace_id)
            )
        ).scalar_one()
        if count >= MAX_MCP_SERVERS_PER_WORKSPACE:
            raise WorkspaceMcpServerLimitReachedError(workspace_id, MAX_MCP_SERVERS_PER_WORKSPACE)

        server = WorkspaceMcpServer(
            workspace_id=resolved_workspace_id,
            name=request.name,
            url=request.url,
            encrypted_token=_encrypted(request.authorization_token) if request.authorization_token else None,
            purpose_hint=request.purpose_hint,
            allowed_tools=request.allowed_tools,
            enabled=request.enabled,
        )
        self.db.add(server)
        # The unique index is the arbiter, not a preceding existence check: two
        # concurrent creates of the same name would both pass such a check and
        # one would still have to lose here. It is also the only constraint this
        # table can violate, which is what makes the blanket catch exact: the
        # schema validated every column before this point.
        try:
            await self.db.flush()
        except IntegrityError:
            await self.db.rollback()
            raise WorkspaceMcpServerAlreadyExistsError(workspace_id, request.name) from None

        await self.db.commit()
        # After the commit, so a server that fails its check is still stored:
        # it may be registered before it is up, and `health` says what failed.
        await self._record_check(server)
        return WorkspaceMcpServerPublic.from_model(server)

    async def update_server(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        server_id: uuid.UUID,
        request: WorkspaceMcpServerUpdate,
    ) -> WorkspaceMcpServerPublic:
        """Apply a partial update. See :class:`WorkspaceMcpServerUpdate` for the token's three states."""
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        fields = request.model_dump(exclude_unset=True)

        new_token = fields.pop("authorization_token", _UNSET)
        if new_token == "":
            effective_has_token = False
        elif isinstance(new_token, str):
            effective_has_token = True
        else:  # omitted, or an explicit null
            effective_has_token = server.encrypted_token is not None

        # Re-validated against the *effective* post-update state, and only when
        # the URL or the token actually moves. A PATCH that sets only a token is
        # still checked against the stored URL, which is the only place an http
        # URL newly paired with a bearer token can be caught. A metadata-only
        # edit (rename, disable, change the tool list) does no DNS lookup at
        # all, so disabling a server never depends on its host still resolving.
        url_changed = "url" in fields and fields["url"] != server.url
        if url_changed or isinstance(new_token, str):
            await _validate_url(fields.get("url") or server.url, has_token=effective_has_token)
        # What the stored health described. Rechecked after the commit; cleared
        # here so a check that cannot run leaves no health rather than a stale one.
        connection_changed = (
            url_changed
            or isinstance(new_token, str)
            or ("allowed_tools" in fields and fields["allowed_tools"] != server.allowed_tools)
        )

        if new_token == "":
            server.encrypted_token = None
        elif isinstance(new_token, str):
            server.encrypted_token = _encrypted(new_token)

        for field, value in fields.items():
            setattr(server, field, value)
        if connection_changed:
            server.health = None

        # Read before the flush: the rollback below expires every loaded
        # instance, so reading ``server.name`` inside the handler would emit a
        # lazy SELECT from a synchronous context and raise ``MissingGreenlet``
        # instead of the conflict the caller is owed.
        attempted_name = server.name
        try:
            await self.db.flush()
        except IntegrityError:
            await self.db.rollback()
            raise WorkspaceMcpServerAlreadyExistsError(workspace_id, attempted_name) from None

        await self.db.commit()
        if connection_changed:
            await self._record_check(server)
        else:
            await self.db.refresh(server)
        return WorkspaceMcpServerPublic.from_model(server)

    async def check_connection(
        self, *, user: User, workspace_id: uuid.UUID, request: WorkspaceMcpServerCheckRequest
    ) -> WorkspaceMcpServerHealth:
        """Check connection details without storing anything, for a form to test before it saves.

        An unsafe URL is refused with the same 400 a create gets, since this is
        the form that would go on to store it.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        token: str | None = None
        if request.authorization_token is not None:
            token = request.authorization_token or None
        elif request.server_id is not None:
            server = await self._get_or_404(resolved_workspace_id, request.server_id)
            try:
                token = _stored_token(server)
            except (SecretDecryptionError, SecretBoxUnavailableError):
                check = ServerCheck(tool_names=[], failure=CheckFailure.CREDENTIALS_UNAVAILABLE)
                return _health(request.allowed_tools, check)
        await _validate_url(request.url, has_token=token is not None)
        # Ended before waiting on a remote server, as `_record_check` does.
        await self.db.commit()
        try:
            check = await _connect(request.url, token, server_id=request.server_id)
        except McpCapacityUnavailable:
            raise WorkspaceMcpServerCheckBusyError from None
        logger.info("MCP server connection tested workspace_id=%s outcome=%s", workspace_id, check.failure or "ok")
        return _health(request.allowed_tools, check)

    async def check_server(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID
    ) -> WorkspaceMcpServerPublic:
        """Check a server's connection now and store the outcome as its health.

        A failing check is a result, not an error: the request succeeded in
        finding out that the server did not.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        if not await self._record_check(server):
            raise WorkspaceMcpServerCheckBusyError
        return WorkspaceMcpServerPublic.from_model(server)

    async def delete_server(self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID) -> None:
        """Delete a server and the token stored with it."""
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        await self.db.delete(server)
        await self.db.commit()


__all__ = [
    "MAX_MCP_SERVERS_PER_WORKSPACE",
    "WorkspaceMcpServerCheckRequest",
    "WorkspaceMcpServerCreate",
    "WorkspaceMcpServerHealth",
    "WorkspaceMcpServerPublic",
    "WorkspaceMcpServerService",
    "WorkspaceMcpServerUpdate",
    "WorkspaceMcpServers",
    "WorkspaceMcpServersPublic",
    "resolve_workspace_mcp_server",
    "resolve_workspace_mcp_servers",
]
