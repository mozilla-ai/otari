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
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.exceptions.tools_exceptions import (
    WorkspaceMcpServerAlreadyExistsError,
    WorkspaceMcpServerCheckBusyError,
    WorkspaceMcpServerGrantNotFoundError,
    WorkspaceMcpServerGrantSubjectError,
    WorkspaceMcpServerLimitReachedError,
    WorkspaceMcpServerNotFoundError,
    WorkspaceMcpServerUnsafeUrlError,
)
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig, ResolvedMcpServer
from gateway.models.tenancy import User
from gateway.models.tools import WorkspaceMcpServer, WorkspaceMcpServerGrant
from gateway.repositories.tenancy import WorkspaceRepository
from gateway.repositories.tools import WorkspaceMcpServerGrantRepository

# The module as well as the names, so the check is looked up at call time and a
# test can stand in for the network.
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import CheckFailure, McpCapacityUnavailable, ServerCheck
from gateway.services.model_access import is_mcp_tool_allowed, validate_mcp_tool_patterns
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tenancy import authorization
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.url_safety import UnsafeURLError, redact_url_secrets, validate_mcp_url

if TYPE_CHECKING:
    from gateway.services.api_keys import ApiKeyService, KeySummary
    from gateway.services.model_access import McpAccess

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
    everyone_tools: list[str] = Field(
        default_factory=lambda: ["*"],
        description="What every key of the workspace may call on this server: tool names, '*' for every tool, or "
        "'prefix*'; [] denies everyone. A grant for a key or its owner overrides it",
    )

    @model_validator(mode="after")
    def _check_everyone_tools(self) -> WorkspaceMcpServerCreate:
        self.everyone_tools = validate_mcp_tool_patterns(self.everyone_tools)
        return self


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
    everyone_tools: list[str] | SkipJsonSchema[None] = None

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
            for field in ("name", "url", "enabled", "everyone_tools")
            if field in self.model_fields_set and getattr(self, field) is None
        ]
        if nulled:
            raise ValueError(f"{', '.join(nulled)} cannot be null; omit the field to leave it unchanged")
        if self.everyone_tools is not None:
            self.everyone_tools = validate_mcp_tool_patterns(self.everyone_tools)
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
    everyone_tools: list[str] = Field(description="What every key of the workspace may call; [] denies everyone")
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
            everyone_tools=server.everyone_tools,
            has_token=server.encrypted_token is not None,
            health=WorkspaceMcpServerHealth.model_validate(server.health) if server.health is not None else None,
            created_at=server.created_at.isoformat(),
            updated_at=server.updated_at.isoformat(),
        )


class WorkspaceMcpServersPublic(BaseModel):
    data: list[WorkspaceMcpServerPublic]
    count: int


MAX_GRANTS_PER_SERVER = 1000
"""Grants one server may hold. Bounds the access list a page reads, and what a request's resolve scans."""

McpAccessDecider = Literal["key", "user", "everyone", "server"]


class WorkspaceMcpServerGrantPublic(BaseModel):
    """One exception to a server's ``everyone_tools``: for a key, or for a user and so every key they own."""

    id: uuid.UUID
    subject: Literal["key", "user"]
    api_key_id: str | None = Field(description="The key a key grant names; null on a user's grant")
    key_name: str | None = None
    key_active: bool | None = Field(default=None, description="Whether a key grant's key is active")
    user_id: str | None = Field(description="The user a grant names, or the owner of a key grant's key")
    tools: list[str] = Field(description="Tool patterns: names, '*' or 'prefix*'; [] denies")
    created_at: str
    updated_at: str


class WorkspaceMcpServerAccess(BaseModel):
    """A stored server's access list: what everyone may call, and the exceptions for keys and users.

    A key's own grant decides over its owner's, and either decides over ``everyone_tools``. The
    server's ``allowed_tools`` is a ceiling over all of them.
    """

    everyone_tools: list[str]
    grants: list[WorkspaceMcpServerGrantPublic]


class WorkspaceMcpServerGrantsCreate(BaseModel):
    """Grant the same tools to several keys and users at once; a subject that already has a grant gets these tools."""

    api_key_ids: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(default_factory=list, max_length=500)
    user_ids: list[Annotated[str, Field(min_length=1, max_length=255)]] = Field(default_factory=list, max_length=500)
    tools: list[str] = Field(description="Tool patterns: names, '*' or 'prefix*'; [] denies")

    @model_validator(mode="after")
    def _check(self) -> WorkspaceMcpServerGrantsCreate:
        if not self.api_key_ids and not self.user_ids:
            raise ValueError("name at least one key in api_key_ids or one user in user_ids")
        self.api_key_ids = list(dict.fromkeys(self.api_key_ids))
        self.user_ids = list(dict.fromkeys(self.user_ids))
        self.tools = validate_mcp_tool_patterns(self.tools)
        return self


class WorkspaceMcpServerGrantUpdate(BaseModel):
    """New tools for one grant."""

    tools: list[str] = Field(description="Tool patterns: names, '*' or 'prefix*'; [] denies")

    @model_validator(mode="after")
    def _check(self) -> WorkspaceMcpServerGrantUpdate:
        self.tools = validate_mcp_tool_patterns(self.tools)
        return self


class WorkspaceMcpServerAccessCheck(BaseModel):
    """Whether one key may call one tool on a server, and which rule says so."""

    allowed: bool
    decided_by: McpAccessDecider = Field(
        description="The key's own grant, its owner's, the server's everyone_tools, or the server itself "
        "(disabled, or its allowed_tools leaves the tool out)"
    )
    grant_id: uuid.UUID | None = Field(description="The deciding grant, when a grant decided")
    tools: list[str] = Field(
        description="The deciding rule's tool patterns; the server's allowed_tools when it decided"
    )


class WorkspaceMcpServerKeyAccess(BaseModel):
    """What one key reaches on one stored server, and which rule decides it."""

    server_id: uuid.UUID
    name: str
    enabled: bool
    decided_by: Literal["key", "user", "everyone"]
    grant_id: uuid.UUID | None
    tools: list[str] = Field(description="The deciding rule's tool patterns; [] means the key cannot reach the server")
    allowed_tools: list[str] | None = Field(description="The server's own ceiling; null exposes every tool")


class McpServerSummary(BaseModel):
    """A stored server as a caller sees it: enough to name it in ``mcp_server_ids``, never its URL or token."""

    id: uuid.UUID
    name: str
    purpose_hint: str | None


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
    access: McpAccess,
) -> list[McpServerConfig]:
    """Swap a request's ``mcp_server_ids`` for the workspace's stored configs.

    IDs are de-duplicated with their order preserved.
    An ID naming no server *in this workspace* raises :class:`WorkspaceMcpServerNotFoundError`.
    So does one the caller's ``access`` grants nothing on, which answers exactly as an unknown ID does,
    and each returned config carries the tool patterns the caller holds on it.
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

    patterns = {row.id: access.patterns_for(row.id, row.everyone_tools) for row in rows}
    missing = [server_id for server_id in ordered_ids if server_id not in by_id or not patterns[server_id]]
    if missing:
        raise WorkspaceMcpServerNotFoundError(missing[0])

    resolved: list[McpServerConfig] = []
    for server_id in ordered_ids:
        row = by_id[server_id]
        if not row.enabled:
            continue
        granted = patterns[server_id]
        resolved.append(
            McpServerConfig(
                name=row.name,
                url=row.url,
                authorization_token=decrypt_secret(row.encrypted_token) if row.encrypted_token else None,
                purpose_hint=row.purpose_hint,
                allowed_tools=row.allowed_tools,
                caller_tool_patterns=None if "*" in granted else granted,
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
        everyone_tools=row.everyone_tools,
        purpose_hint=row.purpose_hint,
        allowed_tools=row.allowed_tools,
    )


async def list_reachable_mcp_servers(
    db: AsyncSession, *, workspace_id: uuid.UUID, access: McpAccess
) -> list[McpServerSummary]:
    """The enabled stored servers of a workspace the caller may reach, oldest first."""
    rows = (
        (
            await db.execute(
                select(WorkspaceMcpServer)
                .where(WorkspaceMcpServer.workspace_id == workspace_id, WorkspaceMcpServer.enabled.is_(True))
                .order_by(WorkspaceMcpServer.created_at, WorkspaceMcpServer.id)
            )
        )
        .scalars()
        .all()
    )
    return [
        McpServerSummary(id=row.id, name=row.name, purpose_hint=row.purpose_hint)
        for row in rows
        if access.patterns_for(row.id, row.everyone_tools)
    ]


def _decide(
    key: KeySummary, server: WorkspaceMcpServer, grants: list[WorkspaceMcpServerGrant]
) -> tuple[Literal["key", "user", "everyone"], uuid.UUID | None, list[str]]:
    """The rule that decides a key's access to a server, among that server's grants: the same order a request uses."""
    for grant in grants:
        if grant.api_key_id == key.id:
            return "key", grant.id, list(grant.tools)
    for grant in grants:
        if key.user_id is not None and grant.user_id == key.user_id:
            return "user", grant.id, list(grant.tools)
    return "everyone", None, list(server.everyone_tools)


class WorkspaceMcpServers:
    """The MCP servers a workspace may reach, as a request reads them.

    A caller connects to a returned server directly; nothing here proxies MCP traffic.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def resolve_many(
        self, workspace_id: uuid.UUID, server_ids: list[uuid.UUID], access: McpAccess
    ) -> list[McpServerConfig]:
        """See :func:`resolve_workspace_mcp_servers`."""
        return await resolve_workspace_mcp_servers(
            self._db, workspace_id=workspace_id, server_ids=server_ids, access=access
        )

    async def list_reachable(self, workspace_id: uuid.UUID, access: McpAccess) -> list[McpServerSummary]:
        """See :func:`list_reachable_mcp_servers`."""
        return await list_reachable_mcp_servers(self._db, workspace_id=workspace_id, access=access)

    async def resolve_one(self, workspace_id: uuid.UUID, server_id: uuid.UUID) -> ResolvedMcpServer | None:
        """See :func:`resolve_workspace_mcp_server`."""
        return await resolve_workspace_mcp_server(self._db, workspace_id=workspace_id, server_id=server_id)


class WorkspaceMcpServerService:
    """CRUD for a workspace's MCP servers. Writes are management-gated; the list is not."""

    def __init__(self, db: AsyncSession, api_keys: ApiKeyService | None = None):
        self.db = db
        self.organizations = OrganizationService(db, membership_listener=None)
        # Reads the workspace's keys, which a server's access list names.
        self.api_keys = api_keys

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
            everyone_tools=request.everyone_tools,
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

    def _key_service(self) -> ApiKeyService:
        if self.api_keys is None:
            raise RuntimeError("WorkspaceMcpServerService was built without the key service its access list needs")
        return self.api_keys

    async def get_server(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID
    ) -> WorkspaceMcpServerPublic:
        """One server, for any member who can see the workspace; its URL's credentials masked for a non-manager."""
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        manages = await authorization.has_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )
        server = await self._get_or_404(workspace.id, server_id)
        return WorkspaceMcpServerPublic.from_model(server, redact_url=not manages)

    async def access(self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID) -> WorkspaceMcpServerAccess:
        """A server's access list: ``everyone_tools`` and every grant, oldest first."""
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        return await self._access_of(resolved_workspace_id, server)

    async def _access_of(self, workspace_id: uuid.UUID, server: WorkspaceMcpServer) -> WorkspaceMcpServerAccess:
        everyone_tools = list(server.everyone_tools)
        grants = await WorkspaceMcpServerGrantRepository(self.db).for_server(server.id)
        key_ids = [grant.api_key_id for grant in grants if grant.api_key_id is not None]
        keys = (
            {key.id: key for key in await self._key_service().summaries_in_workspace(workspace_id, key_ids)}
            if key_ids
            else {}
        )
        public: list[WorkspaceMcpServerGrantPublic] = []
        for grant in grants:
            key = keys.get(grant.api_key_id) if grant.api_key_id is not None else None
            public.append(
                WorkspaceMcpServerGrantPublic(
                    id=grant.id,
                    subject="key" if grant.api_key_id is not None else "user",
                    api_key_id=grant.api_key_id,
                    key_name=key.name if key is not None else None,
                    key_active=key.is_active if key is not None else None,
                    user_id=grant.user_id if grant.user_id is not None else (key.user_id if key is not None else None),
                    tools=list(grant.tools),
                    created_at=grant.created_at.isoformat(),
                    updated_at=grant.updated_at.isoformat(),
                )
            )
        return WorkspaceMcpServerAccess(everyone_tools=everyone_tools, grants=public)

    async def add_grants(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, request: WorkspaceMcpServerGrantsCreate
    ) -> WorkspaceMcpServerAccess:
        """Give keys and users the same tools on a server, replacing what any of them held.

        A key must belong to the server's workspace, and a user must own a key in it.

        Raises:
            WorkspaceMcpServerGrantSubjectError: a key or user is not of this workspace, or the server
                would hold more than ``MAX_GRANTS_PER_SERVER`` grants.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server_id = (await self._get_or_404(resolved_workspace_id, server_id)).id
        keys = self._key_service()
        if request.api_key_ids:
            found = {key.id for key in await keys.summaries_in_workspace(resolved_workspace_id, request.api_key_ids)}
            unknown = [key_id for key_id in request.api_key_ids if key_id not in found]
            if unknown:
                raise WorkspaceMcpServerGrantSubjectError(f"Key '{unknown[0]}' is not a key of this workspace")
        if request.user_ids:
            owners = await keys.owners_in_workspace(resolved_workspace_id, request.user_ids)
            strangers = [user_id for user_id in request.user_ids if user_id not in owners]
            if strangers:
                raise WorkspaceMcpServerGrantSubjectError(
                    f"'{strangers[0]}' owns no key in this workspace, so a grant to them would reach nothing"
                )
        repository = WorkspaceMcpServerGrantRepository(self.db)
        # The key reads above end in commits; the server is read again here, locked, so two
        # concurrent adds cannot both pass the cap.
        server = await self.db.get(WorkspaceMcpServer, server_id, with_for_update=True)
        if server is None:
            raise WorkspaceMcpServerNotFoundError(server_id)
        held = {(grant.api_key_id, grant.user_id) for grant in await repository.for_server(server.id)}
        added = sum(1 for key_id in request.api_key_ids if (key_id, None) not in held) + sum(
            1 for user_id in request.user_ids if (None, user_id) not in held
        )
        if len(held) + added > MAX_GRANTS_PER_SERVER:
            raise WorkspaceMcpServerGrantSubjectError(
                f"A server holds at most {MAX_GRANTS_PER_SERVER} grants; grant a user instead of each of their keys"
            )
        await repository.put(server.id, api_key_ids=request.api_key_ids, user_ids=request.user_ids, tools=request.tools)
        await self.db.commit()
        logger.info(
            "MCP server grants added server_id=%s keys=%d users=%d",
            server.id,
            len(request.api_key_ids),
            len(request.user_ids),
        )
        return await self._access_of(resolved_workspace_id, server)

    async def update_grant(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        server_id: uuid.UUID,
        grant_id: uuid.UUID,
        request: WorkspaceMcpServerGrantUpdate,
    ) -> WorkspaceMcpServerAccess:
        """Change the tools of one grant."""
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        repository = WorkspaceMcpServerGrantRepository(self.db)
        grant = await repository.get_on_server(server.id, grant_id)
        if grant is None:
            raise WorkspaceMcpServerGrantNotFoundError(grant_id)
        await repository.set_tools(grant, request.tools)
        await self.db.commit()
        return await self._access_of(resolved_workspace_id, server)

    async def delete_grant(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, grant_id: uuid.UUID
    ) -> WorkspaceMcpServerAccess:
        """Remove one grant: its key or user falls back to the next rule."""
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        repository = WorkspaceMcpServerGrantRepository(self.db)
        grant = await repository.get_on_server(server.id, grant_id)
        if grant is None:
            raise WorkspaceMcpServerGrantNotFoundError(grant_id)
        await repository.remove(grant)
        await self.db.commit()
        return await self._access_of(resolved_workspace_id, server)

    async def check_access(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, api_key_id: str, tool: str
    ) -> WorkspaceMcpServerAccessCheck:
        """Whether a key of the workspace may call ``tool`` on a server, and which rule decides.

        Raises:
            WorkspaceMcpServerGrantSubjectError: the key is not of this workspace.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        decided, grant_id, patterns = await self._decide_for_key(resolved_workspace_id, api_key_id, server)
        if not server.enabled:
            return WorkspaceMcpServerAccessCheck(allowed=False, decided_by="server", grant_id=None, tools=[])
        if server.allowed_tools is not None and tool not in server.allowed_tools:
            return WorkspaceMcpServerAccessCheck(
                allowed=False, decided_by="server", grant_id=None, tools=list(server.allowed_tools)
            )
        return WorkspaceMcpServerAccessCheck(
            allowed=is_mcp_tool_allowed(patterns, tool), decided_by=decided, grant_id=grant_id, tools=patterns
        )

    async def _decide_for_key(
        self, workspace_id: uuid.UUID, api_key_id: str, server: WorkspaceMcpServer
    ) -> tuple[Literal["key", "user", "everyone"], uuid.UUID | None, list[str]]:
        """Which rule decides a key's access to a server: its grant, its owner's, or ``everyone_tools``."""
        found = await self._key_service().summaries_in_workspace(workspace_id, [api_key_id])
        if not found:
            raise WorkspaceMcpServerGrantSubjectError(f"Key '{api_key_id}' is not a key of this workspace")
        return _decide(found[0], server, await WorkspaceMcpServerGrantRepository(self.db).for_server(server.id))

    async def key_access(
        self, *, user: User, workspace_id: uuid.UUID, api_key_id: str
    ) -> list[WorkspaceMcpServerKeyAccess]:
        """What one key of the workspace reaches on each of its stored servers, oldest server first.

        Raises:
            WorkspaceMcpServerGrantSubjectError: the key is not of this workspace.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        found = await self._key_service().summaries_in_workspace(resolved_workspace_id, [api_key_id])
        if not found:
            raise WorkspaceMcpServerGrantSubjectError(f"Key '{api_key_id}' is not a key of this workspace")
        key = found[0]
        servers = (
            (
                await self.db.execute(
                    select(WorkspaceMcpServer)
                    .where(WorkspaceMcpServer.workspace_id == resolved_workspace_id)
                    .order_by(WorkspaceMcpServer.created_at, WorkspaceMcpServer.id)
                )
            )
            .scalars()
            .all()
        )
        by_server: dict[uuid.UUID, list[WorkspaceMcpServerGrant]] = {}
        for grant in await WorkspaceMcpServerGrantRepository(self.db).for_caller(key.id, key.user_id):
            by_server.setdefault(grant.server_id, []).append(grant)
        result: list[WorkspaceMcpServerKeyAccess] = []
        for server in servers:
            decided, grant_id, patterns = _decide(key, server, by_server.get(server.id, []))
            result.append(
                WorkspaceMcpServerKeyAccess(
                    server_id=server.id,
                    name=server.name,
                    enabled=server.enabled,
                    decided_by=decided,
                    grant_id=grant_id,
                    tools=patterns,
                    allowed_tools=server.allowed_tools,
                )
            )
        return result

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
    "MAX_GRANTS_PER_SERVER",
    "McpServerSummary",
    "WorkspaceMcpServerAccess",
    "WorkspaceMcpServerAccessCheck",
    "WorkspaceMcpServerGrantPublic",
    "WorkspaceMcpServerGrantUpdate",
    "WorkspaceMcpServerGrantsCreate",
    "WorkspaceMcpServerKeyAccess",
    "WorkspaceMcpServerCheckRequest",
    "WorkspaceMcpServerCreate",
    "WorkspaceMcpServerHealth",
    "WorkspaceMcpServerPublic",
    "WorkspaceMcpServerService",
    "WorkspaceMcpServerUpdate",
    "WorkspaceMcpServers",
    "WorkspaceMcpServersPublic",
    "list_reachable_mcp_servers",
    "resolve_workspace_mcp_server",
    "resolve_workspace_mcp_servers",
]
