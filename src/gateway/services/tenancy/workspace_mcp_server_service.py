"""Workspace-scoped MCP server configuration: CRUD, and the request-path resolve.

Ported from otari-ai's `MCPServerService` (otari#658). The platform stores an
MCP server per workspace, encrypts its bearer token on save, decrypts it when
the gateway asks for it by id, and never returns it in a public response. This
module is the same thing against the local database, for a standalone
deployment that has no platform to ask.

**Where it plugs in.** A request names stored servers with `mcp_server_ids`.
Hybrid mode resolves those through the platform
(`api/routes/_platform._resolve_platform_mcp_servers`); standalone mode
resolves them here, through :func:`resolve_workspace_mcp_servers`, called at
admission in `prepare_gateway_tools` where the request's session is live and
`RequestContext.workspace_id` already names the workspace its key belongs to.
That is the seam otari#655 settled and otari#678 wrote down; MCP is the
exception that decision names, because there is no deployment-wide server list
for a workspace row to narrow, so a row here is the only source of the
configuration rather than a narrowing of one.

**Deliberate divergences from the platform's version**, all of them because a
standalone deployment is not a hosted multi-tenant one:

- Authorization is this repository's own workspace check
  (`services.tenancy.authorization`), so a workspace owner/admin qualifies and
  not only an organization owner/admin. The platform's stricter *shape* is
  kept: every operation including the list read is management-gated, because
  these rows decide which external endpoints the gateway connects to and
  authenticates against on the workspace's behalf.
- URL safety is one check in this layer rather than the platform's split
  between a synchronous ingress validator and a service-layer SSRF pass. This
  repository's `validate_mcp_url` is a single async function that already does
  both, and these routes are async, so the reason for the split does not apply.
- No analytics events, and no `status` field on the public shape (the
  platform's is an affordance of its hosted dashboard; the page here,
  `web/src/features/tools/McpServersPage.tsx`, reports `enabled` and
  `has_token` and needs no third state).
"""

from __future__ import annotations

import uuid
from typing import Annotated

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import WorkspaceMcpServer
from gateway.models.mcp import McpServerConfig
from gateway.models.tenancy import User
from gateway.repositories.tenancy import WorkspaceRepository
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tenancy import authorization
from gateway.services.tenancy.errors import (
    SecretBoxUnavailableTenancyError,
    WorkspaceMcpServerAlreadyExistsError,
    WorkspaceMcpServerLimitReachedError,
    WorkspaceMcpServerNotFoundError,
    WorkspaceMcpServerUnsafeUrlError,
)
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url

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
    `entities.ProviderCredential` and `OrgProviderKey` already use.
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
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, server: WorkspaceMcpServer) -> WorkspaceMcpServerPublic:
        return cls(
            id=server.id,
            workspace_id=server.workspace_id,
            name=server.name,
            url=server.url,
            purpose_hint=server.purpose_hint,
            allowed_tools=server.allowed_tools,
            enabled=server.enabled,
            has_token=server.encrypted_token is not None,
            created_at=server.created_at.isoformat(),
            updated_at=server.updated_at.isoformat(),
        )


class WorkspaceMcpServersPublic(BaseModel):
    data: list[WorkspaceMcpServerPublic]
    count: int


async def _validate_url(url: str, *, has_token: bool) -> None:
    """Run the same SSRF/TLS check a request-body MCP server faces, at write time.

    Checked here as well as on the request path (`_validate_mcp_server_urls`)
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


async def resolve_workspace_mcp_servers(
    db: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    server_ids: list[uuid.UUID],
) -> list[McpServerConfig]:
    """Swap a request's ``mcp_server_ids`` for the workspace's stored configs.

    The standalone counterpart of `_platform._resolve_platform_mcp_servers`,
    and deliberately the same contract: ids are de-duplicated with their order
    preserved, an id naming no server *in this workspace* raises
    :class:`WorkspaceMcpServerNotFoundError` (the platform answers 404 for the
    same case, so the two modes refuse identically), and a disabled server is
    skipped rather than refused, so one decommissioned server does not break a
    caller whose stored id list still names it.

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


class WorkspaceMcpServerService:
    """CRUD for a workspace's MCP servers. Writes are management-gated; the list is not."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.organizations = OrganizationService(db)

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

    async def list_servers(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> WorkspaceMcpServersPublic:
        """List a page of the workspace's servers, plus the total.

        Reachable by any member who can see the workspace, like the workspace
        surfaces beside it (`workspace_budget_default_service` is the pattern);
        see `_require_management` for why the gate here is visibility alone.
        The rows never carry a token either way.
        """
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        workspace_id = workspace.id
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
            data=[WorkspaceMcpServerPublic.from_model(server) for server in servers],
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

        # "At most N rows for this workspace" spans a variable set of rows, so no
        # single unique index can hold it and the count below would otherwise be
        # a read that a concurrent create invalidates before this one inserts.
        # Same lock, for the same read-decide-write reason, that
        # `workspace_budget_default_service` and `org_provider_key_service` take.
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
        await self.db.refresh(server)
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

        if new_token == "":
            server.encrypted_token = None
        elif isinstance(new_token, str):
            server.encrypted_token = _encrypted(new_token)

        for field, value in fields.items():
            setattr(server, field, value)

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
        await self.db.refresh(server)
        return WorkspaceMcpServerPublic.from_model(server)

    async def delete_server(self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID) -> None:
        """Delete a server and the token stored with it."""
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        await self.db.delete(server)
        await self.db.commit()


__all__ = [
    "MAX_MCP_SERVERS_PER_WORKSPACE",
    "WorkspaceMcpServerCreate",
    "WorkspaceMcpServerPublic",
    "WorkspaceMcpServerService",
    "WorkspaceMcpServerUpdate",
    "WorkspaceMcpServersPublic",
    "resolve_workspace_mcp_servers",
]
