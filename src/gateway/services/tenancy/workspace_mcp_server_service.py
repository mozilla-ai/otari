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

import asyncio
import dataclasses
import uuid
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Annotated, Literal
from urllib.parse import urlsplit

from fastapi import BackgroundTasks
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions import TenancyError
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.exceptions.tools_exceptions import (
    McpCheckSignInRequiredError,
    McpConnectionFlowError,
    McpConnectionSetupError,
    McpConnectionUnavailableError,
    McpSignInRequiredError,
    WorkspaceMcpServerAccessDeniedError,
    WorkspaceMcpServerAlreadyExistsError,
    WorkspaceMcpServerCheckBusyError,
    WorkspaceMcpServerCheckOriginError,
    WorkspaceMcpServerGrantNotFoundError,
    WorkspaceMcpServerGrantSubjectError,
    WorkspaceMcpServerKeyNotFoundError,
    WorkspaceMcpServerLimitReachedError,
    WorkspaceMcpServerNotFoundError,
    WorkspaceMcpServerUnsafeUrlError,
)
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig, ResolvedMcpServer
from gateway.models.tenancy import MANAGEMENT_ROLES, User
from gateway.models.tools import (
    McpCredentialMode,
    McpSharedConnection,
    WorkspaceMcpServer,
    WorkspaceMcpServerGrant,
)
from gateway.ports.growth_signal_port import GrowthAction, GrowthSignalPort
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.tools import (
    McpOAuthFlowRepository,
    McpUserConnectionRepository,
    WorkspaceMcpServerGrantRepository,
    WorkspaceMcpServerRepository,
)
from gateway.repositories.users_repository import lock_live_users, provision_attribution_users
from gateway.schemas.tools import (
    McpConnectionCompleted,
    McpConnectionPublic,
    McpConnectionsPublic,
    McpConnectionStarted,
    McpServerConnectionPublic,
    McpServerConnectionsPublic,
)

# The module as well as the names, so the check is looked up at call time and a
# test can stand in for the network.
from gateway.services import mcp_stateless
from gateway.services.mcp_client import ATTACHED_CONNECT_BUDGET_S
from gateway.services.mcp_stateless import CheckFailure, McpCapacityUnavailable, ServerCheck
from gateway.services.model_access import (
    MCP_READ_PRESET,
    expand_mcp_tool_patterns,
    is_mcp_tool_allowed,
    mcp_read_tools,
    validate_mcp_read_tools,
    validate_mcp_tool_patterns,
)
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tenancy import authorization
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.tools import (
    CompletedConnection,
    McpConnectionService,
    McpRevocation,
    McpServerRef,
    McpServerRequestService,
    McpSignInNeeded,
    McpSignInReason,
    McpUserTokens,
    SignInLinkTarget,
    StoredMcpOAuthClient,
    WorkspaceMcpServerHealth,
    mcp_server_contacts,
)
from gateway.services.url_safety import UnsafeURLError, redact_url_secrets, validate_mcp_url

MAX_ALLOWED_TOOLS = WorkspaceMcpServerHealth.MAX_ALLOWED_TOOLS
MAX_TOOL_NAME_LENGTH = WorkspaceMcpServerHealth.MAX_TOOL_NAME_LENGTH

if TYPE_CHECKING:
    from gateway.services.api_keys import ApiKeyService, KeySummary
    from gateway.services.model_access import McpAccess

# What a workspace may configure. A resolved request opens a session to every
# server it names, so this bounds the fan-out one workspace can ask a gateway
# process for. Same value the platform uses: generous for any real setup, low
# enough that the list is not a resource in itself.
MAX_MCP_SERVERS_PER_WORKSPACE = 50

MAX_OAUTH_CLIENT_ID_LENGTH = 512

_MAX_LIST_LIMIT = 1000

_PER_USER_TAKES_NO_TOKEN = "a per_user server takes no authorization_token: each user signs in with their own account"

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
        "'prefix*', or '@read' for read_tools; [] denies everyone. A grant for a key or its owner overrides it",
    )
    read_tools: list[str] | None = Field(
        default=None,
        description="The tool names a manager labeled read, which '@read' in an access rule stands for. "
        "Omit it, or send null, and the labels follow the tools the server itself marks readOnlyHint",
    )
    credential_mode: McpCredentialMode = Field(
        default=McpCredentialMode.SHARED,
        description="Whose credential a call carries: 'shared' sends one credential for every caller, "
        "authorization_token or the account a manager signs in to for everyone once the server is stored; "
        "'per_user' sends the calling user's own token, which each user gets by signing in to the server, "
        "and takes no authorization_token",
    )
    oauth_client_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_OAUTH_CLIENT_ID_LENGTH,
        description="For a server whose authorization server does not register clients on its own: "
        "the OAuth client ID a sign-in goes through. Omit it to let Otari register itself",
    )
    oauth_client_secret: str | None = Field(
        default=None,
        min_length=1,
        max_length=8192,
        description="The secret of oauth_client_id, when it has one. Encrypted at rest, never returned",
    )
    attach_by_default: bool = Field(
        default=False,
        description="Add this server to every model request from the workspace's keys, as if the request named it "
        "in mcp_server_ids. A key the server's access rules grant nothing is left without it",
    )

    @model_validator(mode="after")
    def _check_tool_patterns(self) -> WorkspaceMcpServerCreate:
        self.everyone_tools = validate_mcp_tool_patterns(self.everyone_tools)
        if self.read_tools is not None:
            self.read_tools = validate_mcp_read_tools(self.read_tools)
        if self.credential_mode == McpCredentialMode.PER_USER and self.authorization_token:
            raise ValueError(_PER_USER_TAKES_NO_TOKEN)
        if self.oauth_client_secret and not self.oauth_client_id:
            raise ValueError("oauth_client_secret needs oauth_client_id")
        return self


class WorkspaceMcpServerUpdate(BaseModel):
    """Partial update. Only the fields the caller sets are applied.

    ``authorization_token`` has three states rather than two, which is what a
    write-only field needs: omit it to leave the stored token alone, send
    ``""`` to clear it, send a value to rotate it. An explicit ``null`` also
    leaves it alone, matching the platform: a client that serializes its whole
    form back, with an empty token box it never filled in, must not destroy a
    token it was never shown. ``oauth_client_secret`` follows the same three
    states, and so does ``oauth_client_id``, where ``""`` goes back to letting
    Otari register itself.

    Changing ``credential_mode``, the URL, or the OAuth client ends every
    sign-in to the server, each user's own and the shared account: the tokens
    were issued for the server and the client as they were. Setting a token on
    a shared server ends its shared account too, since a server sends one
    credential.
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
    read_tools: list[str] | None = Field(
        default=None,
        description="The tool names a manager labeled read. An explicit null goes back to following the tools "
        "the server marks readOnlyHint; omit it to leave the labels as they are",
    )
    credential_mode: McpCredentialMode | SkipJsonSchema[None] = None
    oauth_client_id: str | None = Field(default=None, max_length=MAX_OAUTH_CLIENT_ID_LENGTH)
    oauth_client_secret: str | None = Field(default=None, max_length=8192)
    attach_by_default: bool | SkipJsonSchema[None] = None

    @model_validator(mode="after")
    def _reject_explicit_nulls(self) -> WorkspaceMcpServerUpdate:
        """Refuse an explicit ``null`` for a column that cannot hold one.

        ``None`` is this schema's "not sent" marker, so a caller that means to
        clear a required field is expressing something the row cannot store.
        Caught here rather than at the flush: a ``NOT NULL`` violation arrives
        as the same ``IntegrityError`` the unique index raises, and the service
        would report a name collision that never happened.
        ``purpose_hint``, ``allowed_tools`` and ``read_tools`` are nullable and are
        deliberately clearable this way, and the token has its own three-state rule above.
        """
        nulled = [
            field
            for field in (
                "name",
                "url",
                "enabled",
                "everyone_tools",
                "credential_mode",
                "attach_by_default",
            )
            if field in self.model_fields_set and getattr(self, field) is None
        ]
        if nulled:
            raise ValueError(f"{', '.join(nulled)} cannot be null; omit the field to leave it unchanged")
        if self.everyone_tools is not None:
            self.everyone_tools = validate_mcp_tool_patterns(self.everyone_tools)
        if self.read_tools is not None:
            self.read_tools = validate_mcp_read_tools(self.read_tools)
        return self


class WorkspaceMcpServerCheckRequest(BaseModel):
    """Connection details to check without storing them, for a form to test before it saves.

    ``server_id`` names a stored server whose credential stands in when
    ``authorization_token`` is omitted, so an edit form can test a new path with
    a token it was never shown. The credential is only sent to that server's own
    origin. ``""`` checks with no token at all.
    """

    url: str = Field(min_length=1, max_length=2048, description="Streamable HTTP MCP endpoint")
    authorization_token: str | None = Field(
        default=None, max_length=8192, description="Bearer token; omit to use the stored server's, if any"
    )
    allowed_tools: list[Annotated[str, Field(max_length=MAX_TOOL_NAME_LENGTH)]] | None = Field(
        default=None, max_length=MAX_ALLOWED_TOOLS, description="The allow-list to report the catalog against"
    )
    server_id: uuid.UUID | None = Field(
        default=None, description="A stored server whose credential to use, when `url` is on that server's origin"
    )


McpCallerAccess = Literal["all", "read", "some", "none"]


class McpSharedAccountPublic(BaseModel):
    """The account a manager signed in to for everyone on a shared server. Never carries a token."""

    signed_in_by: str | None = Field(
        description="The name, or else the email, of the dashboard account that signed in; null once it is gone"
    )
    signed_in_at: str
    renewable: bool = Field(
        description="Whether Otari can renew the token on its own; false means signing in again once it expires"
    )


def _needs_setup(server: WorkspaceMcpServer, *, has_account: bool) -> bool:
    """Whether a shared server holds no credential and its server asked for one the last time it was reached.

    Read from the recorded health, so a server that takes no credential at all is never held back.
    """
    if server.credential_mode != McpCredentialMode.SHARED or server.encrypted_token is not None or has_account:
        return False
    return (server.health or {}).get("error") == CheckFailure.UNAUTHORIZED


def _caller_access(server: WorkspaceMcpServer, tools: Sequence[str]) -> McpCallerAccess:
    """What a rule's tool patterns come to on a server, in one word, under the server's own allow-list."""
    if not server.enabled or not tools or server.allowed_tools == []:
        return "none"
    if "*" in tools:
        return "all" if server.allowed_tools is None else "some"
    return "read" if list(tools) == [MCP_READ_PRESET] else "some"


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
    read_tools: list[str] = Field(
        description="The tool names '@read' stands for: a manager's labels, or until one edits them the tools "
        "the server marks readOnlyHint, as last listed"
    )
    read_tools_from_server: bool = Field(
        description="True while no manager has edited read_tools, so they follow the server's own readOnlyHint"
    )
    has_token: bool
    credential_mode: McpCredentialMode = Field(
        description="'shared': one workspace credential for every caller. 'per_user': each user's own connection"
    )
    shared_account: McpSharedAccountPublic | None = Field(
        default=None,
        description="On a shared server: the account a manager signed in to for everyone, which every call "
        "carries in place of a token; null when it has none",
    )
    connected_users: int | None = Field(
        default=None,
        description="On a per_user server, for a caller who may manage the workspace: how many users have "
        "connected their own account. Null on a shared server, and for a caller who may not manage",
    )
    needs_setup: bool = Field(
        default=False,
        description="A shared server with no token and no shared account, whose server asked for a credential "
        "when last reached. Requests skip it, as they skip a disabled server, until it has one",
    )
    oauth_client_id: str | None = Field(
        description="The OAuth client ID a manager supplied for sign-ins; null when Otari registers itself"
    )
    has_oauth_client_secret: bool
    attach_by_default: bool = Field(
        description="Whether every model request from the workspace's keys gets this server without naming it"
    )
    health: WorkspaceMcpServerHealth | None = Field(
        description="What the gateway last learned by talking to the server, from a check or from a real request; "
        "null until either has happened"
    )
    caller_access: McpCallerAccess | None = Field(
        default=None,
        description="On a read of one server or of the list: what the access rules give the caller's own user. "
        "'all' every tool, 'read' the read tools only, 'some' a named subset, 'none' nothing. "
        "A key with a rule of its own can differ",
    )
    created_at: str
    updated_at: str

    @classmethod
    def from_model(
        cls,
        server: WorkspaceMcpServer,
        *,
        redact_url: bool = False,
        caller_tools: Sequence[str] | None = None,
        shared_account: McpSharedAccountPublic | None = None,
        connected_users: int | None = None,
    ) -> WorkspaceMcpServerPublic:
        """Shape one row, optionally masking any credential embedded in its URL.

        ``redact_url`` is what keeps the member read from being a wider
        disclosure than the management read it was split from: the token column
        was never returned either way, but nothing rejects a URL that carries a
        credential in its userinfo (``https://user:secret@host``) or a query
        parameter, so a caller who may read the row without managing it gets
        those components masked. A manager sees the URL as stored, because the
        edit form restates it and would otherwise save the mask back.
        """
        client = McpConnectionService.stored_oauth_client(server)
        own_client = client if client is not None and not client.dynamic else None
        return cls(
            id=server.id,
            workspace_id=server.workspace_id,
            name=server.name,
            url=redact_url_secrets(server.url) if redact_url else server.url,
            purpose_hint=server.purpose_hint,
            allowed_tools=server.allowed_tools,
            enabled=server.enabled,
            everyone_tools=server.everyone_tools,
            read_tools=_read_tools(server),
            read_tools_from_server=server.read_tools is None,
            has_token=server.encrypted_token is not None,
            credential_mode=McpCredentialMode(server.credential_mode),
            shared_account=shared_account,
            connected_users=connected_users,
            needs_setup=_needs_setup(server, has_account=shared_account is not None),
            oauth_client_id=own_client.client_id if own_client is not None else None,
            has_oauth_client_secret=own_client is not None and own_client.encrypted_secret is not None,
            attach_by_default=server.attach_by_default,
            health=WorkspaceMcpServerHealth.model_validate(server.health) if server.health is not None else None,
            caller_access=None if caller_tools is None else _caller_access(server, caller_tools),
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
    tools: list[str] = Field(
        description="Tool patterns: names, '*', 'prefix*' or '@read' for the server's read_tools; [] denies"
    )
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
    tools: list[str] = Field(
        description="Tool patterns: names, '*', 'prefix*' or '@read' for the server's read_tools; [] denies"
    )

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

    tools: list[str] = Field(
        description="Tool patterns: names, '*', 'prefix*' or '@read' for the server's read_tools; [] denies"
    )

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
    read_tools: list[str] = Field(
        default_factory=list, description="The server's tools labeled read, which '@read' in tools stands for"
    )


class McpToolSummary(BaseModel):
    """One tool of a server: its name and, where the server described it, the first line of that description."""

    name: str
    description: str | None = Field(default=None, description="Untrusted text the server wrote; null when it has none")
    read_only: bool = Field(default=False, description="Whether the tool is one of the server's read tools")


class WorkspaceMcpServerMyTools(BaseModel):
    """The tools the caller may use on one server, under the access rules that apply to them.

    The rule is the caller's own: their user's grant, else the server's ``everyone_tools``. A key
    with a grant of its own can reach more or less than this.
    """

    server_id: uuid.UUID
    decided_by: Literal["user", "everyone", "server"] = Field(
        description="The caller's own grant, the server's everyone_tools, or the server itself when it is disabled"
    )
    access: Literal["all", "some", "none"] = Field(
        description="'all' when the rule admits every tool the server offers, 'none' when it admits nothing"
    )
    known: bool = Field(
        description="Whether the server's tool list is known: from its last passing check, or listed just now "
        "with the caller's own connection to a per-user server"
    )
    checked_at: str | None = Field(description="When the tool list was read; null when it is not known")
    tools: list[McpToolSummary] = Field(
        description="The tools the caller may use, in the server's order; empty when not known"
    )


class McpServerSummary(BaseModel):
    """A stored server as a caller sees it: enough to name it in ``mcp_server_ids``, never its URL or token."""

    id: uuid.UUID
    name: str
    purpose_hint: str | None
    sign_in_required: bool = Field(
        default=False,
        description="The server sends each user's own credential and this key's user has not signed in to it, "
        "so a call is refused with mcp_sign_in_required until they do",
    )
    sign_in_url: str | None = Field(
        default=None,
        description="With sign_in_required: a link that starts this key's user's own sign-in to the server, "
        "working for 30 minutes",
    )


def _own_client(client_id: str | None, client_secret: str | None) -> dict[str, object] | None:
    """A manager's own OAuth client in its stored shape, its secret encrypted; ``None`` when they named none."""
    if not client_id:
        return None
    return StoredMcpOAuthClient(
        client_id=client_id, encrypted_secret=_encrypted(client_secret) if client_secret else None
    ).model_dump()


def _edited_client(
    current: StoredMcpOAuthClient | None, client_id: object, client_secret: object, *, moves_host: bool = False
) -> tuple[dict[str, object] | None, bool, bool]:
    """A server's stored client after an edit: the client, whether to store it, and whether it is another one.

    Each of the two fields is left alone when omitted or null, cleared by ``""``, and set by a value.
    Clearing the ID goes back to the client Otari registers itself. A new secret on the same ID is
    the same client, so it keeps the users' connections: their grants name the client, not its secret.
    An edit that ``moves_host`` keeps no stored secret, even under the same ID: it was issued for the
    old host, and the new host's token endpoint would be sent it. Only a secret in the edit is kept.
    """
    stored = current if current is not None and not current.dynamic else None
    own = stored.model_copy(update={"encrypted_secret": None}) if moves_host and stored is not None else stored
    target_id = own.client_id if own is not None else None
    if isinstance(client_id, str):
        target_id = client_id or None
    if target_id is None:
        if isinstance(client_secret, str) and client_secret:
            raise McpConnectionSetupError("oauth_client_secret needs oauth_client_id")
        return None, stored is not None, stored is not None
    same_client = own is not None and own.client_id == target_id
    encrypted_secret = own.encrypted_secret if own is not None and same_client else None
    if isinstance(client_secret, str):
        encrypted_secret = _encrypted(client_secret) if client_secret else None
    edited = StoredMcpOAuthClient(client_id=target_id, encrypted_secret=encrypted_secret)
    return edited.model_dump(), edited != stored, not same_client


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> str:
    """A URL's scheme, host and port, as a client registered with one authorization server is bound to them.

    Compared as an origin rather than as text: the default port written out (``:443``), letter case
    and credentials in the URL name the same host, so none of them counts as a move.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    try:
        port = parts.port or _DEFAULT_PORTS.get(scheme)
    except ValueError:
        return f"{scheme}://{parts.netloc.lower()}"
    return f"{scheme}://{(parts.hostname or '').lower()}:{port}"


@dataclasses.dataclass(frozen=True, slots=True)
class McpSignInLinkPreview:
    """What the page a sign-in link opens says before anything starts."""

    server_name: str
    server_host: str
    # The user the sign-in will be attached to, by the most recognizable name it has.
    user_label: str
    # The dashboard account the link was opened in, when there is one.
    opened_as: str | None
    # That account is not the user the link was made for, so the sign-in is not offered.
    other_user: bool


class WorkspaceMcpConnectionEligibility:
    """Whether a user may still hold a connection to a workspace's servers, for ``McpConnectionService``.

    Runs inside the connection service's Unit of Work block, on the repositories it is given. It takes
    the workspace lock a member's removal takes before it reads the membership, and the user row
    lock a user's deletion takes, so a sign-in finishing and a removal running serialize: whichever
    commits second sees the other.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        workspaces: WorkspaceRepository,
        members: WorkspaceMemberRepository,
        organization_members: OrganizationMemberRepository,
        accounts: UserRepository,
    ) -> None:
        self._uow = uow
        self._workspaces = workspaces
        self._members = members
        self._organization_members = organization_members
        self._accounts = accounts

    async def hold(self, workspace_id: uuid.UUID) -> None:
        """Take the workspace lock a member's removal and the workspace's deletion take first."""
        await self._workspaces.lock(workspace_id)

    async def may_connect(self, workspace_id: uuid.UUID, user_id: str) -> bool:
        """A live user who, when they are a dashboard account, can still reach the workspace.

        A dashboard account reaches it as an active member of it, or as an owner or admin of its
        organization. A request-plane user that is no dashboard account belongs to no workspace,
        so only its deletion ends its connections. Locks the user's row as their deletion does.
        """
        if user_id not in await lock_live_users(self._uow, [user_id]):
            return False
        try:
            identity = uuid.UUID(user_id)
        except ValueError:
            return True
        if str(identity) != user_id:
            return True
        # A dashboard account is a request-plane user under its own id. A UUID-shaped id with no
        # account behind it is a request-plane user like any other.
        if not await self._accounts.get_by_ids([identity]):
            return True
        # Locked as read: an organization's removal of the person changes these rows without the
        # workspace lock, and must wait for this sign-in, or be seen by it.
        if await self._members.get_active_pairs([workspace_id], [identity], lock=True):
            return True
        workspace = await self._workspaces.get(workspace_id)
        if workspace is None:
            return False
        managers = await self._organization_members.active_with_roles(
            workspace.organization_id, {identity}, MANAGEMENT_ROLES, lock=True
        )
        return identity in managers

    async def may_manage(self, workspace_id: uuid.UUID, identity_id: uuid.UUID) -> bool:
        """An owner or admin of the workspace, or of its organization, as the management routes require."""
        membership = await self._members.get_active_by_workspace_and_user(workspace_id, identity_id)
        if membership is not None and membership.role in MANAGEMENT_ROLES:
            return True
        workspace = await self._workspaces.get(workspace_id)
        if workspace is None:
            return False
        managers = await self._organization_members.active_with_roles(
            workspace.organization_id, {identity_id}, MANAGEMENT_ROLES, lock=True
        )
        return identity_id in managers


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


def _read_tools(server: WorkspaceMcpServer) -> list[str]:
    """The tools ``@read`` stands for on a server: its labels, or the server's own until a manager sets them."""
    return mcp_read_tools(server.read_tools, server.health)


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


class _NoCredential:
    """Stands for a server's token that could not be read, where a token or ``None`` would go."""


_UNREADABLE = _NoCredential()


async def _check_credential(
    server: WorkspaceMcpServer, tokens: McpUserTokens, user_id: str
) -> str | None | _NoCredential:
    """The token a connection check sends: the workspace's, or on a per-user server the asker's own.

    Raises:
        McpSignInRequiredError: the server is per-user and the person asking has not signed in to it.
    """
    try:
        if server.credential_mode == McpCredentialMode.PER_USER:
            return await tokens.token_for(server, user_id)
        return _stored_token(server)
    except (SecretDecryptionError, SecretBoxUnavailableError, McpConnectionUnavailableError):
        return _UNREADABLE


async def _check(server: WorkspaceMcpServer, token: str | None | _NoCredential) -> ServerCheck:
    """Run the connection check on a stored row with the token settled for it, checking its URL first.

    The same order the request path takes: a token that will not decrypt and a
    URL that is no longer safe each stop the check before anything is sent.

    Raises:
        McpCapacityUnavailable: no check slot was free, so nothing was checked.
    """
    if isinstance(token, _NoCredential):
        return ServerCheck(tool_names=[], failure=CheckFailure.CREDENTIALS_UNAVAILABLE)
    try:
        await validate_mcp_url(server.url, has_authorization_token=token is not None)
    except UnsafeURLError:
        return ServerCheck(tool_names=[], failure=CheckFailure.UNSAFE_URL)
    return await _connect(server.url, token, server_id=server.id)


async def resolve_workspace_mcp_servers(
    db: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    server_ids: list[uuid.UUID],
    access: McpAccess,
    tokens: McpUserTokens | None = None,
) -> list[McpServerConfig]:
    """Swap a request's ``mcp_server_ids`` for the workspace's stored configs.

    IDs are de-duplicated with their order preserved.
    An ID naming no server *in this workspace* raises :class:`WorkspaceMcpServerNotFoundError`.
    So does one the caller's ``access`` grants nothing on, which answers exactly as an unknown ID does,
    and each returned config carries the tool patterns the caller holds on it.
    A disabled server is skipped rather than refused.

    A per-user server carries the token of ``access.user_id``, read through ``tokens``. A caller
    with no user cannot use one, which also answers as an unknown ID. One whose user has not
    signed in raises :class:`McpSignInRequiredError` naming the server, after the access rules
    admitted it, and one whose expired token cannot be renewed right now raises
    :class:`McpConnectionUnavailableError`.

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
    resolved = await _resolve_for_request(
        db, workspace_id=workspace_id, server_ids=server_ids, attach=False, access=access, tokens=tokens, defer=False
    )
    return resolved.servers


@dataclasses.dataclass(frozen=True, slots=True)
class RequestMcpServers:
    """The stored servers one model request reaches, and the per-user ones it reaches only through a sign-in."""

    servers: list[McpServerConfig] = dataclasses.field(default_factory=list)
    sign_in: list[McpSignInNeeded] = dataclasses.field(default_factory=list)


async def _resolve_for_request(
    db: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    server_ids: list[uuid.UUID],
    attach: bool,
    access: McpAccess,
    tokens: McpUserTokens | None,
    defer: bool,
) -> RequestMcpServers:
    if not server_ids and not attach:
        return RequestMcpServers()
    tokens = tokens or McpUserTokens(McpUserConnectionRepository(db))

    named = list(dict.fromkeys(server_ids))
    rows = await WorkspaceMcpServerRepository(db).for_request(workspace_id, named, attached=attach)
    by_id = {row.id: row for row in rows}

    patterns = {row.id: access.patterns_for(row.id, row.everyone_tools, _read_tools(row)) for row in rows}
    # Only a deferring caller is told a per-user server needs a user it does not have; the others answer 404.
    missing = [
        server_id
        for server_id in named
        if server_id not in by_id or not patterns[server_id] or (not defer and _needs_a_user(by_id[server_id], access))
    ]
    if missing:
        raise WorkspaceMcpServerNotFoundError(missing[0])
    attached = [
        row.id
        for row in rows
        if row.attach_by_default and row.id not in named and patterns[row.id] and not _needs_a_user(row, access)
    ]

    # Every row is read before any token is: reading one may release the session, and a release
    # whose commit fails rolls back, which expires the rows still to come.
    planned: list[tuple[McpServerConfig, McpServerRef | None, McpSignInNeeded]] = []
    # A shared server with no token: its account's token is read below, and with neither it is
    # skipped as a disabled one is when its server last asked for a credential.
    accounts: dict[uuid.UUID, tuple[McpServerRef, bool]] = {}
    for server_id in named + attached:
        row = by_id[server_id]
        if not row.enabled:
            continue
        granted = patterns[server_id]
        per_user = row.credential_mode == McpCredentialMode.PER_USER
        try:
            shared_token = None if per_user else _stored_token(row)
        except (SecretDecryptionError, SecretBoxUnavailableError):
            if server_id in named:
                raise
            _leave_out_unreadable(row.id)
            continue
        if not per_user and shared_token is None:
            accounts[row.id] = (McpServerRef.of(row), _needs_setup(row, has_account=False))
        config = McpServerConfig(
            name=row.name,
            url=row.url,
            authorization_token=shared_token,
            renew_token=_renewer(row, access, tokens),
            purpose_hint=row.purpose_hint,
            allowed_tools=row.allowed_tools,
            caller_tool_patterns=None if "*" in granted else granted,
            caller_access_source=f"mcp_{access.decide(row.id, row.everyone_tools)[0]}",
            stored_server_id=str(row.id),
            attached=server_id not in named,
        )
        need = McpSignInNeeded(
            server_id=row.id,
            name=row.name,
            purpose_hint=row.purpose_hint,
            user_id=access.user_id,
            reason=McpSignInReason.NOT_SIGNED_IN if access.user_id else McpSignInReason.NO_USER,
            attached=config.attached,
        )
        planned.append((config, McpServerRef.of(row) if per_user else None, need))

    resolved = RequestMcpServers()
    # The attached servers' renewals share the budget their connections get, from the first of
    # them: an expired token on several attached servers cannot hold a completion past it.
    loop = asyncio.get_running_loop()
    attached_by: float | None = None
    for config, ref, need in planned:
        bound: asyncio.Timeout | None = None
        if config.attached and (ref is not None or need.server_id in accounts):
            attached_by = attached_by or loop.time() + ATTACHED_CONNECT_BUDGET_S
            bound = asyncio.timeout_at(attached_by)
        if ref is not None:
            try:
                if access.user_id is None:
                    raise WorkspaceMcpServerNotFoundError(need.server_id)
                if bound is None:
                    config.authorization_token = await tokens.token_for(ref, access.user_id)
                else:
                    async with bound:
                        config.authorization_token = await tokens.token_for(ref, access.user_id)
            except TimeoutError:
                if bound is None or not bound.expired():
                    raise
                resolved.sign_in.append(dataclasses.replace(need, reason=McpSignInReason.RENEWAL_FAILED))
                continue
            except (McpSignInRequiredError, WorkspaceMcpServerNotFoundError):
                if not defer:
                    raise
                resolved.sign_in.append(need)
                continue
            except McpConnectionUnavailableError:
                if not defer:
                    raise
                resolved.sign_in.append(dataclasses.replace(need, reason=McpSignInReason.RENEWAL_FAILED))
                continue
            except (SecretDecryptionError, SecretBoxUnavailableError):
                if not config.attached:
                    raise
                _leave_out_unreadable(need.server_id)
                continue
        elif need.server_id in accounts:
            shared_ref, unset = accounts[need.server_id]
            try:
                if bound is None:
                    account_token = await tokens.shared_token_for(shared_ref)
                else:
                    async with bound:
                        account_token = await tokens.shared_token_for(shared_ref)
            except TimeoutError:
                if bound is None or not bound.expired():
                    raise
                logger.warning(
                    "Attached MCP server left out: its token was not renewed in time server_id=%s", need.server_id
                )
                continue
            except (SecretDecryptionError, SecretBoxUnavailableError, McpConnectionUnavailableError):
                if not config.attached:
                    raise
                _leave_out_unreadable(need.server_id)
                continue
            if account_token is not None:
                config.authorization_token = account_token
                config.renew_token = partial(tokens.renew_rejected_shared, shared_ref)
            elif unset:
                continue
        resolved.servers.append(config)
    return resolved


def _leave_out_unreadable(server_id: uuid.UUID) -> None:
    """Log an attached server left out of a request because a credential it sends will not decrypt.

    A named server fails the request instead: connecting without the credential would send an
    unauthenticated request. An attached one joins every request of its workspace, so it cannot fail them all.
    """
    logger.error("Attached MCP server left out: a stored credential will not decrypt server_id=%s", server_id)


def _needs_a_user(row: WorkspaceMcpServer, access: McpAccess | None) -> bool:
    """Whether a server sends each user's own credential and the caller has no user to send one for."""
    return row.credential_mode == McpCredentialMode.PER_USER and (access is None or access.user_id is None)


def _renewer(
    row: WorkspaceMcpServer, access: McpAccess | None, tokens: McpUserTokens
) -> Callable[[str], Awaitable[str]] | None:
    """What the call path asks for a new token with when the server refuses one, on a per-user server only."""
    if row.credential_mode != McpCredentialMode.PER_USER or access is None or access.user_id is None:
        return None
    # A copy: the call runs after the request's session may have been released, which can expire the row.
    return partial(tokens.renew_rejected, McpServerRef.of(row), access.user_id)


async def _caller_token(ref: McpServerRef, access: McpAccess | None, tokens: McpUserTokens) -> str:
    """A per-user server's token for the caller's own user, read from a copy of the row.

    Raises:
        WorkspaceMcpServerNotFoundError: the caller has no user, who could hold a connection.
        McpSignInRequiredError: the caller's user has not signed in to it.
        McpConnectionUnavailableError: that user's token has expired and cannot be renewed right now.
        SecretDecryptionError: a stored token will not decrypt.
    """
    if access is None or access.user_id is None:
        raise WorkspaceMcpServerNotFoundError(ref.id)
    return await tokens.token_for(ref, access.user_id)


async def resolve_workspace_mcp_server(
    db: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    server_id: uuid.UUID,
    access: McpAccess | None = None,
    tokens: McpUserTokens | None = None,
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

    With the caller's ``access``, a disabled server is ``None`` and one the
    access list grants nothing on raises
    :class:`~gateway.exceptions.tools_exceptions.WorkspaceMcpServerAccessDeniedError`,
    both before the token is decrypted, so a token that will not decrypt cannot
    tell a caller with no access that the server exists.

    A per-user server is ``None`` for a caller with no user, and raises
    :class:`McpSignInRequiredError` when the caller's user has not signed in
    to it, which is only ever said to a caller the access list admitted.

    A shared server with no token sends its shared account's. With neither, and a
    server that last asked for a credential, it is ``None`` for a caller with ``access``,
    as a disabled one is.
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
    if access is not None:
        if not row.enabled:
            return None
        if not access.patterns_for(row.id, row.everyone_tools, _read_tools(row)):
            source, _ = access.decide(row.id, row.everyone_tools)
            raise WorkspaceMcpServerAccessDeniedError(row.id, name=row.name, decided_by=f"mcp_{source}")
    if _needs_a_user(row, access):
        return None
    per_user = row.credential_mode == McpCredentialMode.PER_USER
    tokens = tokens or McpUserTokens(McpUserConnectionRepository(db))
    # Read in full before the token is: reading a per-user one may release the session and expire the row.
    resolved = ResolvedMcpServer(
        id=row.id,
        name=row.name,
        url=row.url,
        authorization_token=None if per_user else _stored_token(row),
        renew_token=_renewer(row, access, tokens),
        # A user's token changes at every renewal, which is not a change to the server's configuration.
        credential_revision=McpCredentialMode.PER_USER if per_user else None,
        enabled=row.enabled,
        everyone_tools=row.everyone_tools,
        read_tools=_read_tools(row),
        purpose_hint=row.purpose_hint,
        allowed_tools=row.allowed_tools,
    )
    ref, unset = McpServerRef.of(row), _needs_setup(row, has_account=False)
    if per_user:
        resolved.authorization_token = await _caller_token(ref, access, tokens)
    elif resolved.authorization_token is None:
        account = await tokens.shared_account_for(ref)
        if account is not None:
            resolved.authorization_token, generation = account
            resolved.renew_token = partial(tokens.renew_rejected_shared, ref)
            # The account's token changes at every renewal, which changes nothing an application
            # authorized; a manager signing in to another account does.
            resolved.credential_revision = f"shared_account:{generation}"
        elif unset and access is not None:
            return None
    return resolved


async def list_reachable_mcp_servers(
    db: AsyncSession, *, workspace_id: uuid.UUID, access: McpAccess, tokens: McpUserTokens | None = None
) -> list[McpServerSummary]:
    """The enabled stored servers of a workspace the caller may reach, oldest first.

    A per-user server is left out for a caller with no user, and marked ``sign_in_required`` for
    one whose user has not signed in to it. A shared server that still needs a credential is left out.
    """
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
    reachable = [
        row
        for row in rows
        if access.patterns_for(row.id, row.everyone_tools, _read_tools(row)) and not _needs_a_user(row, access)
    ]
    per_user = [row.id for row in reachable if row.credential_mode == McpCredentialMode.PER_USER]
    tokens = tokens or McpUserTokens(McpUserConnectionRepository(db))
    connected = await tokens.connected(access.user_id, per_user)
    unset = [row.id for row in reachable if _needs_setup(row, has_account=False)]
    with_account = await tokens.with_shared_account(unset)
    reachable = [row for row in reachable if row.id not in unset or row.id in with_account]
    return [
        McpServerSummary(
            id=row.id,
            name=row.name,
            purpose_hint=row.purpose_hint,
            sign_in_required=row.id in per_user and row.id not in connected,
        )
        for row in reachable
    ]


async def _grant_or_404(
    repository: WorkspaceMcpServerGrantRepository, server_id: uuid.UUID, grant_id: uuid.UUID
) -> WorkspaceMcpServerGrant:
    grant = await repository.get_on_server(server_id, grant_id)
    if grant is None:
        raise WorkspaceMcpServerGrantNotFoundError(grant_id)
    return grant


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

    def __init__(self, db: AsyncSession, tokens: McpUserTokens | None = None) -> None:
        self._db = db
        # Reads the token a per-user server's caller carries, on the same session.
        self._tokens = tokens or McpUserTokens(McpUserConnectionRepository(db))

    async def resolve_many(
        self, workspace_id: uuid.UUID, server_ids: list[uuid.UUID], access: McpAccess
    ) -> list[McpServerConfig]:
        """See :func:`resolve_workspace_mcp_servers`."""
        return await resolve_workspace_mcp_servers(
            self._db, workspace_id=workspace_id, server_ids=server_ids, access=access, tokens=self._tokens
        )

    async def resolve_for_request(
        self, workspace_id: uuid.UUID, server_ids: list[uuid.UUID], access: McpAccess, *, attach: bool
    ) -> RequestMcpServers:
        """The stored servers a model request reaches: the ones it names, then with ``attach`` the attached ones.

        A named server follows :func:`resolve_workspace_mcp_servers`, except that a per-user one the
        request cannot open does not fail it: it comes back in ``sign_in``, with why. A server marked
        ``attach_by_default`` that the request did not name is added as if it had been, after the named
        ones, and is silently left out where the caller's access grants nothing on it, where the caller
        has no user to sign in as, or where it is disabled.

        Raises:
            WorkspaceMcpServerNotFoundError: a named ID is unknown or grants the caller nothing.
            SecretDecryptionError: a stored token will not decrypt.
        """
        return await _resolve_for_request(
            self._db,
            workspace_id=workspace_id,
            server_ids=server_ids,
            attach=attach,
            access=access,
            tokens=self._tokens,
            defer=True,
        )

    async def any_attached(self, workspace_id: uuid.UUID) -> bool:
        """Whether the workspace has an enabled server marked ``attach_by_default``, in one indexed read."""
        return await WorkspaceMcpServerRepository(self._db).any_attached(workspace_id)

    async def list_reachable(self, workspace_id: uuid.UUID, access: McpAccess) -> list[McpServerSummary]:
        """See :func:`list_reachable_mcp_servers`."""
        return await list_reachable_mcp_servers(self._db, workspace_id=workspace_id, access=access, tokens=self._tokens)

    async def resolve_one(
        self, workspace_id: uuid.UUID, server_id: uuid.UUID, access: McpAccess | None = None
    ) -> ResolvedMcpServer | None:
        """See :func:`resolve_workspace_mcp_server`."""
        return await resolve_workspace_mcp_server(
            self._db, workspace_id=workspace_id, server_id=server_id, access=access, tokens=self._tokens
        )


class WorkspaceMcpServerService:
    """CRUD for a workspace's MCP servers. Writes are management-gated; the list is not."""

    def __init__(
        self,
        db: AsyncSession,
        api_keys: ApiKeyService | None = None,
        *,
        connections: McpConnectionService | None = None,
        requests: McpServerRequestService | None = None,
        config: GatewayConfig | None = None,
    ):
        self.db = db
        self.organizations = OrganizationService(db, membership_listener=None)
        # Reads the workspace's keys, which a server's access list names.
        self.api_keys = api_keys
        # Runs a user's sign-in to a per-user server, on the request's Unit of Work.
        self.connections = connections
        # Approves the members' requests a new server answers, in the transaction that stores it.
        self.requests = requests
        self.config = config
        self._tokens = McpUserTokens(McpUserConnectionRepository(db))

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

    def _connection_service(self) -> McpConnectionService:
        if self.connections is None:
            raise RuntimeError("WorkspaceMcpServerService was built without the connection service a sign-in needs")
        return self.connections

    def _require_per_user_support(self) -> None:
        """Refuse per-user mode where this deployment could not run a sign-in or keep its tokens."""
        if self.config is None:
            raise RuntimeError("WorkspaceMcpServerService was built without the config per-user mode reads")
        McpConnectionService.require_per_user_support(self.config)

    async def _revocations(self, server: WorkspaceMcpServer) -> tuple[list[McpRevocation], int]:
        """What to ask the authorization server to end once this server's connections are deleted.

        Read while the server still holds the client the tokens were issued to. Also how many it
        leaves unread past the page one removal reads.
        """
        connections = McpUserConnectionRepository(self.db)
        rows, total = await connections.revocable(server_ids=[server.id])
        account = await connections.find_shared(server.id)
        return McpConnectionService.revocations_for(
            server, [*rows, *([account] if account is not None else [])]
        ), total - len(rows)

    async def _end_connections(self, server: WorkspaceMcpServer) -> None:
        """Stage the end of every sign-in to a server, and of the client Otari registered for it.

        For a change after which the tokens no longer fit: another URL, another client, the other mode,
        or a token set in the shared account's place. A manager's own client is kept, so a server
        moved to the other mode and back still has it.
        """
        connections = McpUserConnectionRepository(self.db)
        await connections.remove_for_server(server.id)
        await connections.remove_shared_for_server(server.id)
        await McpOAuthFlowRepository(self.db).remove_for_server(server.id)
        client = McpConnectionService.stored_oauth_client(server)
        if client is not None and client.dynamic:
            server.oauth_client = None
            server.oauth_client_issuer = None
            server.oauth_client_token_endpoint = None

    async def _record_check(
        self, server: WorkspaceMcpServer, *, required: bool = False, check_token: str | None = None
    ) -> bool:
        """Check a stored server's connection and save the outcome as its health.

        A shared server is checked with its stored token, or its shared account's. A per-user server is checked with
        ``check_token``, from a sign-in made for this check alone, and never with anyone's
        connection. Without one (unless the check is ``required``), only its sign-in setup is
        checked, so a setup nobody could sign in through shows on the server as soon as it is saved.

        The transaction is ended first, so no pooled connection is held while
        the gateway waits on a remote server for up to the discovery deadline.
        Returns false when no check slot was free, leaving the row's health as
        it was.

        An edit can land while the check waits. The row is re-read under a lock
        before the write, and a check of a URL, token or allow-list the row no
        longer holds is dropped: that edit cleared health and runs its own.

        Raises:
            McpCheckSignInRequiredError: ``required`` on a per-user server with no ``check_token``.
        """
        checked = (server.url, server.encrypted_token, server.allowed_tools, server.credential_mode)
        per_user = server.credential_mode == McpCredentialMode.PER_USER
        if per_user and check_token is None:
            await self.db.commit()
            if required:
                raise McpCheckSignInRequiredError
            problem = await McpConnectionService.sign_in_problem(server.url)
            if problem is None:
                await self.db.refresh(server)
                return True
            check = ServerCheck(tool_names=[], failure=problem)
        else:
            # Read before the transaction ends, so nothing reopens it across the remote check.
            token: str | None | _NoCredential
            if per_user:
                token = check_token
            else:
                try:
                    token = _stored_token(server)
                    if token is None:
                        token = await self._tokens.shared_token_for(server)
                except (SecretDecryptionError, SecretBoxUnavailableError, McpConnectionUnavailableError):
                    token = _UNREADABLE
            await self.db.commit()
            try:
                check = await _check(server, token)
            except McpCapacityUnavailable:
                logger.info("MCP server check skipped server_id=%s reason=capacity", server.id)
                return False
        await self.db.refresh(server, with_for_update=True)
        if (server.url, server.encrypted_token, server.allowed_tools, server.credential_mode) != checked:
            await self.db.commit()
            logger.info("MCP server check dropped server_id=%s reason=edited", server.id)
            return True
        health = WorkspaceMcpServerHealth.from_check(server.allowed_tools, check)
        if check.failure is not None and server.health is not None:
            # What '@read' stands for until a manager labels tools: one failed check, possibly with one
            # user's token, must not empty it for everyone, as one failed contact does not.
            previous = WorkspaceMcpServerHealth.model_validate(server.health)
            health = health.model_copy(update={"read_only_hinted_tools": previous.read_only_hinted_tools})
        server.health = health.model_dump(mode="json")
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
        own = await self._own_rules(user, [server.id for server in servers])
        accounts = await self._shared_accounts(servers)
        connected = await self._connected_users(servers) if manages else {}
        return WorkspaceMcpServersPublic(
            data=[
                WorkspaceMcpServerPublic.from_model(
                    server,
                    redact_url=not manages,
                    caller_tools=own.get(server.id, server.everyone_tools),
                    shared_account=accounts.get(server.id),
                    connected_users=connected.get(server.id),
                )
                for server in servers
            ],
            count=count,
        )

    async def _shared_accounts(self, servers: Sequence[WorkspaceMcpServer]) -> dict[uuid.UUID, McpSharedAccountPublic]:
        """The shared account of each of these servers that has one, naming who signed in to it."""
        shared = [server.id for server in servers if server.credential_mode == McpCredentialMode.SHARED]
        rows: dict[uuid.UUID, McpSharedConnection] = await McpUserConnectionRepository(self.db).shared_for_servers(
            shared
        )
        if not rows:
            return {}
        signers = {row.connected_by for row in rows.values() if row.connected_by is not None}
        identities = {identity.id: identity for identity in await UserRepository(self.db).get_by_ids(signers)}
        accounts: dict[uuid.UUID, McpSharedAccountPublic] = {}
        for server_id, row in rows.items():
            identity = identities.get(row.connected_by) if row.connected_by is not None else None
            accounts[server_id] = McpSharedAccountPublic(
                signed_in_by=(identity.full_name or identity.email) if identity is not None else None,
                signed_in_at=row.created_at.isoformat(),
                renewable=row.encrypted_refresh_token is not None,
            )
        return accounts

    async def _connected_users(self, servers: Sequence[WorkspaceMcpServer]) -> dict[uuid.UUID, int]:
        """How many users are connected to each per-user server of these, in one query; 0 where nobody is."""
        per_user = [server.id for server in servers if server.credential_mode == McpCredentialMode.PER_USER]
        counts = await McpUserConnectionRepository(self.db).counts_for_servers(per_user)
        return {server_id: counts.get(server_id, 0) for server_id in per_user}

    async def _public(
        self, server: WorkspaceMcpServer, *, redact_url: bool = False, caller_tools: Sequence[str] | None = None
    ) -> WorkspaceMcpServerPublic:
        """One server's public shape, with its shared account read, and for a manager's read its connection count."""
        accounts = await self._shared_accounts([server])
        connected = {} if redact_url else await self._connected_users([server])
        return WorkspaceMcpServerPublic.from_model(
            server,
            redact_url=redact_url,
            caller_tools=caller_tools,
            shared_account=accounts.get(server.id),
            connected_users=connected.get(server.id),
        )

    async def _own_rules(self, user: User, server_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[str]]:
        """The tools the caller's own user grant gives them on each of these servers that has one."""
        grants = await WorkspaceMcpServerGrantRepository(self.db).for_caller(None, str(user.id), server_ids)
        return {grant.server_id: list(grant.tools) for grant in grants if grant.api_key_id is None}

    async def create_server(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        request: WorkspaceMcpServerCreate,
        growth: GrowthSignalPort | None = None,
        background_tasks: BackgroundTasks | None = None,
    ) -> WorkspaceMcpServerPublic:
        """Register a server, encrypting its token before it is stored.

        With ``growth``, the server is reported as ``MCP_SERVER_CREATED`` once it commits.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        per_user = request.credential_mode == McpCredentialMode.PER_USER
        if per_user:
            self._require_per_user_support()
        # A per-user server is held to https as one with a token is: a user's token travels to it.
        await _validate_url(request.url, has_token=per_user or bool(request.authorization_token))

        # The workspace lock stops a concurrent create from invalidating the count below before this insert.
        workspaces = WorkspaceRepository(self.db)
        await workspaces.lock(resolved_workspace_id)
        organization_id = await workspaces.get_organization_id(resolved_workspace_id)

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
            read_tools=request.read_tools,
            credential_mode=request.credential_mode,
            oauth_client=_own_client(request.oauth_client_id, request.oauth_client_secret),
            attach_by_default=request.attach_by_default,
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

        if self.requests is not None:
            workspace = await workspaces.get(resolved_workspace_id)
            await self.requests.server_added(
                workspace_id=resolved_workspace_id,
                workspace_name=workspace.name if workspace is not None else "your workspace",
                server_id=server.id,
                url=server.url,
                added_by=user.id,
            )
        await self.db.commit()
        # After the commit, so a server that fails its check is still stored:
        # it may be registered before it is up, and `health` says what failed.
        await self._record_check(server)
        if growth is not None and background_tasks is not None and organization_id is not None:
            await growth.record_action(
                background_tasks=background_tasks,
                action=GrowthAction.MCP_SERVER_CREATED,
                user_id=user.id,
                email=user.email,
                organization_id=organization_id,
            )
        return await self._public(server)

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
        # Locked and read again: an edit that moved the URL meanwhile must be what this one decides
        # against, or a secret saved here would land on a client the move had dropped.
        await self.db.refresh(server, with_for_update=True)
        fields = request.model_dump(exclude_unset=True)

        new_token = fields.pop("authorization_token", _UNSET)
        new_client_id = fields.pop("oauth_client_id", _UNSET)
        new_client_secret = fields.pop("oauth_client_secret", _UNSET)
        was_per_user = server.credential_mode == McpCredentialMode.PER_USER
        per_user = fields.get("credential_mode", server.credential_mode) == McpCredentialMode.PER_USER
        mode_changed = per_user != was_per_user
        if per_user and isinstance(new_token, str) and new_token:
            raise McpConnectionSetupError(_PER_USER_TAKES_NO_TOKEN)
        connections = McpUserConnectionRepository(self.db)
        has_account = not was_per_user and await connections.find_shared(server.id) is not None
        if per_user and mode_changed:
            self._require_per_user_support()
            # The shared token goes with the mode: nothing sends it any more.
            new_token = ""
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
        if url_changed or mode_changed or (isinstance(new_token, str) and not per_user):
            await _validate_url(fields.get("url") or server.url, has_token=per_user or effective_has_token)
        # What the stored health described. Rechecked after the commit; cleared
        # here so a check that cannot run leaves no health rather than a stale one.
        connection_changed = (
            url_changed
            or mode_changed
            or isinstance(new_token, str)
            or ("allowed_tools" in fields and fields["allowed_tools"] != server.allowed_tools)
        )

        if new_token == "":
            server.encrypted_token = None
        elif isinstance(new_token, str):
            server.encrypted_token = _encrypted(new_token)

        moves_host = url_changed and _origin(fields["url"]) != _origin(server.url)
        client, store_client, another_client = _edited_client(
            McpConnectionService.stored_oauth_client(server), new_client_id, new_client_secret, moves_host=moves_host
        )
        sign_ins_moved = mode_changed or url_changed or another_client
        # A server sends one credential: a token set on a shared server takes its account's place.
        ends_account = has_account and (sign_ins_moved or (isinstance(new_token, str) and bool(new_token)))
        ends_connections = (was_per_user and sign_ins_moved) or ends_account
        connection_changed = connection_changed or ends_account
        revocations, unread = await self._revocations(server) if ends_connections else ([], 0)
        if store_client:
            server.oauth_client = client
        # A manager's client, and its secret, were for the old host: a new origin does not inherit them.
        names_a_client = isinstance(new_client_id, str) and bool(new_client_id)
        if moves_host and not names_a_client:
            server.oauth_client = None
        # Saving the client again approves whichever authorization server its next sign-in finds.
        if store_client or moves_host or isinstance(new_client_id, str) or isinstance(new_client_secret, str):
            server.oauth_client_issuer = None
            server.oauth_client_token_endpoint = None
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

        # The users' tokens were issued for the server and the client as they were. After the flush,
        # which holds the row: a sign-in in flight then either sees this edit or is removed here.
        if ends_connections:
            await self._end_connections(server)

        await self.db.commit()
        McpConnectionService.revoke_in_background(revocations, unread=unread)
        if connection_changed:
            await self._record_check(server)
        else:
            await self.db.refresh(server)
        return await self._public(server)

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
        own = await self._own_rules(user, [server.id])
        return await self._public(
            server, redact_url=not manages, caller_tools=own.get(server.id, server.everyone_tools)
        )

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

    async def _members_by_attribution(self, workspace_id: uuid.UUID, user_ids: list[str]) -> set[str]:
        """Which of ``user_ids`` name someone the Playground lets into the workspace, minting missing rows.

        That is an active member of the workspace, or an owner or admin of its organization, who needs no workspace
        row. Their request-plane id is their identity's UUID as a string, as the Playground and member keys write it,
        so only that canonical spelling is accepted: a rule under another spelling would bind nobody.
        """
        identities: dict[uuid.UUID, str] = {}
        for user_id in user_ids:
            try:
                identity = uuid.UUID(user_id)
            except ValueError:
                continue
            if str(identity) == user_id:
                identities[identity] = user_id
        if not identities:
            return set()
        workspace = await WorkspaceRepository(self.db).get(workspace_id)
        pairs = await WorkspaceMemberRepository(self.db).get_active_pairs([workspace_id], list(identities))
        eligible = {identity for _, identity in pairs}
        if workspace is not None:
            eligible |= await OrganizationMemberRepository(self.db).active_with_roles(
                workspace.organization_id, identities.keys() - eligible, MANAGEMENT_ROLES
            )
        return await provision_attribution_users(self.db, [identities[identity] for identity in eligible])

    async def add_grants(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, request: WorkspaceMcpServerGrantsCreate
    ) -> WorkspaceMcpServerAccess:
        """Give keys and users the same tools on a server, replacing what any of them held.

        A key must belong to the server's workspace. A user must own a key in it, or be an active member of it,
        whom the Playground binds as their user before they hold any key; such a member's request-plane row is
        minted here if they have none yet.

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
            members = await self._members_by_attribution(
                resolved_workspace_id, [user_id for user_id in request.user_ids if user_id not in owners]
            )
            strangers = [user_id for user_id in request.user_ids if user_id not in owners and user_id not in members]
            if strangers:
                raise WorkspaceMcpServerGrantSubjectError(
                    f"'{strangers[0]}' is not a member of this workspace and owns no key in it"
                )
        repository = WorkspaceMcpServerGrantRepository(self.db)
        # The key reads above end in commits; the server is read again here, locked, so two
        # concurrent adds cannot both pass the cap.
        server = await self.db.get(WorkspaceMcpServer, server_id, with_for_update=True)
        if server is None:
            raise WorkspaceMcpServerNotFoundError(server_id)
        # Locked as a user's deletion locks it, so a grant and the deletion that removes the user's grants cannot
        # interleave: whichever commits second sees the other, and a deleted user is refused here.
        live = await lock_live_users(self.db, request.user_ids)
        gone = [user_id for user_id in request.user_ids if user_id not in live]
        if gone:
            raise WorkspaceMcpServerGrantSubjectError(f"'{gone[0]}' has been deleted")
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
        grant = await _grant_or_404(repository, server.id, grant_id)
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
        grant = await _grant_or_404(repository, server.id, grant_id)
        await repository.remove(grant)
        await self.db.commit()
        return await self._access_of(resolved_workspace_id, server)

    async def check_access(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, api_key_id: str, tool: str
    ) -> WorkspaceMcpServerAccessCheck:
        """Whether a key of the workspace may call ``tool`` on a server, and which rule decides.

        Raises:
            WorkspaceMcpServerKeyNotFoundError: the key is not of this workspace.
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
            allowed=is_mcp_tool_allowed(expand_mcp_tool_patterns(patterns, _read_tools(server)), tool),
            decided_by=decided,
            grant_id=grant_id,
            tools=patterns,
        )

    async def _decide_for_key(
        self, workspace_id: uuid.UUID, api_key_id: str, server: WorkspaceMcpServer
    ) -> tuple[Literal["key", "user", "everyone"], uuid.UUID | None, list[str]]:
        """Which rule decides a key's access to a server: its grant, its owner's, or ``everyone_tools``."""
        key = await self._key_in_workspace(workspace_id, api_key_id)
        return _decide(key, server, await WorkspaceMcpServerGrantRepository(self.db).for_server(server.id))

    async def _key_in_workspace(self, workspace_id: uuid.UUID, api_key_id: str) -> KeySummary:
        """One key of the workspace, or :class:`WorkspaceMcpServerKeyNotFoundError` when it is not one."""
        found = await self._key_service().summaries_in_workspace(workspace_id, [api_key_id])
        if not found:
            raise WorkspaceMcpServerKeyNotFoundError(api_key_id)
        return found[0]

    async def key_access(
        self, *, user: User, workspace_id: uuid.UUID, api_key_id: str
    ) -> list[WorkspaceMcpServerKeyAccess]:
        """What one key of the workspace reaches on each of its stored servers, oldest server first.

        Raises:
            WorkspaceMcpServerKeyNotFoundError: the key is not of this workspace.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        key = await self._key_in_workspace(resolved_workspace_id, api_key_id)
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
        for grant in await WorkspaceMcpServerGrantRepository(self.db).for_caller(
            key.id, key.user_id, [server.id for server in servers]
        ):
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
                    read_tools=_read_tools(server),
                )
            )
        return result

    async def check_connection(
        self, *, user: User, workspace_id: uuid.UUID, request: WorkspaceMcpServerCheckRequest
    ) -> WorkspaceMcpServerHealth:
        """Check connection details without storing anything, for a form to test before it saves.

        An unsafe URL is refused with the same 400 a create gets, since this is
        the form that would go on to store it.

        Raises:
            WorkspaceMcpServerCheckOriginError: the check borrows a stored server's credential for a
                URL on another origin than that server's.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        token: str | None = None
        if request.authorization_token is not None:
            token = request.authorization_token or None
        elif request.server_id is not None:
            server = await self._get_or_404(resolved_workspace_id, request.server_id)
            if server.credential_mode == McpCredentialMode.PER_USER:
                raise McpConnectionSetupError(
                    "A per-user server holds no token to test with: connect your account, then check the server"
                )
            # The stored token or shared account goes only where the server it was stored for is.
            if _origin(request.url) != _origin(server.url):
                logger.warning("MCP connection test refused reason=other_origin server_id=%s", server.id)
                raise WorkspaceMcpServerCheckOriginError
            try:
                token = _stored_token(server) or await self._tokens.shared_token_for(server)
            except (SecretDecryptionError, SecretBoxUnavailableError, McpConnectionUnavailableError):
                check = ServerCheck(tool_names=[], failure=CheckFailure.CREDENTIALS_UNAVAILABLE)
                return WorkspaceMcpServerHealth.from_check(request.allowed_tools, check)
        await _validate_url(request.url, has_token=token is not None)
        # Ended before waiting on a remote server, as `_record_check` does.
        await self.db.commit()
        try:
            check = await _connect(request.url, token, server_id=request.server_id)
        except McpCapacityUnavailable:
            raise WorkspaceMcpServerCheckBusyError from None
        logger.info("MCP server connection tested workspace_id=%s outcome=%s", workspace_id, check.failure or "ok")
        return WorkspaceMcpServerHealth.from_check(request.allowed_tools, check)

    async def check_server(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID
    ) -> WorkspaceMcpServerPublic:
        """Check a server's connection now and store the outcome as its health.

        A failing check is a result, not an error: the request succeeded in
        finding out that the server did not.

        Raises:
            McpCheckSignInRequiredError: the server is per-user, which is checked through
                :meth:`start_check_sign_in` instead.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        if not await self._record_check(server, required=True):
            raise WorkspaceMcpServerCheckBusyError
        return await self._public(server)

    async def delete_server(self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID) -> None:
        """Delete a server, the token stored with it, and every user's connection to it.

        The connections' tokens are deleted here, and their authorization server is asked to end
        them once the deletion has committed, best effort and without waiting.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        # The server before its connections, in the order a sign-in's completion takes them.
        await self.db.refresh(server, with_for_update=True)
        revocations, unread = await self._revocations(server)
        # Staged here as well as cascaded by the foreign keys, so the tokens go where a database enforces none.
        connections = McpUserConnectionRepository(self.db)
        await connections.remove_for_server(server.id)
        await connections.remove_shared_for_server(server.id)
        await McpOAuthFlowRepository(self.db).remove_for_server(server.id)
        await self.db.delete(server)
        await self.db.commit()
        McpConnectionService.revoke_in_background(revocations, unread=unread)

    async def my_tools(self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID) -> WorkspaceMcpServerMyTools:
        """The tools the caller may use on a server, for any member who can see the workspace.

        Read from the server's last recorded tool listing. Where none has listed its tools and the server
        is per-user, they are listed now with the caller's own connection, when they have one, and
        what the server answered is recorded as its health.
        """
        resolved_workspace_id = await self._visible_workspace_id(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        user_id = str(user.id)
        grants = await WorkspaceMcpServerGrantRepository(self.db).for_server(server.id)
        own = next((grant for grant in grants if grant.api_key_id is None and grant.user_id == user_id), None)
        decided_by: Literal["user", "everyone", "server"] = "user" if own is not None else "everyone"
        read_tools = _read_tools(server)
        patterns = expand_mcp_tool_patterns(
            list(own.tools) if own is not None else list(server.everyone_tools), read_tools
        )
        ceiling = server.allowed_tools
        if not server.enabled:
            decided_by, patterns = "server", []
        if not patterns or ceiling == []:
            access: Literal["all", "some", "none"] = "none"
        elif "*" in patterns and ceiling is None:
            access = "all"
        else:
            access = "some"

        health = WorkspaceMcpServerHealth.model_validate(server.health) if server.health is not None else None
        listed: tuple[list[str], dict[str, str], str] | None = None
        if health is not None and health.status == "healthy" and health.tool_count is not None:
            listed = (health.tools, health.tool_descriptions, health.checked_at)
        elif access != "none" and server.credential_mode == McpCredentialMode.PER_USER:
            listed = await self._list_with_own_connection(server, user_id)
        if listed is None:
            return WorkspaceMcpServerMyTools(
                server_id=server.id, decided_by=decided_by, access=access, known=False, checked_at=None, tools=[]
            )
        names, descriptions, checked_at = listed
        tools = [
            McpToolSummary(name=name, description=descriptions.get(name), read_only=name in read_tools)
            for name in names
            if access != "none" and (ceiling is None or name in ceiling) and is_mcp_tool_allowed(patterns, name)
        ]
        return WorkspaceMcpServerMyTools(
            server_id=server.id, decided_by=decided_by, access=access, known=True, checked_at=checked_at, tools=tools
        )

    async def _list_with_own_connection(
        self, server: WorkspaceMcpServer, user_id: str
    ) -> tuple[list[str], dict[str, str], str] | None:
        """A per-user server's tools as the caller's own connection lists them; ``None`` when it cannot."""
        try:
            token = await _check_credential(server, self._tokens, user_id)
        except McpSignInRequiredError:
            return None
        # Ended before waiting on a remote server, as `_record_check` does.
        await self.db.commit()
        try:
            check = await _check(server, token)
        except McpCapacityUnavailable:
            return None
        mcp_server_contacts.observed(server.id, server.url, check)
        if check.failure is not None:
            logger.info("MCP server tools not listed server_id=%s outcome=%s", server.id, check.failure)
            return None
        health = WorkspaceMcpServerHealth.from_check(None, check)
        return health.tools, health.tool_descriptions, health.checked_at

    async def _visible_workspace_id(self, user: User, workspace_id: uuid.UUID) -> uuid.UUID:
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        return workspace.id

    async def start_connection(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, return_to: str | None
    ) -> McpConnectionStarted:
        """Start the caller's own sign-in to a per-user server, for any member who can see the workspace.

        The connection belongs to the caller's request-plane user, the one their keys and the
        Playground bill to, so every key they own in this workspace then reaches the server as them.
        The sign-in finishes only in the caller's own dashboard session.

        Raises:
            McpConnectionSetupError: the caller's user was deleted by an operator.
        """
        resolved_workspace_id = await self._visible_workspace_id(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        user_id = str(user.id)
        if user_id not in await provision_attribution_users(self.db, [user_id]):
            raise McpConnectionSetupError("Your user was removed from this deployment, so it cannot hold a connection")
        # The row minted above is committed by the connection service's first block, on this same session.
        url = await self._connection_service().start(
            server_id=server.id, user_id=user_id, return_to=return_to, identity_id=user.id
        )
        return McpConnectionStarted(authorization_url=url)

    async def start_check_sign_in(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, return_to: str | None
    ) -> McpConnectionStarted:
        """Start a manager's sign-in made only to check a per-user server.

        The manager signs in with whichever account they choose. Finishing it lists the server's
        tools once with the token it returns, stores the outcome as the server's health, and drops
        the token: no connection is stored, and the manager's own connection, if any, is untouched.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        user_id = str(user.id)
        if user_id not in await provision_attribution_users(self.db, [user_id]):
            raise McpConnectionSetupError("Your user was removed from this deployment, so it cannot sign in")
        url = await self._connection_service().start(
            server_id=server.id, user_id=user_id, return_to=return_to, identity_id=user.id, for_check=True
        )
        return McpConnectionStarted(authorization_url=url)

    async def start_shared_sign_in(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, return_to: str | None
    ) -> McpConnectionStarted:
        """Start a manager's sign-in for everyone, to a shared server.

        Finishing it stores the account the manager signed in to as the server's one credential,
        in place of a token: every caller's call then carries it.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        self._require_per_user_support()
        # A token travels to the server from here on, so it is held to https as one with a token is.
        await _validate_url(server.url, has_token=True)
        user_id = str(user.id)
        if user_id not in await provision_attribution_users(self.db, [user_id]):
            raise McpConnectionSetupError("Your user was removed from this deployment, so it cannot sign in")
        url = await self._connection_service().start(
            server_id=server.id, user_id=user_id, return_to=return_to, identity_id=user.id, for_shared=True
        )
        return McpConnectionStarted(authorization_url=url)

    async def disconnect_shared(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID
    ) -> WorkspaceMcpServerPublic:
        """End a shared server's account, and check the server again without it. Workspace managers only.

        Raises:
            McpConnectionNotFoundError: the server has no shared account.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        await self._connection_service().disconnect_shared(server_id=server.id)
        await self.db.refresh(server)
        await self._record_check(server)
        return await self._public(server)

    async def _record_after_sign_in(self, done: CompletedConnection) -> None:
        """List a server's tools once with a sign-in that just finished, and record what it answered.

        A sign-in that worked is news about the server, so it is recorded as a check is, a
        failed listing included. Nothing here fails the sign-in, which is already stored.
        """
        try:
            server = await self._get_or_404(done.workspace_id, done.server_id)
            if server.url != done.server_url:
                # Edited since the sign-in: that edit ended the sign-in and runs its own check.
                return
            if done.shared:
                await self._record_check(server)
            elif done.user_id is not None:
                token = await _check_credential(server, self._tokens, done.user_id)
                if isinstance(token, str):
                    await self._record_check(server, check_token=token)
        except TenancyError as exc:
            logger.info(
                "MCP server not listed after sign-in server_id=%s reason=%s", done.server_id, type(exc).__name__
            )
        except DATABASE_ERRORS as exc:
            logger.warning(
                "MCP server not listed after sign-in server_id=%s error=%s",
                done.server_id,
                describe_database_error(exc),
            )

    async def complete_connection(
        self, *, user: User, state: str, code: str, iss: str | None = None
    ) -> McpConnectionCompleted:
        """Finish a sign-in the caller started, from the code its authorization server returned.

        A sign-in made to check a server ends here in that check, run once with its token, which is
        then dropped and, where the authorization server allows it, revoked. Any other lists the
        server's tools once with the sign-in it just stored and records the outcome as its health.

        Raises:
            WorkspaceMcpServerCheckBusyError: a check sign-in finished, but no check slot was free.
        """
        done = await self._connection_service().complete(state=state, code=code, identity_id=user.id, iss=iss)
        if done.check_token is not None:
            try:
                # Asked again: the caller may have lost the right to manage the workspace while signing in.
                resolved_workspace_id = await self._require_management(user, done.workspace_id)
                server = await self._get_or_404(resolved_workspace_id, done.server_id)
                if server.url != done.server_url:
                    # The URL was edited after the sign-in was approved: its token goes nowhere else.
                    logger.info("MCP check sign-in dropped server_id=%s reason=url_changed", done.server_id)
                    raise McpConnectionFlowError("The server changed while you were signing in. Start it again.")
                recorded = await self._record_check(server, required=True, check_token=done.check_token)
            finally:
                if done.revocation is not None:
                    McpConnectionService.revoke_in_background([done.revocation])
            if not recorded:
                raise WorkspaceMcpServerCheckBusyError
        else:
            await self._record_after_sign_in(done)
        return McpConnectionCompleted(
            server_id=done.server_id,
            workspace_id=done.workspace_id,
            server_name=done.server_name,
            return_to=done.return_to,
            checked=done.check_token is not None,
            shared=done.shared,
        )

    async def sign_in_link_preview(self, *, ticket: str, opened_by: User | None) -> McpSignInLinkPreview:
        """What a conversation's sign-in link would do, read without starting or storing anything.

        ``opened_by`` is the dashboard identity whose session the link was opened in, when there is one.

        Raises:
            McpConnectionFlowError: the link is forged, altered or expired.
            WorkspaceMcpServerNotFoundError: the server is gone.
            McpConnectionModeError: the server no longer takes each user's own sign-in.
        """
        target = await self._connection_service().read_link(ticket)
        return McpSignInLinkPreview(
            server_name=target.server_name,
            server_host=target.server_host,
            user_label=await self._sign_in_user_label(target),
            opened_as=(opened_by.email or opened_by.full_name or str(opened_by.id)) if opened_by is not None else None,
            other_user=opened_by is not None
            and not McpConnectionService.sign_in_link_is_for(target.user_id, opened_by.id),
        )

    async def _sign_in_user_label(self, target: SignInLinkTarget) -> str:
        """A dashboard account's email, which it cannot choose freely; else a chosen name beside the user's ID."""
        try:
            parsed = uuid.UUID(target.user_id)
        except ValueError:
            parsed = None
        # Only the canonical spelling names a dashboard account, as ``may_connect`` reads it.
        if parsed is not None and str(parsed) == target.user_id:
            identities = await UserRepository(self.db).get_by_ids([parsed])
            if identities and identities[0].email:
                return identities[0].email
            if identities and identities[0].full_name:
                return f"{identities[0].full_name} (user ID {target.user_id})"
        if target.user_alias and target.user_alias != target.user_id:
            return f"{target.user_alias} (user ID {target.user_id})"
        return target.user_id

    async def start_from_sign_in_link(self, *, ticket: str, opened_by: User | None) -> str:
        """Start the sign-in a conversation's link names, once the person confirmed it, and return where to send them.

        A link confirmed inside a dashboard session is bound to it: it must be that identity's own, and
        the sign-in finishes only in that session.

        Raises:
            McpConnectionFlowError: the link is forged, altered or expired.
            McpSignInLinkOtherUserError: ``opened_by`` is not the user the link was made for.
        """
        return await self._connection_service().start_from_link(
            ticket, opened_by=opened_by.id if opened_by is not None else None
        )

    async def my_connections(
        self, *, user: User, workspace_id: uuid.UUID, skip: int = 0, limit: int = 100
    ) -> McpConnectionsPublic:
        """A page of the caller's own connections to this workspace's servers, oldest first, plus the total."""
        resolved_workspace_id = await self._visible_workspace_id(user, workspace_id)
        server_ids = await WorkspaceMcpServerRepository(self.db).ids_in_workspace(resolved_workspace_id)
        connections = McpUserConnectionRepository(self.db)
        count = await connections.count_for_user(str(user.id), server_ids)
        rows = await connections.for_user(str(user.id), server_ids, skip=skip, limit=min(limit, _MAX_LIST_LIMIT))
        return McpConnectionsPublic(
            count=count,
            data=[
                McpConnectionPublic(
                    server_id=row.server_id,
                    connected_at=row.created_at,
                    updated_at=row.updated_at,
                    expires_at=row.expires_at,
                    renewable=row.encrypted_refresh_token is not None,
                )
                for row in rows
            ]
        )

    async def disconnect(self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID) -> None:
        """End the caller's own connection to a server.

        Raises:
            McpConnectionNotFoundError: the caller has no connection to it.
        """
        resolved_workspace_id = await self._visible_workspace_id(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        await self._connection_service().disconnect(server_id=server.id, user_id=str(user.id))

    async def server_connections(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, skip: int = 0, limit: int = 100
    ) -> McpServerConnectionsPublic:
        """A page of who is connected to a server, oldest first, plus the total. Workspace managers only.

        A connection's user is named by the dashboard account it belongs to, when it is one's.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        connections = McpUserConnectionRepository(self.db)
        count = await connections.count_for_server(server.id)
        rows = await connections.for_server(server.id, skip=skip, limit=min(limit, _MAX_LIST_LIMIT))
        identity_ids: dict[str, uuid.UUID] = {}
        for row in rows:
            try:
                parsed = uuid.UUID(row.user_id)
            except ValueError:
                continue
            # Only the canonical spelling names a dashboard account, as ``may_connect`` reads it: another
            # spelling of the same digits is a request-plane user of its own, not that account.
            if str(parsed) == row.user_id:
                identity_ids[row.user_id] = parsed
        identities = {
            identity.id: identity for identity in await UserRepository(self.db).get_by_ids(identity_ids.values())
        }
        data: list[McpServerConnectionPublic] = []
        for row in rows:
            identity = identities.get(identity_ids[row.user_id]) if row.user_id in identity_ids else None
            data.append(
                McpServerConnectionPublic(
                    user_id=row.user_id,
                    email=identity.email if identity is not None else None,
                    full_name=identity.full_name if identity is not None else None,
                    connected_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )
        return McpServerConnectionsPublic(data=data, count=count)

    async def remove_connection(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, user_id: str
    ) -> None:
        """End one user's connection to a server. Workspace managers only.

        Raises:
            McpConnectionNotFoundError: that user has no connection to it.
        """
        resolved_workspace_id = await self._require_management(user, workspace_id)
        server = await self._get_or_404(resolved_workspace_id, server_id)
        await self._connection_service().disconnect(server_id=server.id, user_id=user_id)


__all__ = [
    "MAX_ALLOWED_TOOLS",
    "MAX_MCP_SERVERS_PER_WORKSPACE",
    "MAX_GRANTS_PER_SERVER",
    "McpServerSummary",
    "McpSharedAccountPublic",
    "McpSignInLinkPreview",
    "RequestMcpServers",
    "WorkspaceMcpServerAccess",
    "WorkspaceMcpServerAccessCheck",
    "WorkspaceMcpServerGrantPublic",
    "WorkspaceMcpServerGrantUpdate",
    "WorkspaceMcpServerGrantsCreate",
    "WorkspaceMcpServerKeyAccess",
    "WorkspaceMcpServerMyTools",
    "McpToolSummary",
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
