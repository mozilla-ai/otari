"""ORM tables for gateway-run tools: search credentials, uploaded files, and workspace tool policies."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime
from gateway.models.secret_fields import redact_secret_like_values


def _epoch_seconds(value: datetime | None) -> int | None:
    """Return a UTC epoch from a stored datetime.

    SQLite hands datetimes back naive; ``datetime.timestamp()`` would then read
    them as local time and skew the epoch by the server's UTC offset. Treat a
    naive value as the UTC it was stored as before converting.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp())


def _rfc3339(value: datetime | None) -> str | None:
    """Return an RFC 3339 timestamp from a stored datetime, reading a naive value as UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat().replace("+00:00", "Z")


class SearchToolCredential(Base):
    """A ``POST /v1/search`` tool configured at runtime through the dashboard.

    The database counterpart of a ``search_tools:`` entry in config.yml: it is
    merged over the config-file tools at runtime (see
    ``search_tool_store_service``), with the stored row winning on a name
    collision, exactly as ``ProviderCredential`` does for providers. The API key
    is held encrypted (``secret_box``) and is optional, because a ``searxng``
    backend is normally keyless; ``last4`` is kept in clear only so the UI can
    show which key is set without ever decrypting. Standalone mode only.
    """

    __tablename__ = "search_tool_credentials"

    name: Mapped[str] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column()
    api_base: Mapped[str | None] = mapped_column()
    encrypted_api_key: Mapped[str | None] = mapped_column()
    last4: Mapped[str | None] = mapped_column()
    # Named for its unit; the config-file key it stands in for is plain ``timeout``,
    # and ``to_public_dict`` / the overlay entry both use that name.
    timeout_seconds: Mapped[float | None] = mapped_column()
    options: Mapped[dict[str, Any]] = mapped_column("options", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize for the API. Never includes the secret, only ``last4``.

        ``options`` is free-form, so it is masked by key name in case it holds a
        credential.
        """
        return {
            "name": self.name,
            "provider": self.provider,
            "api_base": self.api_base,
            "last4": self.last4,
            "timeout": self.timeout_seconds,
            "options": redact_secret_like_values(self.options) or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class FileObject(Base):
    """Uploaded file metadata for the OpenAI-compatible /v1/files API.

    The raw bytes live in a pluggable blob backend (see
    gateway.services.file_store); this row holds metadata plus the backend
    ``storage_ref`` used to fetch them. Files are scoped to ``user_id`` for
    tenant isolation and soft-deleted via ``deleted_at``. ``workspace_id`` is a
    second, independent axis: it says which workspace the upload was made in, so
    a key confined to one workspace never reaches another's files even when the
    same user holds keys in both.
    """

    __tablename__ = "file_objects"
    __table_args__ = (
        # The listing's shape: the tenant predicates, then the keyset sort the
        # cursor pages on. Without them every page sorts the user's whole set;
        # the second serves a master-key listing that names no workspace.
        Index(
            "ix_file_objects_user_workspace_created",
            "user_id",
            "workspace_id",
            "created_at",
            "id",
        ),
        Index("ix_file_objects_user_created", "user_id", "created_at", "id"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: f"file-{uuid.uuid4().hex}")
    # Always set to the authenticated user; non-null enforces the user-scoping
    # contract at the schema level. CASCADE removes a user's files on delete.
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    # The workspace this row belongs to; see `APIKey.workspace_id` for why it is
    # NOT NULL and RESTRICT rather than nullable and cascading. Existing rows were
    # backfilled onto the deployment's default workspace, which is also where a
    # master-key upload lands.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column()
    mime_type: Mapped[str] = mapped_column()
    bytes: Mapped[int] = mapped_column()
    purpose: Mapped[str] = mapped_column(default="user_data")
    # Null for a file whose bytes a provider holds; see ``provider`` below.
    storage_ref: Mapped[str | None] = mapped_column(nullable=True)
    # Set when a provider's own sandbox produced the file, naming the any-llm
    # provider whose files API serves its bytes. The row exists so the
    # deployment knows who may read that id: the provider authenticates the
    # deployment's credential, which is coarser than a workspace-scoped key.
    provider: Mapped[str | None] = mapped_column(nullable=True)
    # The configured instance the run dispatched through, whose credential is
    # the one that can read the file back; None means the provider's own entry.
    provider_instance: Mapped[str | None] = mapped_column(nullable=True)
    # The provider's container, for a provider that keys a download on it
    # (OpenAI does; Anthropic's files API takes the id alone).
    provider_container_id: Mapped[str | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None, index=True)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the OpenAI file object shape."""
        return {
            "id": self.id,
            "object": "file",
            "bytes": self.bytes,
            "created_at": _epoch_seconds(self.created_at),
            "expires_at": _epoch_seconds(self.expires_at),
            "filename": self.filename,
            "purpose": self.purpose,
        }

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Convert to the ``FileMetadata`` shape of Anthropic's GA Files API.

        ``expires_at`` is always present and ``None`` for a file kept indefinitely.
        ``downloadable`` is always true, because the gateway serves every stored file's bytes back.
        """
        return {
            "id": self.id,
            "type": "file",
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.bytes,
            "created_at": _rfc3339(self.created_at),
            "expires_at": _rfc3339(self.expires_at),
            "downloadable": True,
        }


class WorkspaceMcpServer(Base):
    """One MCP server a workspace has configured, referenced by id from a request.

    Ported from otari-ai's ``mcp_server`` table (otari#658). A request names
    stored servers with ``mcp_server_ids``; hybrid mode resolves those ids
    through the platform and standalone mode resolves them here, against the
    workspace the request's key belongs to. There is no deployment-wide MCP
    server list for these rows to narrow, which is why MCP is the stated
    exception to the "a workspace row never grants" rule in
    ``src/gateway/AGENTS.md``.

    ``encrypted_token`` holds the server's bearer token, Fernet-encrypted with
    ``OTARI_SECRET_KEY`` (``services/secret_box.py``), the same treatment
    ``ProviderCredential.encrypted_api_key`` gets. Nothing serializes it: the
    public shape carries ``has_token`` and no prefix or suffix of the value,
    because unlike a provider key's ``last4`` there is no operator workflow
    here that needs to tell two tokens apart at a glance.

    ``enabled`` is a workspace-level off switch that keeps the row and its
    token: a disabled server is skipped at resolve rather than refusing the
    request, so a caller whose stored id list outlives one server's
    decommissioning still gets the rest.

    CASCADE: a workspace-owned configuration row means nothing without its
    workspace.
    """

    __tablename__ = "workspace_mcp_servers"
    __table_args__ = (
        # Duplicate names within one workspace are rejected at the database, not
        # only in the service layer, so two concurrent creates cannot both land
        # (otari#658's third Definition-of-Done item). The name is what an
        # operator recognizes a server by and what the tool loop labels its
        # tools with, so collapsing two onto one name would silently hide a
        # server.
        UniqueConstraint("workspace_id", "name", name="uq_workspace_mcp_servers_workspace_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(nullable=False)
    url: Mapped[str] = mapped_column(nullable=False)
    encrypted_token: Mapped[str | None] = mapped_column(Text, default=None)
    purpose_hint: Mapped[str | None] = mapped_column(Text, default=None)
    allowed_tools: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    # ``UtcDateTime`` for the same reason ``WorkspaceBudgetDefault``'s are: these
    # go over the wire and a naive SQLite round-trip would drop the offset.
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class WorkspaceCodeExecutionPolicy(Base):
    """A workspace's policy over the deployment-wide code-execution sandbox.

    The sandbox itself stays deployment-wide (``sandbox_url`` and its
    credential are operator concerns and never move here, see
    ``src/gateway/AGENTS.md``); this row says who on that deployment may ask
    for it and within which limits. Resolved at admission by
    ``prepare_gateway_tools`` and applied to the tool loop, the standalone
    counterpart of the hybrid path's ``/gateway/code-execution/resolve``.

    A row may only *narrow*: ``enabled=False`` refuses the tool for this
    workspace, and the two limits are floored against the values a request
    would otherwise get. No row means no narrowing, which is what keeps a
    deployment that configures nothing behaving as it did (#655/#678).

    ``workspace_id`` is the primary key: a workspace has one policy or none,
    so there is nothing else to identify a row by. It is a real foreign key
    with ``CASCADE``, like ``workspace_budget_defaults``: nothing else names
    the row, so it rides the workspace's own delete.

    ``image`` and ``tools`` reach the same two decisions the hosted
    ``CodeExecutionConfig`` carries (#740). Neither breaks the rule above:
    ``image`` may only name something the deployment's operator has already
    curated into ``sandbox_allowed_session_images``, so a workspace picks from an
    operator's shelf rather than pointing the gateway at an image of its own,
    and ``tools`` may only remove tool kinds from what the sandbox backend
    already serves.
    """

    __tablename__ = "workspace_code_execution_policies"
    __table_args__ = (
        # Both limits are ceilings that get floored into an effective value, so
        # zero or negative is a storage error rather than a stricter policy: it
        # would floor the loop to nothing runnable while reading as configured.
        # The request schemas refuse it first; these are the backstop for a
        # writer that is not the service.
        CheckConstraint(
            "max_iterations IS NULL OR max_iterations > 0",
            name="ck_workspace_code_execution_policies_max_iterations_positive",
        ),
        CheckConstraint(
            "exec_timeout_s IS NULL OR exec_timeout_s > 0",
            name="ck_workspace_code_execution_policies_exec_timeout_positive",
        ),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    # NULL means "no workspace default": the request's own hint, then the
    # deployment's, then the backend's built-in, exactly as today.
    default_purpose_hint: Mapped[str | None] = mapped_column(Text, default=None)
    # Both NULL-able ceilings, applied with ``min`` against what the request
    # would otherwise get, so a value above the deployment ceiling narrows
    # nothing rather than raising it.
    max_iterations: Mapped[int | None] = mapped_column(default=None)
    exec_timeout_s: Mapped[int | None] = mapped_column(default=None)
    # NULL means "no workspace image": whatever the deployment names in
    # ``sandbox_session_image``, and failing that whatever the sandbox backend runs by
    # default, which is what every request got before this column existed.
    # ``String(255)`` rather than ``Text`` to match the hosted column's own
    # bound; an image reference that long is already pathological.
    image: Mapped[str | None] = mapped_column(String(255), default=None)
    # NULL means "no workspace tool allow-list": the backend offers what it
    # offers. A stored list is an intersection, never a union, so it can only
    # take tool kinds away. JSON rather than a child table for the same reason
    # ``WorkspaceWebSearchConfig`` stores its domain lists that way: short, read
    # whole, and nothing queries into it.
    tools: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    # NULL means "no workspace pin": the deployment's ``code_execution_executor``
    # (and, where it leaves room, the request's header) decides who runs a
    # provider-named code-execution declaration. A stored value is a pin the
    # request cannot argue with. One of ``CodeExecutor``'s values; the service
    # refuses anything else, and the column is sized for that vocabulary.
    executor: Mapped[str | None] = mapped_column(String(16), default=None)
    # ``UtcDateTime`` for the same reason ``WorkspaceBudgetDefault`` uses it:
    # these are serialized with ``.isoformat()`` for the dashboard, and a plain
    # ``DateTime(timezone=True)`` round-trips naive on SQLite.
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class WorkspaceWebSearchConfig(Base):
    """A workspace's configuration over the deployment-wide web-search backend.

    The backend itself stays deployment-wide (``web_search_url`` and the
    credential the adapter in front of it holds are operator concerns and never
    move here, see ``src/gateway/AGENTS.md``); this row says which workspaces
    may reach it and how their searches are constrained. Resolved at admission
    by ``prepare_gateway_tools``, the standalone counterpart of the hybrid
    path's ``/gateway/web-search/resolve``.

    A row may only *narrow*: ``enabled=False`` refuses ``otari_web_search`` for
    the workspace, ``max_results`` is floored against what the request asked
    for, ``blocked_domains`` is added to the request's own block-list, and
    ``allowed_domains`` intersects the request's. No row means no narrowing,
    which is what keeps a deployment that configures nothing behaving as it did
    (#655/#678).

    ``workspace_id`` is the primary key, and a real foreign key with
    ``CASCADE``, for the same reasons as :class:`WorkspaceCodeExecutionPolicy`
    next door: one row per workspace, and nothing else names it.

    There is deliberately no ``provider`` column, which the hosted config
    carries: on this deployment the operator picks the backend by pointing
    ``web_search_url`` somewhere, so a provider named here would either be inert
    or would ask the gateway to reach an endpoint the operator did not choose,
    which is the one thing the narrowing rule forbids.
    """

    __tablename__ = "workspace_web_search_configs"
    __table_args__ = (
        # ``max_results`` is floored into an effective value, so zero or less is
        # a storage error rather than a stricter policy: it would ask for a
        # search that can return nothing while reading as configured. The
        # request schema refuses it first; this is the backstop for a writer
        # that is not the service.
        CheckConstraint(
            "max_results IS NULL OR max_results > 0",
            name="ck_workspace_web_search_configs_max_results_positive",
        ),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    # ``server_default`` mirrors the migration so autogenerate sees no drift, and
    # so a row written by anything other than this mapping still gets a value.
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False, server_default=true())
    # NULL means "no workspace ceiling": the request's own value, then the
    # deployment's, then the backend's built-in, exactly as today.
    max_results: Mapped[int | None] = mapped_column(default=None)
    # NULL means "no workspace default": the request's own hint, then the
    # deployment's, then the backend's built-in.
    purpose_hint: Mapped[str | None] = mapped_column(Text, default=None)
    # Two domain lists and an opaque provider bag, stored as JSON for the same
    # reason the hosted table does: they are short, they are read whole, and
    # nothing queries into them. ``JSON`` rather than ``JSONB`` to match every
    # other JSON column here, which has to work on SQLite too.
    allowed_domains: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    blocked_domains: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    # Provider-specific knobs (Tavily's ``search_depth``, say). Opaque here and
    # forwarded to the backend, which is what lets a new provider need no
    # migration; the adapter in front of it whitelists what it understands.
    provider_options: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    # ``UtcDateTime`` for the same reason ``WorkspaceCodeExecutionPolicy`` uses
    # it: these are serialized with ``.isoformat()`` for the dashboard, and a
    # plain ``DateTime(timezone=True)`` round-trips naive on SQLite. The Python
    # default is what every write here uses; ``server_default`` is the backstop
    # for a writer that is not this mapping, matching ``workspace`` itself.
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
