import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gateway.models.base import Base, UtcDateTime
from gateway.models.money import UsdCost
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


class User(Base):
    """User/Customer model for end-user tracking."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    alias: Mapped[str | None] = mapped_column()
    # The spend ledger, exact to the micro-dollar like the ``usage_logs`` rows
    # that sum into it (mozilla-ai/otari#691). As a float it drifted: four
    # completions whose settled costs were each exact left this at
    # 0.6619999999999999, and the drift accumulated across every reconcile until
    # the budget reset.
    spend: Mapped[Decimal] = mapped_column(UsdCost(), default=Decimal(0))
    # In-flight budget held by requests that have passed the budget gate but
    # whose actual cost is not yet known. The effective committed amount is
    # ``spend + reserved``; reservations are reconciled into ``spend`` (actual
    # cost) on success or released on failure. See gateway.services.budget_service.
    reserved: Mapped[Decimal] = mapped_column(UsdCost(), default=Decimal(0), server_default="0")
    # The token and request counters, gated by the same budget's ``token_limit``
    # and ``request_limit`` the way the pair above is gated by ``max_budget``.
    # Each axis names itself rather than extending the bare ``spend``/``reserved``
    # pair, which is USD and predates them.
    current_tokens: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    reserved_tokens: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    current_requests: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    reserved_requests: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    # Indexed: the budgets list groups users by this column to build each budget's
    # usage rollup, so an unindexed FK turns that page into a users table scan.
    budget_id: Mapped[str | None] = mapped_column(ForeignKey("budgets.budget_id"), index=True)
    # Default model access-list every one of this user's keys inherits when the
    # key has no list of its own. null = unrestricted, [] = deny all, else
    # canonical instance:model entries (see services/model_access.py). A key may
    # narrow this default but never broaden it (validated on key write).
    allowed_models: Mapped[list[str] | None] = mapped_column(JSON)
    budget_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_budget_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    blocked: Mapped[bool] = mapped_column(default=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    budget = relationship("Budget", back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", passive_deletes=True)
    usage_logs = relationship("UsageLog", back_populates="user", passive_deletes=True)
    reset_logs = relationship("BudgetResetLog", back_populates="user", passive_deletes=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "user_id": self.user_id,
            "alias": self.alias,
            "spend": self.spend,
            "reserved": self.reserved,
            "budget_id": self.budget_id,
            "allowed_models": self.allowed_models,
            "budget_started_at": self.budget_started_at.isoformat() if self.budget_started_at else None,
            "next_budget_reset_at": self.next_budget_reset_at.isoformat() if self.next_budget_reset_at else None,
            "blocked": self.blocked,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata_,
        }


class RoutingPolicy(Base):
    """A named routing policy, writable through the API.

    The runtime counterpart of the ``routing.policies`` block in config.yml. The
    spec is stored as JSON rather than as columns because it is a nested,
    versioned document (``select`` entries with conditions, ``on_failure``,
    guardrails); flattening it into columns would mean a migration per
    schema addition and would still need JSON for the conditions. It is validated
    against :class:`gateway.models.routing.PolicySpec` on write and again on load,
    so a row that predates a schema change surfaces as a startup warning rather
    than as a request-time crash.

    Scoping mirrors :class:`ModelAlias` exactly, workspace included, and so does
    the two-constraint uniqueness (SQLite and PostgreSQL both treat NULLs as
    distinct in a unique index, so the composite constraint cannot keep one
    *workspace-wide* row per name). A policy and an alias are the same concept at
    different complexities, so it would be strange for their scoping rules to
    differ.
    """

    __tablename__ = "routing_policies"
    __table_args__ = (
        # Workspace-scoped for the same reason, and on the same precondition, as
        # :class:`ModelAlias`: ``services/policy_store`` keys its cache by
        # workspace, so two workspaces holding a "fast" policy each resolve their
        # own rather than one shadowing the other.
        UniqueConstraint("workspace_id", "name", "user_id", name="uq_routing_policies_workspace_name_user"),
        Index(
            "uq_routing_policies_workspace_global_name",
            "workspace_id",
            "name",
            unique=True,
            sqlite_where=text("user_id IS NULL"),
            postgresql_where=text("user_id IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column()
    spec: Mapped[dict[str, Any]] = mapped_column(JSON)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    # The workspace this row belongs to; see `APIKey.workspace_id` for why.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DashboardSession(Base):
    """A server-side admin-dashboard sign-in session, held by one identity.

    Minted when an operator signs in to the dashboard with the master key: the
    browser holds only an opaque token in an HttpOnly cookie and this table
    stores the token's SHA-256 hash, so neither the master key nor a usable
    session credential is ever persisted in JS-readable storage. Sessions
    expire on a TTL and are revoked on sign-out and on master-key rotation.

    ``user_id`` is what lets a session resolve a caller rather than only prove
    that the master key was presented once. It names a tenancy identity
    (`models.tenancy.User`), whose ``active_organization_id`` is the
    organization the session acts in, so a tenancy surface reads its scope off
    the session. Master-key sign-in binds the session to the deployment's
    bootstrap operator; a per-user sign-in flow binds it to whoever
    authenticated.

    NOT NULL on purpose: a session that names nobody cannot answer "who is
    calling", which is the whole point of the column, and the migration that
    added it bound existing sessions to that same bootstrap operator. CASCADE
    on the foreign key, so deleting an identity revokes its sessions rather
    than leaving a live cookie pointing at a row that is gone.
    """

    __tablename__ = "dashboard_sessions"

    token_hash: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


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

        ``options`` is masked by key name for the reason ``ProviderCredential``
        gives: it is free-form backend configuration, so a second
        credential an operator put there is not echoed back either.
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


class UsageLog(Base):
    """Usage log model for tracking API requests."""

    __tablename__ = "usage_logs"
    __table_args__ = (
        Index("ix_usage_logs_user_id_timestamp", "user_id", "timestamp"),
        # Supports the activity-log viewer's primary "show errors, newest-first"
        # query. status is low-cardinality; model is high-cardinality and left
        # unindexed on purpose.
        Index("ix_usage_logs_status_timestamp", "status", "timestamp"),
        # Supports the setup guide's two questions about one workspace: has any
        # request in it ever succeeded (oldest first), and what did the last one
        # do (newest first). Both filter a workspace, a source and a status and
        # then order by time, which the workspace-only and status-first indexes
        # above can each answer only halfway: on a deployment with real traffic
        # the guide would otherwise scan the workspace's rows on every dashboard
        # load, and where usage is imported as well most of those rows are the
        # wrong source anyway. Equality columns first, the ordering column last.
        Index(
            "ix_usage_logs_workspace_source_status_timestamp",
            "workspace_id",
            "source",
            "status",
            "timestamp",
        ),
        # Idempotency for imported usage: re-submitting the same (source,
        # source_event_id) must not create a second row. Gateway-originated rows
        # keep source_event_id NULL, and SQL treats NULLs as distinct on both
        # SQLite and Postgres, so many (gateway, NULL) rows coexist freely.
        UniqueConstraint("source", "source_event_id", name="uq_usage_logs_source_event"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    # The workspace this row belongs to; see `APIKey.workspace_id` for why.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)

    model: Mapped[str] = mapped_column()
    provider: Mapped[str | None] = mapped_column()
    endpoint: Mapped[str] = mapped_column()

    # Provenance. "gateway" for requests Otari served itself; a source slug (e.g.
    # "claude_code") for usage imported through POST /v1/usage/external-events. A row
    # backfilled from hosted history keeps its origin's slug behind a legacy prefix
    # ("otari-ai:gateway", "otari-ai:claude_code"), so asking whether this deployment
    # served a row means asking about the slug behind that prefix: core/usage_source.
    # source_event_id is the upstream event id used for idempotent import (NULL for
    # gateway rows); source_label carries optional session/project attribution.
    source: Mapped[str] = mapped_column(default="gateway", index=True)
    source_event_id: Mapped[str | None] = mapped_column()
    source_label: Mapped[str | None] = mapped_column()
    # Whether this row's cost participates in budget enforcement. True for normal
    # gateway rows; false for imported usage and for rows from keys flagged
    # exclude_from_budget. False rows are recorded (and appear in cost analytics)
    # but their cost is never written to User.spend.
    counts_toward_budget: Mapped[bool] = mapped_column(default=True)

    prompt_tokens: Mapped[int | None] = mapped_column()
    completion_tokens: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()
    cache_read_tokens: Mapped[int | None] = mapped_column()
    cache_write_tokens: Mapped[int | None] = mapped_column()
    cache_write_1h_tokens: Mapped[int | None] = mapped_column()
    # Which cached-token convention the counts above were reported under: True
    # when the cache buckets are already inside ``prompt_tokens`` (OpenAI shape),
    # False when they are additive to it (Anthropic / Claude Code shape). Written
    # by settlement from ``GatewayUsage.cache_tokens_in_prompt`` and by the
    # external-usage ingest from the value the submitter sent, so a row can be
    # repriced under the convention it was recorded with rather than one inferred
    # from the numbers, which cannot tell the two apart.
    #
    # Nullable, and deliberately not defaulted: "not recorded" and "inclusive" are
    # different answers. Rows written before this column existed are NULL, and
    # repricing falls back to recovering the convention from ``billing_meters``
    # for exactly those (see ``usage_admin_service._row_cache_tokens_included``).
    # A default would make every historical row claim a convention nothing
    # checked, and mis-price the half that were the other one.
    cache_tokens_in_prompt: Mapped[bool | None] = mapped_column()
    billing_meters: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    pricing_breakdown: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    # The settled amount, and the accounting truth for this row
    # (mozilla-ai/otari-ai#1751). Exact to the micro-dollar; see
    # ``models/money.py`` for what that costs on each engine.
    cost: Mapped[Decimal | None] = mapped_column(UsdCost())

    # Why ``cost`` is the amount it is, which the row cannot re-derive on its own:
    # ``pricing_source`` names the price list that settled it ("organization",
    # "managed", "genai_prices"), ``pricing_reference`` identifies the entry in it
    # (a pricing row's id, or a ``provider:model`` key), ``pricing_effective_at``
    # is when that rate took effect, and ``pricing_version`` pins the revision of
    # the list. ``calculated_at`` is when the amount was priced, which is not
    # ``timestamp`` (when the request ran): usage settled or repriced later moves
    # the two apart.
    #
    # All nullable with no backfill. The gateway's own settlement does not record
    # provenance, so these are written by the hosted-usage backfill
    # (mozilla-ai/otari-ai#1798) from the platform's ``gateway_usage_settlement``
    # row, and null reads correctly as "not recorded". The lengths mirror that
    # table's columns rather than this file's usual unbounded strings, so a value
    # copied across always fits.
    #
    # ``pricing_source`` speaks the platform's settlement vocabulary, the values
    # ``_platform.SettledCost.pricing_source`` already carries on the hybrid wire
    # (echoed to callers as ``usage.pricing_source``). It is not the same field as
    # the one on a listed model in ``api/routes/models.py`` ("configured",
    # "default", "dynamic", "none"), which says where a price list entry came from
    # in this deployment rather than what settled one row's amount.
    pricing_source: Mapped[str | None] = mapped_column(String(32))
    pricing_reference: Mapped[str | None] = mapped_column(String(511))
    pricing_effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pricing_version: Mapped[str | None] = mapped_column(String(255))
    calculated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # "success", "error", or "absorbed". ``absorbed`` is a failed attempt that a
    # routing policy recovered from by trying the next candidate: the request
    # itself succeeded (or failed on a later attempt), so counting it as an error
    # would make a working fallback chain look like an outage. Every error metric
    # in the product counts ``status == "error"`` exactly, and ``request_count``
    # excludes absorbed rows, because a request that took two attempts is still one
    # request.
    status: Mapped[str] = mapped_column()
    error_message: Mapped[str | None] = mapped_column()

    # Routing attribution. All nullable: a request that named a plain model was not
    # routed through a policy, and null reads correctly as exactly that.
    #
    # `policy_name` is the name the caller sent. `selection_reason` says why this
    # candidate was chosen ("default", "condition:<keys>", "on_failure",
    # "router:<name>"). `attempt_position` and `attempt_count` locate the row in
    # the plan, so "served on attempt 2 of 3" is a query rather than a log grep.
    # `request_group_id` ties a request's rows together, which is what makes the
    # absorbed attempts findable from the row that served.
    policy_name: Mapped[str | None] = mapped_column(index=True)
    selection_reason: Mapped[str | None] = mapped_column()
    attempt_position: Mapped[int | None] = mapped_column()
    attempt_count: Mapped[int | None] = mapped_column()
    request_group_id: Mapped[str | None] = mapped_column(index=True)

    # HTTP status that classifies a failure, so failures can be grouped with a
    # GROUP BY instead of substring-matching provider-specific error prose. It is
    # the status the provider returned when it sent one (an upstream 401 stays
    # visible as a credential fault even though the caller sees the generic 502
    # that keeps gateway config out of the response), otherwise the gateway's own
    # rejection or classification code (402 missing pricing, 422 tool-loop cap,
    # 504 timeout, 502 unreachable). Nullable: historical rows predate the column,
    # a successful request has no failure to classify, and some failures carry no
    # HTTP status at all (e.g. a stream that ended without usage data).
    status_code: Mapped[int | None] = mapped_column()

    # Total server-side wall-clock for the request, in milliseconds. Nullable:
    # historical rows predate the column, and some write paths (batch jobs,
    # provider-never-reached rejections) have no meaningful request duration.
    latency_ms: Mapped[int | None] = mapped_column()

    # Milliseconds from request start to the first streamed chunk. Nullable:
    # non-streaming requests have no first chunk, historical rows predate the
    # column, and a stream that failed before yielding anything never reached one.
    #
    # ``started_at`` is taken in the handler preamble, so on a routing plan the
    # serving row's value also carries every earlier attempt's setup time.
    # Nothing in the column says so; a percentile keyed by the serving model
    # attributes failover time to the model that actually served.
    #
    # Hybrid (platform-fallback) streams never write this column at all: every
    # settlement callback in build_streaming_response returns before reaching
    # log_usage on that path, and run_streaming_with_fallback passes db=None.
    ttft_ms: Mapped[int | None] = mapped_column()

    api_key = relationship("APIKey", back_populates="usage_logs")
    user = relationship("User", back_populates="usage_logs")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "api_key_id": self.api_key_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model": self.model,
            "endpoint": self.endpoint,
            "source": self.source,
            "source_label": self.source_label,
            "counts_toward_budget": self.counts_toward_budget,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_write_1h_tokens": self.cache_write_1h_tokens,
            "cache_tokens_in_prompt": self.cache_tokens_in_prompt,
            "billing_meters": self.billing_meters,
            "pricing_breakdown": self.pricing_breakdown,
            "cost": self.cost,
            "status": self.status,
            "error_message": self.error_message,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "policy_name": self.policy_name,
            "selection_reason": self.selection_reason,
            "attempt_position": self.attempt_position,
            "attempt_count": self.attempt_count,
            "request_group_id": self.request_group_id,
        }


class AgentTelemetry(Base):
    """Content-free outcome metrics and behavioral events from coding agents."""

    __tablename__ = "agent_telemetry"
    __table_args__ = (
        UniqueConstraint("source", "dedup_key", name="uq_agent_telemetry_source_dedup"),
        Index("ix_agent_telemetry_user_id_timestamp", "user_id", "timestamp"),
        # Read-time cumulative-to-delta derivation orders one series' points by time.
        Index("ix_agent_telemetry_series_timestamp", "series_key", "timestamp"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)
    name: Mapped[str] = mapped_column()
    tool_name: Mapped[str | None] = mapped_column()
    decision: Mapped[str | None] = mapped_column()
    success: Mapped[bool | None] = mapped_column()
    duration_ms: Mapped[int | None] = mapped_column()
    status_code: Mapped[int | None] = mapped_column()
    prompt_length: Mapped[int | None] = mapped_column()
    source: Mapped[str] = mapped_column(index=True)
    session_label: Mapped[str | None] = mapped_column()
    dedup_key: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    # Outcome-metric columns. Populated only on a metric row (``kind="metric"``),
    # NULL on a behavioral one, which is the inverse of the allow-list columns
    # above. ``value`` is stored exactly as OTLP reported it (a running total or
    # an increment, per ``temporality``); the read endpoints do the delta
    # arithmetic, so nothing is normalized at ingest. ``series_key`` is the pure
    # OTLP series identity (name plus attributes), which is what makes a
    # dimensioned metric two series rather than one.
    kind: Mapped[str | None] = mapped_column()
    value: Mapped[float | None] = mapped_column()
    temporality: Mapped[str | None] = mapped_column()
    series_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    series_key: Mapped[str | None] = mapped_column()


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
    storage_ref: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
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


class RoutingMemory(Base):
    """One record per scored example: a prompt embedding plus the quality each
    candidate model earned on it.

    The kNN router (:mod:`gateway.services.routing.knn`) retrieves the nearest
    neighbors of an incoming request's task embedding within one user's records
    and votes on the cheapest candidate that is still good enough. One record is
    one example (one prompt), so the vote is over distinct prompts; ``qualities``
    maps each model to its ``[0, 1]`` score for this prompt, keyed on canonical
    ``instance:model`` so a candidate's spelling never decides whether it matches
    (the router canonicalizes what it reads, so older rows keyed on another
    spelling still match). Records are written by the preference-collection flow,
    never by live traffic (passive learning is a fast-follow).

    Vectors are stored as a JSON list of floats for SQLite/PostgreSQL
    portability and scanned linearly in Python. That holds into the low thousands
    of records per user (the ``router_max_records_per_user`` cap); larger pools
    need an indexed vector store.
    ``embedding_model`` tags each row so changing the embedding model invalidates
    stale vectors instead of mixing incomparable spaces.

    Scoped by ``user_id``, which is the identity the request is routed and billed
    under, so one user's examples never steer another's traffic. CASCADE: the
    records are derived training data, worthless once the user is gone.
    ``workspace_id`` narrows that further: the router reads one (user, workspace)
    partition, so a user who holds keys in two workspaces does not have one
    workspace's labels steering the other's traffic.
    """

    __tablename__ = "routing_memory"
    __table_args__ = (
        # Every read filters on the workspace as well as the user, so the
        # workspace leads: the same three shapes, one partition narrower.
        Index("ix_routing_memory_workspace_user_model", "workspace_id", "user_id", "embedding_model"),
        Index("ix_routing_memory_workspace_user_created", "workspace_id", "user_id", "created_at"),
        # A task-scoped read filters on all four; without this it walks every
        # record the user has for the embedding model before partitioning.
        Index(
            "ix_routing_memory_workspace_user_model_task",
            "workspace_id",
            "user_id",
            "embedding_model",
            "task_id",
        ),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    # The workspace this row belongs to; see `APIKey.workspace_id` for why.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    embedding_model: Mapped[str] = mapped_column()
    embedding: Mapped[list[float]] = mapped_column(JSON)
    qualities: Mapped[dict[str, float]] = mapped_column(JSON)
    task_id: Mapped[str | None] = mapped_column(default=None, index=True)
    label_source: Mapped[str] = mapped_column(default="human")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary.

        The embedding itself is deliberately left out: it is thousands of floats
        that no management surface renders, and the prompt it came from is on the
        :class:`RouterPreference` audit row.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "workspace_id": str(self.workspace_id),
            "embedding_model": self.embedding_model,
            "qualities": self.qualities,
            "task_id": self.task_id,
            "label_source": self.label_source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class RouterPreference(Base):
    """An audit record of one preference-collection scoring.

    Each ``/v1/routing/preferences/rank`` submission writes one row here for
    provenance plus one :class:`RoutingMemory` row. The routing-memory row keeps
    only the embedding, so this is where the prompt text and the raw per-model
    scores live: enough to recompute the memory if the scoring changes, and to
    tell a human label from a judge's.

    ``workspace_id`` matches the :class:`RoutingMemory` row written beside it, so
    the audit trail partitions exactly the way the training data does.
    """

    __tablename__ = "router_preferences"
    __table_args__ = (
        Index("ix_router_preferences_workspace_user_created", "workspace_id", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    # The workspace this row belongs to; see `APIKey.workspace_id` for why.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    prompt: Mapped[str] = mapped_column()
    task_id: Mapped[str | None] = mapped_column(default=None)
    scores: Mapped[dict[str, float]] = mapped_column(JSON)
    label_source: Mapped[str] = mapped_column(default="human")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "workspace_id": str(self.workspace_id),
            "prompt": self.prompt,
            "task_id": self.task_id,
            "scores": self.scores,
            "label_source": self.label_source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class WorkspaceActivationState(Base):
    """What the dashboard's first-request setup guide remembers about a workspace.

    The guide walks a workspace from "no traffic" to its first successful
    request (`services/tenancy/workspace_activation_service.py`). Only what
    cannot be observed elsewhere is stored here: whether someone dismissed it,
    when it last handed out a key, and which key that was. Whether the workspace
    has *activated* is deliberately not a column, because ``usage_logs`` already
    records it: the first successful gateway request in the workspace is the
    evidence, so there is no second copy of it to backfill or to disagree with
    the Activity page.

    Ported from the platform's ``workspace_activation_state`` /
    ``workspace_activation_experience_state`` pair
    (`otari-ai` `backend/app/models/workspace_activation.py`), which does carry
    the attempt telemetry as columns, because its usage pipeline is asynchronous
    and crosses services. Here the usage row is written by this process into this
    database, so the derivation is exact.

    One row per workspace, not per workspace and viewer: the guide is about a
    workspace's first request, so dismissing it says "this workspace is set up,
    stop offering the guide" for everyone who can manage it.
    """

    __tablename__ = "workspace_activation_state"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    # When the guide first and last minted an API key for this workspace. The
    # first is what an operator reads as "when was this offered"; the last is
    # what makes a rotation visible next to the key it rotated.
    first_presented_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    last_presented_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    # Set by Skip, and permanent: the guide is a first-run offer, so a workspace
    # that turned it down is not asked again on the next page load.
    dismissed_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    # The key the guide issued, rotated in place on each presentation so a
    # workspace collects one "Setup guide" key rather than one per page load.
    # ``SET NULL`` because deleting that key from the Keys page is a legitimate
    # thing to do, and it must not take this row (or the dismissal on it) with it.
    api_key_id: Mapped[str | None] = mapped_column(
        ForeignKey("api_keys.id", ondelete="SET NULL"), default=None, index=True
    )
    # ``UtcDateTime`` rather than ``DateTime(timezone=True)`` for the same reason
    # ``WorkspaceBudgetDefault`` above uses it: on SQLite, which this edition
    # ships by default, the plain type round-trips naive and a browser would read
    # the value as local time.
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


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

    CASCADE, not the ``RESTRICT`` the request-plane tables above use: this is a
    workspace-owned configuration row, like ``workspace_budget_defaults``, with
    no meaning once its workspace is gone.
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
