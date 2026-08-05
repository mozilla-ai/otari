import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


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


class APIKey(Base):
    """API Key model for authentication and authorization."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(primary_key=True)
    key_hash: Mapped[str] = mapped_column(unique=True, index=True)
    # Display-only leading characters of the plaintext key, kept so the dashboard can
    # recognize a key after its one-time reveal. Nullable: keys minted before this
    # column existed cannot be back-filled (the plaintext is unrecoverable).
    key_prefix: Mapped[str | None] = mapped_column()
    key_name: Mapped[str | None] = mapped_column()
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(default=True)
    # When true, requests authenticated with this key are logged with their computed
    # cost but skip budget reservation/reconciliation: their spend is never written to
    # User.spend and never gates enforcement. Default false keeps every existing key
    # (and all keys minted before this column) on the normal enforced path.
    exclude_from_budget: Mapped[bool] = mapped_column(default=False)
    # Per-key override of the deployment-wide ``reject_user_mismatch`` setting.
    # NULL = inherit (the default, and where every key predating this column
    # stays), True = always reject a request naming a different ``user``, False =
    # always accept it. The override only decides the 403: spend binds to this
    # key's own user either way, so the client value stays a provider-side tag.
    # False is for clients whose ``user`` is telemetry rather than an identity
    # (Claude Code sends a per-session JSON blob); True lets a deployment that
    # relaxed the check globally keep an individual key strict.
    reject_user_mismatch: Mapped[bool | None] = mapped_column(default=None)
    # Per-key model allow-list. NULL = unrestricted (default; every key predating
    # this column stays unrestricted), [] = deny all, a list = canonical
    # instance:model entries (with instance:* / instance:prefix* wildcards).
    allowed_models: Mapped[list[str] | None] = mapped_column(JSON)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    user = relationship("User", back_populates="api_keys")
    usage_logs = relationship("UsageLog", back_populates="api_key", passive_deletes=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "key_prefix": self.key_prefix,
            "key_name": self.key_name,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "exclude_from_budget": self.exclude_from_budget,
            "reject_user_mismatch": self.reject_user_mismatch,
            "allowed_models": self.allowed_models,
            "metadata": self.metadata_,
        }


class Budget(Base):
    """Budget model for spending limits."""

    __tablename__ = "budgets"

    budget_id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str | None] = mapped_column(default=None)
    max_budget: Mapped[float | None] = mapped_column()
    budget_duration_sec: Mapped[int | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    users = relationship("User", back_populates="budget")
    reset_logs = relationship("BudgetResetLog", back_populates="budget")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "budget_id": self.budget_id,
            "name": self.name,
            "max_budget": self.max_budget,
            "budget_duration_sec": self.budget_duration_sec,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class User(Base):
    """User/Customer model for end-user tracking."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    alias: Mapped[str | None] = mapped_column()
    spend: Mapped[float] = mapped_column(default=0.0)
    # In-flight budget held by requests that have passed the budget gate but
    # whose actual cost is not yet known. The effective committed amount is
    # ``spend + reserved``; reservations are reconciled into ``spend`` (actual
    # cost) on success or released on failure. See gateway.services.budget_service.
    reserved: Mapped[float] = mapped_column(default=0.0, server_default="0")
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


class ModelAlias(Base):
    """A display name that resolves to a real model selector.

    The runtime counterpart of the ``aliases:`` block in config.yml: same
    meaning, but writable through the API. Pricing, budgets, and usage all key
    on the resolved target, so nothing here is billed against ``name``.

    ``user_id`` is the scope. ``NULL`` means the alias is global (every caller
    sees it), which is what every row predating this column is. A non-null
    ``user_id`` scopes the alias to that user, so two users can point the same
    display name at different models, and a user-scoped row shadows a global one
    of the same name for that user only.

    Uniqueness needs two constraints rather than one because SQLite and
    PostgreSQL both treat NULLs as distinct in a unique index: the composite
    constraint keeps one row per (name, user), and the partial index keeps one
    global row per name (which the composite one cannot, its ``user_id`` being
    NULL). The surrogate ``id`` exists only because the natural key contains a
    nullable column, which a primary key cannot.
    """

    __tablename__ = "model_aliases"
    __table_args__ = (
        UniqueConstraint("name", "user_id", name="uq_model_aliases_name_user"),
        Index(
            "uq_model_aliases_global_name",
            "name",
            unique=True,
            sqlite_where=text("user_id IS NULL"),
            postgresql_where=text("user_id IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    # No index of its own: uq_model_aliases_name_user already leads with `name`,
    # and uq_model_aliases_global_name indexes it again for the global rows. A
    # third copy would be paid for on every write to serve reads that mostly do
    # not happen, since resolution goes through the process-wide alias cache.
    name: Mapped[str] = mapped_column()
    target: Mapped[str] = mapped_column()
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
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

    Scoping mirrors :class:`ModelAlias` exactly, including the two-constraint
    uniqueness (SQLite and PostgreSQL both treat NULLs as distinct in a unique
    index, so the composite constraint cannot keep one *global* row per name).
    A policy and an alias are the same concept at different complexities, so it
    would be strange for their scoping rules to differ.
    """

    __tablename__ = "routing_policies"
    __table_args__ = (
        UniqueConstraint("name", "user_id", name="uq_routing_policies_name_user"),
        Index(
            "uq_routing_policies_global_name",
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


class RuntimeSetting(Base):
    """A persisted override for a runtime-toggleable config flag.

    A small key/value store for the handful of settings the dashboard can flip
    at runtime (model discovery, default pricing). When a key is present it wins
    over the config-file/env value and is applied on startup; when absent the
    config value stands. The value is stored as a string ("true"/"false") so the
    table can hold future non-boolean settings without a schema change.
    """

    __tablename__ = "runtime_settings"

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class DashboardSession(Base):
    """A server-side admin-dashboard sign-in session.

    Minted when an operator signs in to the dashboard with the master key: the
    browser holds only an opaque token in an HttpOnly cookie and this table
    stores the token's SHA-256 hash, so neither the master key nor a usable
    session credential is ever persisted in JS-readable storage. Sessions
    expire on a TTL and are revoked on sign-out and on master-key rotation.
    """

    __tablename__ = "dashboard_sessions"

    token_hash: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class PricingSnapshot(Base):
    """An approved, source-tagged upstream pricing catalog."""

    __tablename__ = "pricing_snapshots"

    source: Mapped[str] = mapped_column(primary_key=True)
    snapshot: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class ProviderCredential(Base):
    """A provider instance configured at runtime through the dashboard.

    The database counterpart of a ``providers:`` entry in config.yml: it is
    merged over the config-file providers at runtime (see
    ``provider_store_service``), with the stored row winning on an instance-name
    collision. The API key is held encrypted (``secret_box``); ``last4`` is kept
    in clear only so the UI can show which key is set without ever decrypting.
    Standalone mode only, never used in the hybrid platform path.
    """

    __tablename__ = "provider_credentials"

    instance: Mapped[str] = mapped_column(primary_key=True)
    provider_type: Mapped[str | None] = mapped_column()
    api_base: Mapped[str | None] = mapped_column()
    encrypted_api_key: Mapped[str | None] = mapped_column()
    last4: Mapped[str | None] = mapped_column()
    client_args: Mapped[dict[str, Any]] = mapped_column("client_args", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize for the API. Never includes the secret, only ``last4``."""
        return {
            "instance": self.instance,
            "provider_type": self.provider_type,
            "api_base": self.api_base,
            "last4": self.last4,
            "client_args": self.client_args or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ModelPricing(Base):
    """Model pricing configuration."""

    __tablename__ = "model_pricing"

    model_key: Mapped[str] = mapped_column(primary_key=True)
    effective_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        default=lambda: datetime.now(UTC),
    )
    input_price_per_million: Mapped[float] = mapped_column()
    output_price_per_million: Mapped[float] = mapped_column()
    # Nullable: providers without prompt caching (or models without a
    # discounted cache rate) leave these unset. When set, the cost
    # calculation prices cache_read_tokens / cache_write_tokens at these
    # per-million-token rates, following the provider inclusion convention
    # (see log_usage in _pipeline.py).
    cache_read_price_per_million: Mapped[float | None] = mapped_column(nullable=True)
    cache_write_price_per_million: Mapped[float | None] = mapped_column(nullable=True)
    cache_write_1h_price_per_million: Mapped[float | None] = mapped_column(nullable=True)
    # Ordered threshold rules. Each rule applies its supplied rates to the
    # entire request once ``total_input_tokens`` reaches ``min_input_tokens``.
    pricing_tiers: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "model_key": self.model_key,
            "effective_at": self.effective_at.isoformat() if self.effective_at else None,
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "cache_read_price_per_million": self.cache_read_price_per_million,
            "cache_write_price_per_million": self.cache_write_price_per_million,
            "cache_write_1h_price_per_million": self.cache_write_1h_price_per_million,
            "pricing_tiers": self.pricing_tiers,
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
        # Idempotency for imported usage: re-submitting the same (source,
        # source_event_id) must not create a second row. Gateway-originated rows
        # keep source_event_id NULL, and SQL treats NULLs as distinct on both
        # SQLite and Postgres, so many (gateway, NULL) rows coexist freely.
        UniqueConstraint("source", "source_event_id", name="uq_usage_logs_source_event"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)

    model: Mapped[str] = mapped_column()
    provider: Mapped[str | None] = mapped_column()
    endpoint: Mapped[str] = mapped_column()

    # Provenance. "gateway" for requests Otari served itself; a source slug (e.g.
    # "claude_code") for usage imported through POST /v1/usage/external-events.
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
    billing_meters: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    pricing_breakdown: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    cost: Mapped[float | None] = mapped_column()

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


class FileObject(Base):
    """Uploaded file metadata for the OpenAI-compatible /v1/files API.

    The raw bytes live in a pluggable blob backend (see
    gateway.services.file_store); this row holds metadata plus the backend
    ``storage_ref`` used to fetch them. Files are scoped to ``user_id`` for
    tenant isolation and soft-deleted via ``deleted_at``.
    """

    __tablename__ = "file_objects"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: f"file-{uuid.uuid4().hex}")
    # Always set to the authenticated user; non-null enforces the user-scoping
    # contract at the schema level. CASCADE removes a user's files on delete.
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column()
    mime_type: Mapped[str] = mapped_column()
    bytes: Mapped[int] = mapped_column()
    purpose: Mapped[str] = mapped_column(default="user_data")
    storage_ref: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)
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


class BatchRecord(Base):
    """Ownership and accounting record for an asynchronous batch job.

    Written at creation time so results accounting can be made idempotent (bill
    and log once, on the first completed retrieval), the batch cost can be folded
    into ``users.spend``, and ownership can be enforced without depending on the
    provider round-tripping the ``otari_user_id`` metadata marker. Batches created
    before this table existed carry no record and fall back to the
    metadata-anchored ownership path in ``api/routes/batches.py``.
    """

    __tablename__ = "batches"

    # Provider-assigned batch id (globally unique per provider), used as the
    # lookup key on retrieve/cancel/results.
    id: Mapped[str] = mapped_column(primary_key=True)
    # Instance/provider name the batch was created against (echoed to clients).
    provider: Mapped[str] = mapped_column()
    # Billed owner, stamped from the authenticated principal at creation. Non-null:
    # this record is the strict ownership anchor, so it must always name an owner.
    # CASCADE: deleting the user drops the ownership record (the user's keys are
    # gone too, and usage_logs remain the billing history).
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    # SET NULL: a key may be revoked while its batch is still in flight.
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    model: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    # NULL until the first completed results retrieval accounts the batch; the
    # atomic NULL -> now transition is the idempotency gate for billing/logging.
    results_accounted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RoutingMemory(Base):
    """One record per scored example: a prompt embedding plus the quality each
    candidate model earned on it.

    The kNN router (:mod:`gateway.services.knn_router`) retrieves the nearest
    neighbors of an incoming request's task embedding within one user's records
    and votes on the cheapest candidate that is still good enough. One record is
    one example (one prompt), so the vote is over distinct prompts; ``qualities``
    maps each model to its ``[0, 1]`` score for this prompt. Records are written
    by the preference-collection flow, never by live traffic (passive learning is
    a fast-follow).

    Vectors are stored as a JSON list of floats for SQLite/PostgreSQL
    portability and scanned linearly in Python. That holds into the low thousands
    of records per user (the ``router_max_records_per_user`` cap); pgvector or an
    ANN index is the documented next step past that (`docs/routing-scaling.md`).
    ``embedding_model`` tags each row so changing the embedding model invalidates
    stale vectors instead of mixing incomparable spaces.

    Scoped by ``user_id``, which is the identity the request is routed and billed
    under, so one user's examples never steer another's traffic. CASCADE: the
    records are derived training data, worthless once the user is gone.
    """

    __tablename__ = "routing_memory"
    __table_args__ = (
        Index("ix_routing_memory_user_model", "user_id", "embedding_model"),
        Index("ix_routing_memory_user_created", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
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
    """

    __tablename__ = "router_preferences"
    __table_args__ = (Index("ix_router_preferences_user_created", "user_id", "created_at"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
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
            "prompt": self.prompt,
            "task_id": self.task_id,
            "scores": self.scores,
            "label_source": self.label_source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BudgetResetLog(Base):
    """Budget reset log model for tracking budget resets."""

    __tablename__ = "budget_reset_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"), index=True)
    # Indexed: the reset-log drill-down filters on this column, and the table only
    # grows, so an unindexed FK degrades that endpoint to a full scan over time.
    budget_id: Mapped[str] = mapped_column(ForeignKey("budgets.budget_id"), index=True)
    previous_spend: Mapped[float] = mapped_column()
    reset_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    next_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user = relationship("User", back_populates="reset_logs")
    budget = relationship("Budget", back_populates="reset_logs")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "budget_id": self.budget_id,
            "previous_spend": self.previous_spend,
            "reset_at": self.reset_at.isoformat() if self.reset_at else None,
            "next_reset_at": self.next_reset_at.isoformat() if self.next_reset_at else None,
        }
