import uuid
from datetime import UTC, datetime
from decimal import Decimal
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
    text,
    true,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlmodel import SQLModel

# The timezone-aware timestamp type the tenancy tables already use. Imported
# rather than redefined: it exists because the engines disagree about
# ``timezone=True``, and two copies of that reasoning would drift.
from gateway.models.money import UsdCost, UsdRate
from gateway.models.tenancy import UtcDateTime


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models.

    Shares ``SQLModel.metadata`` so the reconciled control plane's SQLModel
    tables (`gateway.models.tenancy`) and the gateway's own declarative tables
    land in one collection. That is what lets Alembic keep a single
    ``target_metadata``, and ``create_all``/``drop_all`` cover the whole schema,
    without either style having to know the other exists. The two classes keep
    separate declarative *registries*, so a same-named model on either side
    (``User``, during the strangle) resolves unambiguously.
    """

    metadata = SQLModel.metadata


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
    # The workspace this row belongs to, and the canonical note for the three
    # tables below that carry the same column. NOT NULL: a workspace is the unit
    # the dashboard scopes by, so "no workspace" is never a real state, only an
    # unmigrated one. Existing rows were backfilled onto the deployment's default
    # workspace, which the same migration seeds when tenancy was never touched.
    # RESTRICT rather than cascade: deleting a workspace must not silently take
    # its keys, usage, aliases and policies with it. Which workspace a write
    # lands in is resolved in `services/workspace_scope.py`.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
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
    # Per-key override of the deployment-wide ``capture_agent_telemetry`` setting.
    # NULL = inherit (the default), True = always store behavioral events from
    # this key, False = always discard them. Usage capture/billing is unaffected
    # either way; this only gates the content-free agent_telemetry row.
    capture_agent_telemetry: Mapped[bool | None] = mapped_column(default=None)
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
            "capture_agent_telemetry": self.capture_agent_telemetry,
            "allowed_models": self.allowed_models,
            "metadata": self.metadata_,
        }


class Budget(Base):
    """Budget model for spending limits."""

    __tablename__ = "budgets"
    __table_args__ = (
        # A period comes from one place or the other, never both, matching the
        # rule ``scoped_budgets`` already enforced when it carried its own. Without
        # it the pair encodes one concept twice and ``(86400, calendar_month)`` is
        # storable and meaningless.
        CheckConstraint(
            "NOT (budget_duration_sec IS NOT NULL AND reset_alignment IS NOT NULL)",
            name="ck_budgets_single_period_source",
        ),
    )

    budget_id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str | None] = mapped_column(default=None)
    # Exact, like the counters it is compared against: the gate is
    # ``spend + reserved <= max_budget``, and a cap stored as a binary float
    # would decide a 403 against an amount an operator never typed
    # (mozilla-ai/otari#691).
    max_budget: Mapped[Decimal | None] = mapped_column(UsdCost())
    budget_duration_sec: Mapped[int | None] = mapped_column()
    # Snap the window to a UTC calendar boundary instead of counting a fixed
    # number of seconds, which is the only way to express a calendar month (2592000
    # seconds is a different, 1.5 percent more generous, product). It lives here
    # rather than on the rows that enforce a budget because a limit and the period
    # it is spent over are one product decision, and splitting them let a ceiling
    # reset on a cadence the budget defining it had never heard of.
    reset_alignment: Mapped[str | None] = mapped_column(default=None)
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
            "reset_alignment": self.reset_alignment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


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
        # Deliberately not workspace-scoped yet. Widening this to
        # (workspace_id, name, user_id) would let two workspaces each hold a
        # "fast" entry, but resolution reads a process-wide cache keyed by name
        # alone, so the second would silently shadow the first at request time.
        # The constraint widens in the change that makes that cache
        # workspace-aware, not before.
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
        # Deliberately not workspace-scoped yet. Widening this to
        # (workspace_id, name, user_id) would let two workspaces each hold a
        # "fast" entry, but resolution reads a process-wide cache keyed by name
        # alone, so the second would silently shadow the first at request time.
        # The constraint widens in the change that makes that cache
        # workspace-aware, not before.
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
        """Serialize for the API. Never includes the secret, only ``last4``."""
        return {
            "name": self.name,
            "provider": self.provider,
            "api_base": self.api_base,
            "last4": self.last4,
            "timeout": self.timeout_seconds,
            "options": self.options or {},
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
    input_price_per_million: Mapped[Decimal] = mapped_column(UsdRate())
    output_price_per_million: Mapped[Decimal] = mapped_column(UsdRate())
    # Nullable: providers without prompt caching (or models without a
    # discounted cache rate) leave these unset. When set, the cost
    # calculation prices cache_read_tokens / cache_write_tokens at these
    # per-million-token rates, following the provider inclusion convention
    # (see log_usage in _pipeline.py).
    cache_read_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    cache_write_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    cache_write_1h_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
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


class BatchRecord(Base):
    """Ownership and accounting record for an asynchronous batch job.

    Written at creation time so results accounting can be made idempotent (bill
    and log once, on the first completed retrieval), the batch cost can be folded
    into ``users.spend``, and ownership can be enforced without depending on the
    provider round-tripping the ``otari_user_id`` metadata marker. Batches created
    before this table existed carry no record and fall back to the
    metadata-anchored ownership path in ``api/routes/batches.py``. ``workspace_id``
    additionally anchors which workspace's organization-scoped provider key
    (otari#643) lifecycle calls should resolve credentials from.
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
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    # SET NULL: a key may be revoked while its batch is still in flight.
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    # The workspace this batch was CREATED in (otari#643 follow-up), so
    # lifecycle calls (retrieve/cancel/results) can resolve organization-scoped
    # credentials from the batch's own origin rather than the retriever's
    # current workspace: a master-key or legitimately cross-workspace retrieval
    # would otherwise use the wrong organization's key, or find none, exactly
    # the failure `api_key_id` going NULL on key revocation already risks for
    # ownership. Nullable and SET NULL, not RESTRICT: batches created before
    # this column existed carry NULL here and fall back to the caller's own
    # workspace in `api/routes/batches.py`, and a workspace deleted out from
    # under an in-flight batch must not block that delete.
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="SET NULL"), index=True
    )
    model: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    # NULL until the first completed results retrieval accounts the batch; the
    # atomic NULL -> now transition is the idempotency gate for billing/logging.
    results_accounted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


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
        # A task-scoped read filters on all three; without this it walks every
        # record the user has for the embedding model before partitioning.
        Index("ix_routing_memory_user_model_task", "user_id", "embedding_model", "task_id"),
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
    # The ledger's record of a counter that is now exact, so it is exact too:
    # a float snapshot of an exact ``users.spend`` would no longer equal the
    # spend it claims to have recorded.
    previous_spend: Mapped[Decimal] = mapped_column(UsdCost())
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


class ScopedBudget(Base):
    """A USD ceiling on one tenancy scope, optionally narrowed to one provider.

    Two axes. The identity axis is ``(scope_type, scope_id)``: who is capped, an
    organization, a workspace, a member of either, or a single API key. The
    resource axis is ``provider_key_id``: NULL caps spend across every provider,
    a value narrows the cap to one provider instance. A request must pass every
    row that applies to it, and each row is an independent ceiling with its own
    counters and its own period window, unlike ``budgets``, where the window and
    the counters live on the user.

    Nothing here is denominated in dollars. A limit is a property of the budget
    this names, which is the only place in the schema that maps a cap to an
    amount.

    ``scope_type`` is a plain string rather than a database enum so a new scope
    needs no enum migration, and ``scope_id`` is a string so it holds both this
    codebase's string ids (an API key's) and the platform's UUIDs. Nothing here
    is a foreign key for the same reason: the rows a scope names live in four
    different tables, and a provider instance may be configured in ``config.yml``
    and have no row at all.

    A row names a ``budgets`` row and holds the counters for spending it. The
    limit and the period are read through the budget, never copied, so editing a
    budget moves every ceiling that names it. That is deliberate: a budget is a
    named thing an operator hands out, and the alternative was the same figure
    typed once per place it applied.

    This table does not replace ``budgets``, and the two enforce differently. A
    budget reached through ``users.budget_id`` is checked against
    ``users.spend + users.reserved``, so N users sharing one each get the full
    limit. A budget reached through a row here is checked against *this row's*
    counters, so everyone the scope names draws on one allowance. Same budget,
    two enforcement shapes, which is why both mechanisms exist.
    """

    __tablename__ = "scoped_budgets"
    __table_args__ = (
        # PostgreSQL treats NULLs as distinct in a plain UNIQUE, so one index
        # over the triple would enforce nothing on the aggregate rows (every one
        # of them has a NULL key, so no two are ever "equal"). Two partial
        # indexes instead: the narrowed rows are unique on the triple, and the
        # aggregate rows are unique on the identity alone, which is what makes
        # "one aggregate cap per scope" a real constraint.
        Index(
            "uq_scoped_budgets_scope_with_key",
            "scope_type",
            "scope_id",
            "provider_key_id",
            unique=True,
            postgresql_where=text("provider_key_id IS NOT NULL"),
            sqlite_where=text("provider_key_id IS NOT NULL"),
        ),
        Index(
            "uq_scoped_budgets_scope_no_key",
            "scope_type",
            "scope_id",
            unique=True,
            postgresql_where=text("provider_key_id IS NULL"),
            sqlite_where=text("provider_key_id IS NULL"),
        ),
        # The request path resolves rows by identity, so the lookup needs a
        # non-partial index: neither unique index above covers a scan that spans
        # narrowed and aggregate rows.
        Index("ix_scoped_budgets_scope", "scope_type", "scope_id"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    scope_type: Mapped[str] = mapped_column()
    scope_id: Mapped[str] = mapped_column()
    provider_key_id: Mapped[str | None] = mapped_column(default=None)
    name: Mapped[str | None] = mapped_column(default=None)
    # The budget this ceiling enforces. NOT NULL: a ceiling with no budget caps
    # nothing. The limit and the period are read through it rather than copied, so
    # editing a budget moves every ceiling that names it, which is the point of a
    # budget being a named thing rather than a number typed twice.
    budget_id: Mapped[str] = mapped_column(
        ForeignKey("budgets.budget_id", ondelete="RESTRICT"), nullable=False, index=True
    )
    current_spend: Mapped[Decimal] = mapped_column(UsdCost(), default=Decimal(0), server_default="0")
    # In-flight holds from reservations that have passed the gate but whose actual
    # cost is not known yet. Headroom is ``max_budget - current_spend -
    # reserved_spend``; a period roll zeroes ``current_spend`` only, so a hold
    # taken before the roll is still released correctly after it.
    reserved_spend: Mapped[Decimal] = mapped_column(UsdCost(), default=Decimal(0), server_default="0")
    period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class OrganizationModelPricing(Base):
    """One organization's rate for a model, sitting above the deployment price list.

    ``model_pricing`` carries no tenancy column: it is one price list for the
    whole deployment. This table is the layer above it, so an organization can
    price the models it uses at its own negotiated rates while every other
    organization, and every model it has not overridden, keeps resolving exactly
    as before. Resolution order is override, then deployment row, then the
    genai-prices dataset (`services.pricing_service.find_model_pricing`).

    **Keyed on ``model_key``, not a split provider and model.** The platform's
    equivalent table (`otari-ai` ``organization_model_pricing``) carries
    ``provider`` and ``model`` as separate columns. Here the whole pricing chain
    keys on one ``provider:model`` string, and that string is not always a
    provider and a model: a pricing key names a provider *instance*
    (``home_lab:llama-3``, over ``provider_type: openai``) and sometimes no model
    at all (``otari:web_search``). Splitting it would make an override
    unmatchable for exactly the keys an operator is most likely to have priced by
    hand, so the override keys the same way the row it overrides does.

    **An interval, where ``model_pricing`` carries a version series.** A price in
    ``model_pricing`` is ``(model_key, effective_at)`` and a later row silently
    shadows an earlier one, which is the right shape for a catalog an operator
    re-imports. An override is a commitment for a period, so it carries both ends
    and overlapping periods for one model are refused rather than shadowed
    (`services.organization_pricing_service`). ``effective_to`` NULL means open
    ended.

    **The overlap rule is enforced in the service, not by the database.** The
    natural constraint is a PostgreSQL ``EXCLUDE`` over a ``tstzrange``, and the
    platform has one. SQLite has no exclusion constraint and no range type, and
    it is what the OSS edition ships by default, so a database-side rule would
    hold on one engine and be a comment on the other. The unique index below is
    what both engines can enforce: it stops the exact-duplicate start, which is
    the collision two concurrent writers actually produce, while a partial
    overlap between two simultaneous inserts remains a narrow race the service's
    check can lose. Single-writer configuration traffic, and a wrong rate is
    visible and correctable rather than silent.

    Rates are the exact ``UsdRate`` type ``ModelPricing`` uses, and the two
    tables carry it together (#661, one migration over both). An override
    resolves *into* a transient ``ModelPricing``, so one implementation of the
    cost math prices both, which it could not if they disagreed about the type
    of money.
    """

    __tablename__ = "organization_model_pricing"
    __table_args__ = (
        # One index, doing both jobs, because they want the same columns in the
        # same order. As a constraint it refuses two rows for one key that begin
        # at the same instant, which is the part of the overlap rule either
        # engine can hold (see the class docstring for why the rest is in the
        # service). As an index it serves the resolution lookup: the two equality
        # columns lead, so the request path gets a prefix scan, and
        # ``effective_from`` trails so picking the newest applicable period is
        # index-ordered rather than a sort.
        Index(
            "uq_organization_model_pricing_period_start",
            "organization_id",
            "model_key",
            "effective_from",
            unique=True,
        ),
        # An inverted period would resolve for no instant at all, so it is a
        # storage error rather than a pricing decision. Equal ends are refused
        # too: a zero-width period is the same silent nothing.
        CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name="ck_organization_model_pricing_period_ordered",
        ),
        # Negative money prices a request as a credit. The service rejects it
        # with a message; these are the backstop for a writer that is not the
        # service, and they are spelled out per column because a single check
        # over all five would not say which rate was wrong.
        CheckConstraint(
            "input_price_per_million >= 0",
            name="ck_organization_model_pricing_input_non_negative",
        ),
        CheckConstraint(
            "output_price_per_million >= 0",
            name="ck_organization_model_pricing_output_non_negative",
        ),
        CheckConstraint(
            "cache_read_price_per_million IS NULL OR cache_read_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_read_non_negative",
        ),
        CheckConstraint(
            "cache_write_price_per_million IS NULL OR cache_write_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_write_non_negative",
        ),
        CheckConstraint(
            "cache_write_1h_price_per_million IS NULL OR cache_write_1h_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_write_1h_non_negative",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # CASCADE, where the request-plane tables above use RESTRICT. Those are kept
    # because a workspace's spend history must survive it; an override is
    # configuration, and an organization's rates mean nothing once the
    # organization is gone. The usage rows priced under it keep their own settled
    # cost, so deleting this loses no accounting.
    # No ``index=True``: the composite lookup index below leads on this column,
    # so a plain one on it would be a second index serving queries the first
    # already answers, paid for on every write.
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="CASCADE"), nullable=False
    )
    model_key: Mapped[str] = mapped_column()
    input_price_per_million: Mapped[Decimal] = mapped_column(UsdRate())
    output_price_per_million: Mapped[Decimal] = mapped_column(UsdRate())
    # Nullable for the same reason ``ModelPricing``'s are: a provider without
    # prompt caching, or a model with no discounted cache rate, leaves them unset
    # and the cost calculation falls back the way it already does.
    cache_read_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    cache_write_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    cache_write_1h_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    # Same shape and same ``min_input_tokens`` key as ``ModelPricing``, so the
    # transient row an override resolves into needs no tier translation.
    pricing_tiers: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    # ``UtcDateTime``, not ``DateTime(timezone=True)``, and this is the one place
    # in this file where that distinction is load-bearing. The flag is a no-op on
    # SQLite, which is what ``core/config.py`` defaults ``database_url`` to, so a
    # plain column reads back naive there and this table's timestamps are the
    # ones that go out over the wire: ``OrganizationModelPricingPublic`` would
    # serialize them with no offset, a browser parses an offset-less date-time as
    # *local*, and the Edit dialog would then round-trip the period shifted by
    # the reader's UTC offset on every save. ``UtcDateTime.impl`` is
    # ``DateTime(timezone=True)``, so the DDL and the migration are unchanged; it
    # normalizes on the way in and stamps UTC on the way out.
    #
    # ``ModelPricing`` above keeps the plain column because nothing renders its
    # ``effective_at`` into an editable control; the transient row an override
    # resolves into is stamped in ``_override_as_model_pricing`` for the cost
    # path, which is a different fix for a different reader.
    effective_from: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
    )
    # NULL means open ended, which is the common case: an organization sets a
    # rate and it applies until something replaces it.
    effective_to: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class WorkspaceBudgetDefault(Base):
    """A workspace-level template for a per-member ``ScopedBudget``.

    ``scoped_budgets`` holds concrete ceilings; this table has no counters of
    its own and enforces nothing directly. It is **materialized**: creating one
    on a workspace that already has members, or a member joining a workspace
    that already has one, stages a ``ScopedBudget(scope_type="workspace_member",
    scope_id=<member id>)`` row for each (see
    ``services/tenancy/workspace_budget_default_service.py``). A member with an
    existing ceiling for the same ``provider_key_id`` is left alone; a
    member-specific override always wins over the template.

    Same two-axis shape as ``ScopedBudget``: ``workspace_id`` is who the
    template belongs to, ``provider_key_id`` optionally narrows it to one
    provider instance (NULL applies to all of them). Unlike ``ScopedBudget``,
    ``workspace_id`` is a real foreign key: a template has exactly one owner
    and nothing else names it, so it is deleted with the workspace rather than
    requiring the same explicit cleanup ``ScopedBudget`` needs (see
    ``WorkspaceService._delete_scoped_budgets_for``).
    """

    __tablename__ = "workspace_budget_defaults"
    __table_args__ = (
        # Same reasoning as ScopedBudget's two partial indexes: PostgreSQL and
        # SQLite both treat NULLs as distinct in a plain UNIQUE, so a single
        # index over the pair would enforce nothing on the aggregate (NULL-key)
        # rows.
        Index(
            "uq_workspace_budget_defaults_with_key",
            "workspace_id",
            "provider_key_id",
            unique=True,
            postgresql_where=text("provider_key_id IS NOT NULL"),
            sqlite_where=text("provider_key_id IS NOT NULL"),
        ),
        Index(
            "uq_workspace_budget_defaults_no_key",
            "workspace_id",
            unique=True,
            postgresql_where=text("provider_key_id IS NULL"),
            sqlite_where=text("provider_key_id IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider_key_id: Mapped[str | None] = mapped_column(default=None)
    # The budget this workspace hands to every member. NOT NULL: a default that
    # names no budget is a template for nothing. ``RESTRICT`` because deleting a
    # budget a workspace hands out should be refused and explained rather than
    # silently withdraw the limit from every ceiling it materialized.
    #
    # The limit and the period live on the budget, not here, which is what lets
    # the Budgets page say that a row is a workspace's default. ``provider_key_id``
    # stays on this side: which provider a workspace applies the budget to is a
    # property of the assignment, and two workspaces may narrow one budget
    # differently.
    budget_id: Mapped[str] = mapped_column(
        ForeignKey("budgets.budget_id", ondelete="RESTRICT"), nullable=False, index=True
    )
    # ``UtcDateTime``, not ``DateTime(timezone=True)``: these two are serialized with
    # ``.isoformat()`` (``WorkspaceMemberBudgetPolicyPublic.from_model``) for the
    # dashboard, and on SQLite (this repo's default ``database_url``)
    # a plain ``DateTime(timezone=True)`` round-trips naive, so the wire value
    # would carry no offset and a browser would read it as local time.
    # ``UtcDateTime.impl`` is ``DateTime(timezone=True)``, so the DDL is unchanged.
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


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


class OrganizationGuardrail(Base):
    """A guardrail an organization runs over the requests of its workspaces.

    The plane *above* the deployment-wide guardrail settings, not a replacement
    for them: ``guardrails_url`` stays in ``runtime_settings`` and a deployment
    that configures no organization guardrails behaves exactly as it did
    (otari#654). A row here is a check the organization mandates; it is merged
    into the effective guardrail list at admission by ``prepare_gateway_tools``
    the same way a routing policy's mandate already is, so an organization can
    only ever add a check or tighten one a caller asked for.

    That is what keeps this inside the rule ``src/gateway/AGENTS.md`` records
    from #655/#678: a mandated guardrail can only make *fewer* requests succeed,
    never more, whichever endpoint it names. Which is also why the entry may
    carry its own ``url`` and credential where a workspace code-execution policy
    may not: the sandbox is a capability a workspace would be acquiring, and a
    guardrail is a restriction the organization is accepting. A caller can
    already point a request-body guardrail at a URL of their own
    (``models/guardrails.GuardrailConfig.url``, SSRF-checked on the request
    path), so storing one here grants nothing that was not already reachable.

    ``profile`` is unique per organization rather than a nickname being unique,
    which is where this parts company with the hosted
    ``organization_guardrail_key`` (unique on ``(organization_id, nickname)``,
    so one profile may be configured twice). The effective guardrail set on this
    request path is keyed by profile, because ``merge_guardrail_layers`` has
    always merged that way; two rows of one profile could therefore never both
    run, and one would silently win.
    """

    __tablename__ = "organization_guardrails"
    __table_args__ = (UniqueConstraint("organization_id", "profile", name="uq_organization_guardrails_org_profile"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="CASCADE"), nullable=False, index=True
    )
    profile: Mapped[str] = mapped_column(nullable=False)
    # NULL means "use the deployment's guardrails_url", which is the ordinary
    # case: an organization that runs its own any-guardrail deployment names it
    # here, and then the credential below is what authenticates to it.
    url: Mapped[str | None] = mapped_column(default=None)
    encrypted_credential: Mapped[str | None] = mapped_column(Text, default=None)
    mode: Mapped[str] = mapped_column(default="monitor", nullable=False)
    on_unavailable: Mapped[str] = mapped_column(default="block", nullable=False)
    validate_kwargs: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    # The organization's own kill switch. A disabled entry runs nowhere,
    # whatever its scope says, so an organization can stop a guardrail without
    # losing the credential and the workspace list it took to set up.
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    # The inheritance rule otari#654 asks for, and the hosted plane's
    # ``is_org_default`` under a name that says what it does: true means every
    # workspace of the organization runs this, including one created tomorrow,
    # and the scope rows below are not consulted. False means it runs only in
    # the workspaces named there, and a new workspace inherits nothing.
    applies_to_all_workspaces: Mapped[bool] = mapped_column(default=False, nullable=False)
    # ``UtcDateTime`` for the reason its neighbors use it: these are serialized
    # with ``.isoformat()`` for the dashboard, and a plain ``DateTime(timezone=True)``
    # round-trips naive on SQLite.
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class OrganizationGuardrailWorkspace(Base):
    """One workspace an organization guardrail is scoped to.

    Membership only: a row means "this guardrail runs in this workspace", and
    its absence means it does not. The hosted plane instead carries a
    ``disabled`` flag on the equivalent row and admits three states, two of
    which resolve to off; there is nothing here for a third state to record,
    because the scope is the organization's to set and a workspace has no veto
    over it (a veto would widen what succeeds, which #655/#678 does not allow).

    Ignored entirely when the guardrail's ``applies_to_all_workspaces`` is set,
    so rows left behind by flipping that on are inert rather than contradictory.

    Both sides cascade: the pairing has no meaning once either end is gone.
    """

    __tablename__ = "organization_guardrail_workspaces"

    organization_guardrail_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization_guardrails.id", ondelete="CASCADE"), primary_key=True
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
