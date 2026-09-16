import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gateway.models.base import Base, UtcDateTime
from gateway.models.money import UsdCost


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
