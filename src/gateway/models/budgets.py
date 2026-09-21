"""ORM tables for budgets: limits, scoped ceilings, reservations, and workspace defaults."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, get_args

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Index, Uuid, false, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gateway.models.base import Base, UtcDateTime
from gateway.models.money import UsdCost

# The largest token or request limit a route accepts, and the largest hold it
# will place. Well below the BIGINT ceiling those columns are, for the two
# reasons :data:`gateway.models.money.MAX_USD_LIMIT` keeps its own headroom.
#
# The wire cannot carry the type's maximum. JSON numbers are doubles, so
# ``9223372036854775807`` renders in the published schema as
# ``9.223372036854776e+18``, which is 9223372036854775808: a client sending
# exactly the maximum the spec advertises sends a value the column refuses. A
# quadrillion is exact as a double, so the schema says what it means.
#
# And the gate adds server-side. Every reserve evaluates
# ``current + reserved + held`` as BIGINT arithmetic, so a nonzero counter plus a
# hold near the type's ceiling overflows and answers with a 500 where a 403 was
# owed. Holds are clamped to this too, because the token estimate derives from a
# client-supplied output bound that nothing else limits.
#
# A quadrillion tokens is four orders of magnitude past any real allowance, and
# leaves the sum of three of them ~9000x inside the type.
MAX_COUNT_LIMIT = 1_000_000_000_000_000

# An enum changes the published OpenAPI schema, so both vocabularies stay `Literal`.
ResetAlignment = Literal["calendar_day", "calendar_week", "calendar_month"]
RESET_ALIGNMENTS: tuple[ResetAlignment, ...] = get_args(ResetAlignment)
ALIGN_DAY: ResetAlignment = "calendar_day"
ALIGN_WEEK: ResetAlignment = "calendar_week"
ALIGN_MONTH: ResetAlignment = "calendar_month"

ScopeType = Literal["organization", "workspace", "workspace_member", "org_member", "api_token"]
SCOPE_TYPES: tuple[ScopeType, ...] = get_args(ScopeType)
SCOPE_ORGANIZATION: ScopeType = "organization"
SCOPE_WORKSPACE: ScopeType = "workspace"
SCOPE_WORKSPACE_MEMBER: ScopeType = "workspace_member"
SCOPE_ORG_MEMBER: ScopeType = "org_member"
SCOPE_API_TOKEN: ScopeType = "api_token"


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
    # Which tenant defined this budget, and therefore who may change it.
    #
    # NULL means the deployment's own: every budget predating `b7e1c4a9d2f5`
    # reads NULL, and so does every one the otari-ai cutover migration mints,
    # because that migration deliberately shares one budget per distinct
    # (cap, period) shape across the ceilings it writes and a shape shared by two
    # tenants' ceilings has no single owner. NULL is not "unowned and up for
    # grabs": the organization-scoped surface never lists, offers or repoints one,
    # so from a tenant's side it does not exist.
    #
    # Nullable and set only on the tenant-scoped create path, which is what keeps
    # that migration working with no backfill: its preflight refuses on a
    # *missing* column and its inserts name theirs explicitly, so a new nullable
    # one is invisible to it.
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(),
        ForeignKey("organization.id", name="fk_budgets_organization_id", ondelete="CASCADE"),
        default=None,
        index=True,
    )
    # Exact, like the counters it is compared against: the gate is
    # ``spend + reserved <= max_budget``, and a cap stored as a binary float
    # would decide a 403 against an amount an operator never typed
    # (mozilla-ai/otari#691).
    max_budget: Mapped[Decimal | None] = mapped_column(UsdCost())
    # The non-USD ceilings, independent of ``max_budget`` and of each other: a
    # budget may cap dollars, tokens, requests, or any combination, and a NULL on
    # an axis is unbounded there. Deliberately no "at least one limit" constraint,
    # because a budget with every limit NULL is a named period that admits
    # everything and predates these columns.
    #
    # BIGINT: a monthly token allowance for one organization outgrows a 32-bit
    # counter, and the counters compared against these are the same width.
    token_limit: Mapped[int | None] = mapped_column(BigInteger(), default=None)
    request_limit: Mapped[int | None] = mapped_column(BigInteger(), default=None)
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
            "token_limit": self.token_limit,
            "request_limit": self.request_limit,
            "budget_duration_sec": self.budget_duration_sec,
            "reset_alignment": self.reset_alignment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
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
    """A spending ceiling on one tenancy scope, optionally narrowed to one provider.

    Two axes. The identity axis is ``(scope_type, scope_id)``: who is capped, an
    organization, a workspace, a member of either, or a single API key. The
    resource axis is ``provider_key_id``: NULL caps spend across every provider,
    a value narrows the cap to one provider instance. A request must pass every
    row that applies to it, and each row is an independent ceiling with its own
    counters and its own period window, unlike ``budgets``, where the window and
    the counters live on the user.

    No limit is stored here. A limit is a property of the budget this names,
    which is the only place in the schema that maps a cap to a figure, on any of
    the three axes it can cap.

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
    # One counter pair per non-USD axis the budget can cap, holding and settling
    # exactly as the money pair above does. A period roll zeroes the ``current_*``
    # of all three axes and leaves every hold, so a hold taken before a roll is
    # still released correctly after it.
    current_tokens: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    reserved_tokens: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    current_requests: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    reserved_requests: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class BudgetReservation(Base):
    """One in-flight budget hold, recorded as a row.

    ``users.reserved`` and ``scoped_budgets.reserved_spend`` stay the O(1)
    counters the gate reads; this is the ledger behind them, and it exists for
    the two things a counter cannot do (mozilla-ai/otari#742):

    * **Release becomes idempotent.** Without an identity, a second release for
      the same request silently subtracts the hold twice. ``_release_reserved``
      clamps at zero, so that shows up not as an error but as an under-count of
      live holds, which weakens the very overspend guarantee the reserve gate
      exists to provide. The status transition here is what makes only the first
      release do the work.
    * **A leaked hold becomes reclaimable individually.** A failure between
      reserve and settle used to leave an amount in the counter that could be
      seen only in aggregate and released by nothing at all: the budget reset
      zeroes ``spend`` and leaves ``reserved`` where it is. With a row it has an
      owner, an age and a TTL.

    The row is written *after* the holds it records, never before. A hold with no
    row is the pre-existing leak the sweep bounds; a row with no hold would have
    the sweep release an amount nobody holds, under-counting the live ones. Of
    the two inconsistent windows only one is safe, and this is it.

    Standalone mode only: hybrid mode reserves nothing locally, because the
    platform holds against its own ledger.
    """

    __tablename__ = "budget_reservations"
    __table_args__ = (
        # The global sweep's access path: active rows whose TTL has elapsed.
        # Equality on ``status`` leads so the range scan on ``expires_at`` rides
        # the same index.
        Index("ix_budget_reservations_status_expires_at", "status", "expires_at"),
        # The per-user reclaim's, which runs on every request that takes a hold.
        # It has to lead on ``user_id``: given only the index above, the planner
        # takes it and filters ``user_id``, so one user's reclaim pays for the
        # whole deployment's backlog of expired rows. Leading on ``user_id`` also
        # serves the FK cascade, so this replaces the plain index on that column
        # rather than joining it.
        Index("ix_budget_reservations_user_status_expires", "user_id", "status", "expires_at"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    # No ``index=True``: the composite in ``__table_args__`` leads on this column,
    # so a plain index here would be a second, redundant one, and the migration
    # deliberately does not create it. Declaring it anyway made a ``create_all``
    # schema and a migrated one disagree.
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    # What the per-user leg holds in ``users.reserved``. Zero when the request
    # held only scoped ceilings (a user with no budget row still passes those).
    estimate: Mapped[Decimal] = mapped_column(UsdCost(), default=Decimal(0), server_default="0")
    # What the same leg holds on the other two axes. Recorded per axis because the
    # sweep has to give back every axis a leaked hold took: a period roll zeroes
    # ``current_*`` and deliberately leaves the holds, so a token hold nothing
    # releases shrinks that ceiling for good.
    token_estimate: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    request_estimate: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    # Whether the ``users.reserved`` write actually happened. Distinct from
    # ``estimate > 0`` because a zero-cost request on an enforced budget still
    # takes the hold, and the release has to match what the reserve did.
    user_reserved: Mapped[bool] = mapped_column(default=False, server_default=false())
    # The status is a plain string and not a database enum, so a new state needs no enum migration.
    # Its values are the ``RESERVATION_*`` constants in ``services/budgets/_ledger.py``.
    status: Mapped[str] = mapped_column(default="active", server_default="active", nullable=False)
    # After this instant a still-active row is treated as leaked and reclaimed.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class BudgetReservationScope(Base):
    """The hold one reservation placed on one scoped ceiling.

    ``scoped_budget_id`` is deliberately not a foreign key, following
    ``ScopedBudget``'s own convention: a ceiling deleted while a request is in
    flight leaves an orphan line that the release skips, rather than forcing the
    delete to cascade into live holds.

    The amounts are stored per line, one per axis, even though today every
    ceiling of a request holds the same figures. A ledger line that does not say
    what it holds is not a ledger line, and reading the amounts from the parent
    would silently become wrong the first time the two diverge.
    """

    __tablename__ = "budget_reservation_scopes"
    __table_args__ = (Index("ix_budget_reservation_scopes_reservation_id", "reservation_id"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    reservation_id: Mapped[str] = mapped_column(
        ForeignKey("budget_reservations.id", ondelete="CASCADE"), nullable=False
    )
    scoped_budget_id: Mapped[str] = mapped_column(nullable=False)
    amount: Mapped[Decimal] = mapped_column(UsdCost(), default=Decimal(0), server_default="0")
    token_amount: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    request_amount: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")


class WorkspaceBudgetDefault(Base):
    """A workspace-level template for a per-member ``ScopedBudget``.

    This table has no counters and enforces nothing.
    Each member of the workspace gets a ``ScopedBudget`` row made from the template.
    A member's own ceiling for the same ``provider_key_id`` wins over the template.
    ``provider_key_id`` narrows the template to one provider instance, and ``None`` applies it to all of them.
    ``workspace_id`` is a foreign key, so a template is deleted with its workspace.
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
