"""The plugin's two tables, on a ``MetaData`` of its own.

**Not on ``gateway.models.entities.Base``.** These tables belong to a module
Otari does not ship, so putting them on Otari's metadata would make core's
Alembic autogenerate see them and try to own them: a contributor running
autogenerate in a checkout with this plugin installed would get a core revision
creating ``alert_rules``, and one without it a core revision dropping them
again. A separate declarative base keeps each side's autogenerate blind to the
other, which is the same separation the two version tables give the histories.

**No foreign key into a core table**, for the reason
``gateway.container.MigrationContribution`` documents: the core chain runs
first and knows nothing about these tables, so a core revision that rebuilds
``organization`` would fail on a constraint the core chain did not create.
``organization_id`` is therefore a plain indexed column. A deleted organization
leaves its rules behind, and they are inert: the evaluator only reaches a rule
through a ``budgets`` row carrying the same organization id, and the CRUD
service only ever reads the caller's own organization.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import CheckConstraint, ForeignKey, Index, MetaData, Text, UniqueConstraint, Uuid, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from gateway.models.tenancy import UtcDateTime


class Base(DeclarativeBase):
    """The plugin's declarative base. Its metadata is what its Alembic chain sees."""

    metadata = MetaData()


class AlertRule(Base):
    """Where an organization wants to be told about its budgets, and when.

    One row is one destination plus the thresholds that reach it. The
    destination is an `Apprise <https://github.com/caronc/apprise>`_ URL, which
    is what keeps this table one column wide instead of one column per vendor:
    ``slack://``, ``discord://``, ``pagerduty://`` and a plain ``json://``
    webhook are all the same string.

    **Organization-scoped, so a tenant is told about its own budgets.** A rule
    watches the ``scoped_budgets`` rows whose ``budgets`` row carries this
    ``organization_id``. A budget with a NULL one is therefore never alerted
    on, which is not an oversight: NULL means the deployment's own, and the
    organization-scoped surface never lists, offers or repoints one.
    Deployment-wide rules are the operator's plane and would need a column here
    saying what a rule watches.

    The URL is a credential, so it is encrypted at rest with
    ``OTARI_SECRET_KEY`` and never returned, the same convention
    ``ProviderCredential`` uses. The API returns ``redact_alert_destination``
    output instead, which masks path segments as well as userinfo because
    Apprise puts its tokens in the path.
    """

    __tablename__ = "alert_rules"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_alert_rules_org_name"),
        CheckConstraint(
            "warn_at_percent IS NULL OR (warn_at_percent > 0 AND warn_at_percent < 100)",
            name="ck_alert_rules_warn_percent_range",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Indexed, not a foreign key. See the module docstring.
    organization_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, index=True)
    name: Mapped[str] = mapped_column(nullable=False)
    encrypted_destination: Mapped[str] = mapped_column(Text, nullable=False)
    # Kept in the clear beside the ciphertext so the list endpoint and the
    # evaluator's log lines can name the destination without the secret key.
    redacted_destination: Mapped[str] = mapped_column(nullable=False)
    # Percent of the cap at which a warning fires, or NULL for no warning. The
    # useful half of this feature: a refusal is already too late to act on.
    #
    # **No column default, deliberately.** A scalar ``default=80`` here fires
    # whenever the attribute is None at INSERT, and SQLAlchemy cannot tell an
    # explicit ``None`` from an omission, so it would silently overwrite the one
    # value a caller most needs to be able to send: ``warn_at_percent: null``
    # means "no early warning, alert only on refusal". The default belongs to
    # the request schema (``AlertRuleCreate``), which is the layer that can
    # distinguish "not sent" from "sent as null".
    warn_at_percent: Mapped[int | None] = mapped_column(default=None)
    # Whether reaching the cap itself fires. Separable because a deployment that
    # routes refusals through its own error monitoring wants only the warning.
    notify_on_exceeded: Mapped[bool] = mapped_column(default=True, nullable=False)
    # Kill switch: stop the alerts without losing the destination it took to set up.
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class AlertDelivery(Base):
    """One alert that has already been sent, so it is not sent again.

    **This table is the load-bearing part of the feature, and the unique
    constraint is the mechanism.** Two duplicate sources collapse into it. A
    threshold stays crossed for the rest of the budget period, so comparing
    spend against the cap alone would re-alert every tick; and a contributed
    background task runs once per worker, so N workers would each send the same
    alert at once. Both are settled by inserting this row *before* dispatching
    and treating an ``IntegrityError`` as "somebody already did it". No advisory
    lock and no leader election, matching
    ``gateway.services.budget_reservation_ledger``.

    ``period_start`` is copied off the ``scoped_budgets`` row rather than
    referenced, which is what re-arms an alert after a budget resets: a new
    period is a different key. It also means a row outlives the period it
    describes, which is why ``purge_delivered_before`` exists.

    A NULL ``period_start`` is a budget with no period at all. NULLs do not
    collide in a unique constraint on PostgreSQL, so those rows are deduped by
    the partial index below instead.
    """

    __tablename__ = "alert_deliveries"
    __table_args__ = (
        UniqueConstraint(
            "alert_rule_id",
            "scoped_budget_id",
            "period_start",
            "kind",
            name="uq_alert_deliveries_rule_budget_period_kind",
        ),
        # The periodless case the class docstring describes. PostgreSQL treats
        # two NULL ``period_start`` values as distinct, so the constraint above
        # would let a never-rolling ceiling alert once per tick forever.
        Index(
            "uq_alert_deliveries_rule_budget_kind_no_period",
            "alert_rule_id",
            "scoped_budget_id",
            "kind",
            unique=True,
            sqlite_where=text("period_start IS NULL"),
            postgresql_where=text("period_start IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # The one foreign key here, and it points at the plugin's own table, so the
    # caution against pointing into a core table does not apply.
    alert_rule_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("alert_rules.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Not a foreign key, matching ``scoped_budgets``' own columns, which declare
    # none. A deleted ceiling leaves its rows to ``purge_delivered_before``.
    scoped_budget_id: Mapped[str] = mapped_column(nullable=False)
    # ``warning`` or ``exceeded``. A plain string rather than a database enum,
    # for the reason ``ScopedBudget.scope_type`` is one: a third kind should not
    # need an enum migration.
    kind: Mapped[str] = mapped_column(nullable=False)
    period_start: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    # The row is claimed before dispatch, so a false here is an alert that was
    # suppressed and never arrived: what an operator debugging silence needs.
    delivered: Mapped[bool] = mapped_column(default=False, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
