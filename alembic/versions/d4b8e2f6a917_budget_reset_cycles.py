"""Replace a budget's duration-or-alignment period with a reset cycle and its settings.

A budget's period used to come from one of two columns: ``budget_duration_sec``,
seconds counted from the last reset, or ``reset_alignment``, one of three UTC
calendar boundaries. That pair could express neither "every other Monday and
Friday" nor "the 15th of each month", and the duration half had a second problem:
its window opened at ``now`` on the request that found the old one expired, so on
a quiet workspace the reset walked forward through the day.

Six columns replace them. ``reset_cycle`` names the cadence and NULL is "never";
each cycle carries exactly its own settings and no others, which the CHECKs below
enforce rather than leave to an "ignored when" rule. Interval cycles
(``every_n_hours``, ``every_n_days``) carry ``reset_every_n`` and
``reset_anchor_at``, and counting whole steps from the anchor is what keeps their
phase.

**Conversions, every one of them tightening or exact.** Direction matters here
and the obvious reading is backwards: a *shorter* period is more permissive over
time, because each period grants the limit again. A $10 budget on an hourly
period admits $240 a day; the same budget made daily admits $10.

===========================  ===================================  ==============
Existing                     Becomes                              Direction
===========================  ===================================  ==============
``calendar_day``             ``daily``                            exact
``calendar_week``            ``weekly`` on Monday                 exact
``calendar_month``           ``monthly`` on day 1                 exact
whole days                   ``every_n_days``, anchored now       exact cadence
whole hours under a day      ``every_n_hours``, anchored now      exact cadence
under an hour                ``every_n_hours`` of 1               tightens
neither column set           NULL, still never                    exact
===========================  ===================================  ==============

Sub-hour is the only arm that changes what a budget admits. An hour is the floor
because below it a period stops being a budget and becomes a rate limit, which is
a separate feature, and because these counters settle asynchronously: a window
shorter than the settle can roll before the spend it was meant to cap is recorded
against it. Rounding up rather than down is deliberate, and it is the safe
direction: a 5-minute $10 budget becomes $10 an hour rather than $120 an hour.

The anchor for a converted interval budget is the migration's own clock. The old
rows carry no anchor to recover, since a duration-based window was defined by
whenever it last rolled, so "from here on" is the only honest answer.

``downgrade()`` is lossy in the direction the new vocabulary is richer: a weekly
budget on several days, a monthly one on any day but the 1st, and every yearly
one have no spelling in the old pair. Each becomes the nearest old cadence that
does not admit more than it did, which is the same rule ``upgrade()`` follows.

Revision ID: d4b8e2f6a917
Revises: e3b7d1a5c9f2
Create Date: 2026-10-05
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision: str = "d4b8e2f6a917"
down_revision: str | Sequence[str] | None = "e3b7d1a5c9f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "budgets"

_OLD_CHECK = "ck_budgets_single_period_source"
_OLD_CHECK_SQL = "NOT (budget_duration_sec IS NOT NULL AND reset_alignment IS NOT NULL)"

_HOUR = 3600
_DAY = 86400

# Monday, as `date.weekday()` numbers it, which is the bit the mask sets for a
# converted `calendar_week` budget.
_MONDAY_MASK = 1 << 0

_NEW_CHECKS: tuple[tuple[str, str], ...] = (
    (
        "ck_budgets_reset_cycle_vocabulary",
        "reset_cycle IS NULL OR reset_cycle IN "
        "('every_n_hours', 'every_n_days', 'daily', 'weekly', 'monthly', 'yearly')",
    ),
    (
        "ck_budgets_reset_every_n_present",
        "(COALESCE(reset_cycle, '') IN ('every_n_hours', 'every_n_days')) = (reset_every_n IS NOT NULL)",
    ),
    ("ck_budgets_reset_every_n_positive", "reset_every_n IS NULL OR reset_every_n > 0"),
    (
        "ck_budgets_reset_anchor_present",
        "(COALESCE(reset_cycle, '') IN ('every_n_hours', 'every_n_days')) = (reset_anchor_at IS NOT NULL)",
    ),
    (
        "ck_budgets_reset_weekdays_present",
        "(COALESCE(reset_cycle, '') = 'weekly') = (reset_weekdays IS NOT NULL)",
    ),
    ("ck_budgets_reset_weekdays_range", "reset_weekdays IS NULL OR (reset_weekdays BETWEEN 1 AND 127)"),
    (
        "ck_budgets_reset_month_day_present",
        "(COALESCE(reset_cycle, '') IN ('monthly', 'yearly')) = (reset_month_day IS NOT NULL)",
    ),
    ("ck_budgets_reset_month_day_range", "reset_month_day IS NULL OR (reset_month_day BETWEEN 1 AND 28)"),
    ("ck_budgets_reset_month_present", "(COALESCE(reset_cycle, '') = 'yearly') = (reset_month IS NOT NULL)"),
    ("ck_budgets_reset_month_range", "reset_month IS NULL OR (reset_month BETWEEN 1 AND 12)"),
)

_NEW_COLUMNS: tuple[sa.Column, ...] = (
    sa.Column("reset_cycle", sa.String(), nullable=True),
    sa.Column("reset_every_n", sa.Integer(), nullable=True),
    sa.Column("reset_anchor_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("reset_weekdays", sa.Integer(), nullable=True),
    sa.Column("reset_month_day", sa.Integer(), nullable=True),
    sa.Column("reset_month", sa.Integer(), nullable=True),
)


def _anchor() -> datetime:
    """Where a converted interval budget's first period opens.

    The migration's own clock, because a duration-based window was defined by
    whenever it last rolled and no column recorded that, so "from here on" is the
    only honest answer.
    """
    return datetime.now(UTC)


def _budgets_table(*, old_period: bool, new_period: bool) -> sa.Table:
    """The table as it stands at a given point, for SQLite's rebuild.

    ``copy_from`` is what carries the columns and constraints reflection cannot
    recover through a batch rebuild, so the shape here has to match the database
    the batch is about to rewrite.
    """
    columns: list[sa.Column] = [
        sa.Column("budget_id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=True),
        # Named, because `copy_from` is what SQLite's rebuild recreates the table
        # from: an unnamed FK here is one the batch cannot find by name and the
        # rebuild drops, which the schema chain tests catch.
        sa.Column(
            "organization_id",
            sa.Uuid(),
            sa.ForeignKey("organization.id", name="fk_budgets_organization_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("max_budget", sa.Numeric(18, 6), nullable=True),
        sa.Column("token_limit", sa.BigInteger(), nullable=True),
        sa.Column("request_limit", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]
    # The index on the FK, declared for the same reason the FK is named: SQLite
    # rebuilds the table from this definition, and what is not here is dropped.
    constraints: list[sa.schema.SchemaItem] = [
        sa.Index("ix_budgets_organization_id", "organization_id"),
    ]
    if old_period:
        columns.append(sa.Column("budget_duration_sec", sa.Integer(), nullable=True))
        columns.append(sa.Column("reset_alignment", sa.String(), nullable=True))
        constraints.append(sa.CheckConstraint(_OLD_CHECK_SQL, name=_OLD_CHECK))
    if new_period:
        columns.extend(sa.Column(c.name, c.type, nullable=True) for c in _NEW_COLUMNS)
        constraints.extend(sa.CheckConstraint(sql, name=name) for name, sql in _NEW_CHECKS)
    return sa.Table(_TABLE, sa.MetaData(), *columns, *constraints)


def upgrade() -> None:
    for column in _NEW_COLUMNS:
        op.add_column(_TABLE, sa.Column(column.name, column.type, nullable=True))

    budgets = sa.table(
        _TABLE,
        sa.column("reset_alignment", sa.String),
        sa.column("reset_cycle", sa.String),
        sa.column("reset_weekdays", sa.Integer),
        sa.column("reset_month_day", sa.Integer),
    )

    op.execute(budgets.update().where(budgets.c.reset_alignment == "calendar_day").values(reset_cycle="daily"))
    op.execute(
        budgets.update()
        .where(budgets.c.reset_alignment == "calendar_week")
        .values(reset_cycle="weekly", reset_weekdays=_MONDAY_MASK)
    )
    op.execute(
        budgets.update()
        .where(budgets.c.reset_alignment == "calendar_month")
        .values(reset_cycle="monthly", reset_month_day=1)
    )

    # The interval arms are computed in Python rather than in SQL, because the
    # arithmetic is not dialect-neutral: PostgreSQL's `/` on two integers
    # truncates and SQLite's promotes to a float, so one expression writes 2 on
    # one engine and 2.5 on the other, and `CAST` does not rescue it either
    # (SQLite truncates a cast, PostgreSQL rounds it). The table holds one row per
    # named budget, so reading them back costs nothing worth a dialect branch.
    connection = op.get_bind()
    durations = connection.execute(
        sa.text(
            "SELECT budget_id, budget_duration_sec FROM budgets "
            "WHERE budget_duration_sec IS NOT NULL AND budget_duration_sec > 0"
        )
    ).fetchall()
    for budget_id, seconds in durations:
        if seconds % _DAY == 0:
            cycle, every_n = "every_n_days", seconds // _DAY
        else:
            # Ceiling division, so the period is never shorter than it was: a
            # shorter period admits the limit again sooner. A sub-hour duration
            # ceilings to the 1-hour floor by the same expression.
            cycle, every_n = "every_n_hours", -(-seconds // _HOUR)
        connection.execute(
            sa.text(
                "UPDATE budgets SET reset_cycle = :cycle, reset_every_n = :every_n, "
                "reset_anchor_at = :anchor WHERE budget_id = :budget_id"
            ),
            {"cycle": cycle, "every_n": every_n, "anchor": _anchor(), "budget_id": budget_id},
        )

    with op.batch_alter_table(_TABLE, copy_from=_budgets_table(old_period=True, new_period=True)) as batch:
        batch.drop_constraint(_OLD_CHECK, type_="check")
        batch.drop_column("budget_duration_sec")
        batch.drop_column("reset_alignment")
        for name, sql in _NEW_CHECKS:
            batch.create_check_constraint(name, sql)


def downgrade() -> None:
    op.add_column(_TABLE, sa.Column("budget_duration_sec", sa.Integer(), nullable=True))
    op.add_column(_TABLE, sa.Column("reset_alignment", sa.String(), nullable=True))

    budgets = sa.table(
        _TABLE,
        sa.column("budget_duration_sec", sa.Integer),
        sa.column("reset_alignment", sa.String),
        sa.column("reset_cycle", sa.String),
        sa.column("reset_every_n", sa.Integer),
    )

    op.execute(budgets.update().where(budgets.c.reset_cycle == "daily").values(reset_alignment="calendar_day"))
    # Lossy in the direction the new vocabulary is richer, and lossy towards the
    # tighter cadence for the same reason upgrade() rounds up: a weekly budget on
    # Monday and Friday has no old spelling, and a week admits its limit less
    # often than those two periods did.
    op.execute(budgets.update().where(budgets.c.reset_cycle == "weekly").values(reset_alignment="calendar_week"))
    op.execute(budgets.update().where(budgets.c.reset_cycle == "monthly").values(reset_alignment="calendar_month"))
    # A year has no old spelling at all; a month is the longest the old pair
    # reaches, so a yearly budget comes back monthly rather than annual.
    op.execute(budgets.update().where(budgets.c.reset_cycle == "yearly").values(reset_alignment="calendar_month"))
    op.execute(
        budgets.update()
        .where(budgets.c.reset_cycle == "every_n_hours")
        .values(budget_duration_sec=budgets.c.reset_every_n * _HOUR)
    )
    op.execute(
        budgets.update()
        .where(budgets.c.reset_cycle == "every_n_days")
        .values(budget_duration_sec=budgets.c.reset_every_n * _DAY)
    )

    with op.batch_alter_table(_TABLE, copy_from=_budgets_table(old_period=True, new_period=True)) as batch:
        for name, _sql in _NEW_CHECKS:
            batch.drop_constraint(name, type_="check")
        for column in _NEW_COLUMNS:
            batch.drop_column(column.name)
        batch.create_check_constraint(_OLD_CHECK, _OLD_CHECK_SQL)
