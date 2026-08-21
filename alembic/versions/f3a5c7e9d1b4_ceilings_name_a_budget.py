"""Move the period onto ``budgets``, and make a scoped ceiling name one.

``scoped_budgets`` carried its own ``max_budget``, ``budget_duration_sec`` and
``reset_alignment``, so a cap could be defined in two places that had never
heard of each other: a budget an operator had named, and a figure typed into a
ceiling. This collapses that. A budget is now the only row in the schema that
maps a cap to an amount, and everything that enforces one names it and holds
counters.

Two consequences worth stating.

``budgets`` gains ``reset_alignment``, because a limit and the period it is
spent over are one product decision and splitting them is what allowed a ceiling
to reset on a cadence its budget had never specified. The CHECK matches the one
``scoped_budgets`` used to carry.

Enforcement becomes retroactive, deliberately. A ceiling used to copy the numbers
at materialization, so editing a budget left everyone already holding one on the
old figure. It now reads through, so editing a budget moves every ceiling that
names it, which is what makes a budget a named thing an operator hands out rather
than a number typed once per place it applies.

Backfill mints one budget per distinct ``(max_budget, budget_duration_sec,
reset_alignment)`` across the existing ceilings, so a deployment that gave forty
members the same figure gets one budget rather than forty. The downgrade copies
the numbers back and leaves the minted budgets in place, since nothing records
which of them existed only to back a ceiling.

Revision ID: f3a5c7e9d1b4
Revises: e2f4a6c8b0d3
Create Date: 2026-08-21
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision: str = "f3a5c7e9d1b4"
down_revision: str | Sequence[str] | None = "e2f4a6c8b0d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WITH_KEY = "uq_scoped_budgets_scope_with_key"
_NO_KEY = "uq_scoped_budgets_scope_no_key"
_SCOPE_INDEX = "ix_scoped_budgets_scope"
_BUDGET_INDEX = "ix_scoped_budgets_budget_id"
_BUDGET_FK = "fk_scoped_budgets_budget_id"
_PERIOD_CHECK = "ck_budgets_single_period_source"


def _scoped_budgets(*, with_budget_id: bool, with_inline: bool, budget_id_nullable: bool = True) -> sa.Table:
    """The table as it stands at each ``batch_alter_table`` below.

    ``copy_from`` is what stops SQLite's rebuild from dropping what reflection
    could not see, so the three indexes are declared here, partial clauses
    included. The budget foreign key is deliberately not declared: a rebuild that
    omits the column drops it either way, and naming it in a ``drop_constraint``
    fails on a table the rebuild does not believe has one.
    """
    columns = [
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("scope_type", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("provider_key_id", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("current_spend", sa.Float(), nullable=False),
        sa.Column("reserved_spend", sa.Float(), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]
    if with_budget_id:
        columns.insert(5, sa.Column("budget_id", sa.String(), nullable=budget_id_nullable))
    if with_inline:
        columns.extend(
            [
                sa.Column("max_budget", sa.Float(), nullable=True),
                sa.Column("budget_duration_sec", sa.Integer(), nullable=True),
                sa.Column("reset_alignment", sa.String(), nullable=True),
            ]
        )
    table = sa.Table("scoped_budgets", sa.MetaData(), *columns, sa.PrimaryKeyConstraint("id"))
    sa.Index(
        _WITH_KEY,
        table.c.scope_type,
        table.c.scope_id,
        table.c.provider_key_id,
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NOT NULL"),
        sqlite_where=sa.text("provider_key_id IS NOT NULL"),
    )
    sa.Index(
        _NO_KEY,
        table.c.scope_type,
        table.c.scope_id,
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NULL"),
        sqlite_where=sa.text("provider_key_id IS NULL"),
    )
    sa.Index(_SCOPE_INDEX, table.c.scope_type, table.c.scope_id)
    return table


def _budgets(*, with_alignment: bool) -> sa.Table:
    columns = [
        sa.Column("budget_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("max_budget", sa.Float(), nullable=True),
        sa.Column("budget_duration_sec", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]
    if with_alignment:
        columns.insert(4, sa.Column("reset_alignment", sa.String(), nullable=True))
    return sa.Table("budgets", sa.MetaData(), *columns, sa.PrimaryKeyConstraint("budget_id"))


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    op.add_column("budgets", sa.Column("reset_alignment", sa.String(), nullable=True))
    with op.batch_alter_table("budgets", copy_from=_budgets(with_alignment=True)) as batch:
        batch.create_check_constraint(
            _PERIOD_CHECK,
            "NOT (budget_duration_sec IS NOT NULL AND reset_alignment IS NOT NULL)",
        )

    op.add_column("scoped_budgets", sa.Column("budget_id", sa.String(), nullable=True))

    # One budget per distinct shape, not per ceiling: a deployment that gave
    # forty members the same figure should end up with one budget, not forty.
    now = datetime.now(UTC)
    minted: dict[tuple[object, object, object], str] = {}
    rows = bind.execute(
        sa.text("SELECT id, name, max_budget, budget_duration_sec, reset_alignment FROM scoped_budgets")
    ).fetchall()
    for row in rows:
        shape = (row.max_budget, row.budget_duration_sec, row.reset_alignment)
        budget_id = minted.get(shape)
        if budget_id is None:
            budget_id = str(uuid.uuid4())
            minted[shape] = budget_id
            bind.execute(
                sa.text(
                    "INSERT INTO budgets (budget_id, name, max_budget, budget_duration_sec, reset_alignment,"
                    " created_at, updated_at)"
                    " VALUES (:budget_id, :name, :max_budget, :duration, :alignment, :created_at, :updated_at)"
                ),
                {
                    "budget_id": budget_id,
                    "name": row.name or _shape_name(row.max_budget, row.budget_duration_sec, row.reset_alignment),
                    "max_budget": row.max_budget,
                    "duration": row.budget_duration_sec,
                    "alignment": row.reset_alignment,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        bind.execute(
            sa.text("UPDATE scoped_budgets SET budget_id = :budget_id WHERE id = :id"),
            {"budget_id": budget_id, "id": row.id},
        )

    with op.batch_alter_table(
        "scoped_budgets", copy_from=_scoped_budgets(with_budget_id=True, with_inline=True)
    ) as batch:
        batch.alter_column("budget_id", existing_type=sa.String(), nullable=False)
        batch.create_foreign_key(_BUDGET_FK, "budgets", ["budget_id"], ["budget_id"], ondelete="RESTRICT")
        batch.drop_column("max_budget")
        batch.drop_column("budget_duration_sec")
        batch.drop_column("reset_alignment")
    op.create_index(op.f(_BUDGET_INDEX), "scoped_budgets", ["budget_id"])


def _shape_name(max_budget: object, duration: object, alignment: object) -> str:
    """A readable name for a budget minted from a ceiling that had none."""
    amount = "no limit" if max_budget is None else f"${max_budget:g}"
    if alignment:
        return f"{amount} {str(alignment).replace('calendar_', 'per ')}"
    if duration:
        days = int(duration) // 86400
        return f"{amount} every {days}d" if days else f"{amount} every {duration}s"
    return f"{amount}, no reset"


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    op.drop_index(op.f(_BUDGET_INDEX), table_name="scoped_budgets")
    with op.batch_alter_table(
        "scoped_budgets",
        copy_from=_scoped_budgets(with_budget_id=True, with_inline=False, budget_id_nullable=False),
    ) as batch:
        batch.add_column(sa.Column("max_budget", sa.Float(), nullable=True))
        batch.add_column(sa.Column("budget_duration_sec", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("reset_alignment", sa.String(), nullable=True))

    bind.execute(
        sa.text(
            "UPDATE scoped_budgets SET "
            "max_budget = (SELECT b.max_budget FROM budgets b WHERE b.budget_id = scoped_budgets.budget_id), "
            "budget_duration_sec = (SELECT b.budget_duration_sec FROM budgets b"
            " WHERE b.budget_id = scoped_budgets.budget_id), "
            "reset_alignment = (SELECT b.reset_alignment FROM budgets b WHERE b.budget_id = scoped_budgets.budget_id)"
        )
    )

    with op.batch_alter_table(
        "scoped_budgets", copy_from=_scoped_budgets(with_budget_id=True, with_inline=True, budget_id_nullable=False)
    ) as batch:
        # No `drop_constraint` for the budget foreign key: `copy_from` does not
        # declare it, so SQLite's rebuild drops it with the column either way.
        batch.drop_column("budget_id")
        batch.create_check_constraint(
            "ck_scoped_budgets_single_period_source",
            "NOT (budget_duration_sec IS NOT NULL AND reset_alignment IS NOT NULL)",
        )

    with op.batch_alter_table("budgets", copy_from=_budgets(with_alignment=True)) as batch:
        # No `drop_constraint` for the period CHECK: `_budgets` does not declare
        # it, so the rebuild below drops it along with the column it constrains.
        batch.drop_column("reset_alignment")
