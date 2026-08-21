"""store the budget counters as exact numerics

Revision ID: b3f8d1c6a4e7
Revises: f3a5c7e9d1b4
Create Date: 2026-08-21 10:00:00.000000

The counters the budget gate enforces against were binary floating point while
the rows that sum into them are exact. They become ``NUMERIC(18, 6)``: the same
type and scale ``usage_logs.cost`` took in ``a7c3e5d9b1f4``, because these hold
sums of those amounts and a counter at a coarser scale would round every
settlement it recorded. mozilla-ai/otari#691, following mozilla-ai/otari#661.

Six columns over four tables: ``users.spend`` / ``reserved``,
``budgets.max_budget``, ``budget_reset_logs.previous_spend``, and
``scoped_budgets.current_spend`` / ``reserved_spend``. ``budgets.max_budget`` is
the only limit column since ``f3a5c7e9d1b4`` consolidated them, so every
enforced ceiling reads its cap from the one exact place.

**A stored value moves only where it was already wrong.** PostgreSQL casts
``double precision`` to ``numeric`` through the float's shortest round-trip
decimal representation, so a cap an operator typed as ``25`` or ``0.5`` arrives
unchanged. A *spend* counter is the one that can move, and that is the point:
``0.6619999999999999`` becomes ``0.662000``, which is the sum of the settled
rows it was supposed to have been all along. Anything below half a micro-dollar
of accumulated error is discarded with it. Rounding is half away from zero, so a
counter moves up as readily as down, by less than half a micro-dollar. The
downgrade restores the column type and cannot restore the discarded digits,
which is no loss: the digits it discards are float error, not money.

**On PostgreSQL this rewrites all four tables.** ``double precision`` to
``numeric`` is not binary-coercible, so each ALTER copies the table under an
ACCESS EXCLUSIVE lock, which blocks reads as well as writes. Three of the four
are small by construction (one row per user, budget, and ceiling).
``budget_reset_logs`` is the one that only grows, at one row per user per budget
period, so a deployment resetting a thousand users daily accumulates a few
hundred thousand rows a year; on the timings #661 measured for ``usage_logs``
(about 2.3 seconds per million rows) that is well under a second. It is still a
pause that happens at startup when ``auto_migrate`` is set, so a deployment with
real history should run ``otari migrate`` deliberately.

**An amount too large for the column stops the migration before it starts.**
``NUMERIC(18, 6)`` tops out just under $1T, and a float column had no ceiling, so
a deployment that typed a big round number as "no limit" would otherwise meet a
bare ``numeric field overflow`` from an ALTER, at startup, with nothing naming
the row. The pre-flight below names the table and column instead. It is
PostgreSQL-only because SQLite has no numeric storage class to overflow.
``models.money.MAX_USD_LIMIT`` keeps a new cap under the ceiling from the API
side, so this is about history rather than about what a deployment can write now.

SQLite gets the same DDL through a table rebuild, and there the change is
declarative: SQLite has no numeric storage class, so the values stay REAL and
exactness at rest needs PostgreSQL. What the rebuild buys is that a migrated
SQLite database declares the same money columns the models do, and that
``budgets`` keeps its single-period CHECK across the rebuild;
``tests/unit/test_exact_budget_schema_chain.py`` asserts both. That CHECK is why
``budgets`` is handed an explicit ``copy_from``: a reflected SQLite ``Table``
carries no CHECK constraints, so a reflection-driven rebuild would drop the rule
that keeps a budget from naming both a rolling duration and a calendar
alignment. It does not converge the whole schema: ``budget_reset_logs.user_id``
is still ``NOT NULL`` behind a plain foreign key on SQLite, because
``5911f4bbf98d`` skipped that engine, and resolving that is not this revision's
business.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3f8d1c6a4e7"
down_revision: str | Sequence[str] | None = "f3a5c7e9d1b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Spelled out rather than imported from ``gateway.models.money``: a migration
# describes the schema at one point in the chain, so it must not change meaning
# when the application's idea of the scale does.
_COST_TYPE = sa.Numeric(18, 6)
_FLOAT_TYPE = sa.Float()

# ``(table, column, nullable, server_default)``, in dependency order so a
# PostgreSQL rewrite touches a parent before its children.
#
# The default is *re-applied* rather than merely declared as existing. PostgreSQL
# keeps a default written against the column's old type and rewrites it as
# ``'0'::double precision`` when the column is retyped: still correct, but a
# migrated database would then describe the column differently from one built by
# ``create_all``, which is the same drift the SQLite rebuild exists to avoid.
_COUNTERS: tuple[tuple[str, str, bool, str | None], ...] = (
    ("budgets", "max_budget", True, None),
    ("users", "spend", False, None),
    ("users", "reserved", False, "0"),
    ("budget_reset_logs", "previous_spend", False, None),
    ("scoped_budgets", "current_spend", False, "0"),
    ("scoped_budgets", "reserved_spend", False, "0"),
)


def _budgets_table(numeric: bool) -> sa.Table:
    """``budgets`` as it stands, for SQLite's batch rebuild.

    Spelled out rather than reflected because of the CHECK: an Alembic batch
    rebuild recreates the table from the reflected ``Table``, and a reflected
    SQLite ``Table`` carries no CHECK constraints, so a reflection-driven
    rebuild would silently drop the one rule that keeps a budget from naming
    both a rolling duration and a calendar alignment. ``f3a5c7e9d1b4`` moved
    that rule here when it made this the only table holding a limit, so this is
    the table the care belongs to now.
    """
    money_type: sa.types.TypeEngine[object] = _COST_TYPE if numeric else _FLOAT_TYPE
    return sa.Table(
        "budgets",
        sa.MetaData(),
        sa.Column("budget_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("max_budget", money_type, nullable=True),
        sa.Column("budget_duration_sec", sa.Integer(), nullable=True),
        sa.Column("reset_alignment", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("budget_id"),
        sa.CheckConstraint(
            "NOT (budget_duration_sec IS NOT NULL AND reset_alignment IS NOT NULL)",
            name="ck_budgets_single_period_source",
        ),
    )


def _convert(to_numeric: bool) -> None:
    """Retype every counter, in whichever direction is being run."""
    is_sqlite = op.get_bind().dialect.name == "sqlite"
    new_type: sa.types.TypeEngine[object] = _COST_TYPE if to_numeric else _FLOAT_TYPE
    old_type: sa.types.TypeEngine[object] = _FLOAT_TYPE if to_numeric else _COST_TYPE
    using = "numeric" if to_numeric else "double precision"

    for table in ("budgets", "users", "budget_reset_logs", "scoped_budgets"):
        columns = [entry for entry in _COUNTERS if entry[0] == table]
        if is_sqlite:
            # The rebuild starts from the table as it is now, so ``copy_from``
            # describes the *old* column types.
            copy_from = _budgets_table(not to_numeric) if table == "budgets" else None
            with op.batch_alter_table(table, copy_from=copy_from) as batch_op:
                for _table, column, nullable, server_default in columns:
                    batch_op.alter_column(
                        column,
                        existing_type=old_type,
                        type_=new_type,
                        existing_nullable=nullable,
                        server_default=server_default,
                    )
            continue
        for _table, column, nullable, server_default in columns:
            op.alter_column(
                table,
                column,
                existing_type=old_type,
                type_=new_type,
                existing_nullable=nullable,
                server_default=server_default,
                postgresql_using=f"{column}::{using}",
            )


# Just past the column's ceiling: ``NUMERIC(18, 6)`` holds up to
# 999999999999.999999, so anything at or above 1e12 cannot convert. Written as a
# literal for the same reason the types above are: a migration must not change
# meaning when the application's idea of the scale does.
_OVERFLOWS_AT = "1000000000000"


def _refuse_amounts_the_column_cannot_hold() -> None:
    """Fail with a message naming the row, rather than with a raw ALTER error."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # Nothing to guard: SQLite has no numeric storage class, so the value
        # stays a REAL and the rebuild cannot overflow. Refusing here would be a
        # failure this engine would not otherwise have had.
        return
    for table, column, _nullable, _default in _COUNTERS:
        offending = bind.execute(
            # ``ABS``: the range is symmetric, so a counter at or below
            # -1e12 overflows just as a cap at or above +1e12 does, and a
            # one-sided guard would let the very case it exists for through.
            sa.text(f"SELECT count(*) FROM {table} WHERE ABS({column}) >= :ceiling"),  # noqa: S608 (literals above)
            {"ceiling": _OVERFLOWS_AT},
        ).scalar_one()
        if offending:
            raise RuntimeError(
                f"{offending} row(s) in {table}.{column} hold an amount of ${_OVERFLOWS_AT} or more in "
                f"magnitude, which NUMERIC(18, 6) cannot store. Correct or clear them, then migrate again."
            )


def upgrade() -> None:
    """Retype the budget counters as exact numerics."""
    _refuse_amounts_the_column_cannot_hold()
    _convert(to_numeric=True)


def downgrade() -> None:
    """Return the budget counters to binary floating point.

    Lossy in principle and not in practice: a micro-dollar amount within this
    schema's range round-trips through a double unchanged. What a downgrade
    really gives back is the drift, which will start accumulating again.
    """
    _convert(to_numeric=False)
