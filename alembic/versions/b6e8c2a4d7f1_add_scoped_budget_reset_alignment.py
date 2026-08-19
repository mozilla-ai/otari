"""Let a scoped budget reset on a UTC calendar boundary.

``scoped_budgets`` expressed a period only as ``budget_duration_sec``, seconds
added to the last reset. A calendar month is not a fixed number of seconds, so a
monthly cap could not be expressed at all: 2592000 is a different product, since
a year holds 12.17 thirty-day periods against 12 months.

``reset_alignment`` is the other way to carry a period, and a row holds one or
the other. The CHECK is what keeps that true: without it ``(86400,
calendar_month)`` would be storable, and the two columns would encode one concept
with an implicit "ignored when" rule. A plain string rather than a database enum,
for the same reason ``scope_type`` is one: a new alignment then needs no enum
migration.

Existing rows keep ``reset_alignment`` NULL and behave exactly as before. The
column is added through ``batch_alter_table`` because SQLite has no ``ALTER TABLE
... ADD CONSTRAINT``, and ``copy_from`` is what carries the table's three indexes
(two of them partial) through SQLite's rebuild, which reflection could not
recover on its own.

Revision ID: b6e8c2a4d7f1
Revises: a3c7e1b9d5f2
Create Date: 2026-08-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b6e8c2a4d7f1"
down_revision: str | Sequence[str] | None = "a3c7e1b9d5f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CHECK = "ck_scoped_budgets_single_period_source"
_CHECK_SQL = "NOT (budget_duration_sec IS NOT NULL AND reset_alignment IS NOT NULL)"


def _scoped_budgets(*, with_alignment: bool) -> sa.Table:
    """The table as it stands on either side of this revision.

    Passed as ``copy_from`` so SQLite's rebuild keeps every index and constraint,
    including the two partial unique indexes it cannot reflect.
    """
    columns = [
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("scope_type", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("provider_key_id", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("max_budget", sa.Float(), nullable=True),
        sa.Column("current_spend", sa.Float(), nullable=False, server_default="0"),
        sa.Column("reserved_spend", sa.Float(), nullable=False, server_default="0"),
        sa.Column("budget_duration_sec", sa.Integer(), nullable=True),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]
    if with_alignment:
        columns.append(sa.Column("reset_alignment", sa.String(), nullable=True))

    table = sa.Table(
        "scoped_budgets",
        sa.MetaData(),
        *columns,
        sa.PrimaryKeyConstraint("id"),
        *([sa.CheckConstraint(_CHECK_SQL, name=_CHECK)] if with_alignment else []),
    )
    sa.Index(
        "uq_scoped_budgets_scope_with_key",
        table.c.scope_type,
        table.c.scope_id,
        table.c.provider_key_id,
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NOT NULL"),
        sqlite_where=sa.text("provider_key_id IS NOT NULL"),
    )
    sa.Index(
        "uq_scoped_budgets_scope_no_key",
        table.c.scope_type,
        table.c.scope_id,
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NULL"),
        sqlite_where=sa.text("provider_key_id IS NULL"),
    )
    sa.Index("ix_scoped_budgets_scope", table.c.scope_type, table.c.scope_id)
    return table


def upgrade() -> None:
    with op.batch_alter_table("scoped_budgets", copy_from=_scoped_budgets(with_alignment=False)) as batch_op:
        batch_op.add_column(sa.Column("reset_alignment", sa.String(), nullable=True))
        batch_op.create_check_constraint(_CHECK, _CHECK_SQL)


def downgrade() -> None:
    with op.batch_alter_table("scoped_budgets", copy_from=_scoped_budgets(with_alignment=True)) as batch_op:
        batch_op.drop_constraint(_CHECK, type_="check")
        batch_op.drop_column("reset_alignment")
