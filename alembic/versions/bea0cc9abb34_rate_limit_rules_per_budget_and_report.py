"""Let a stored rate-limit rule count per budget, and report instead of refuse.

Adds ``budgets``, the budgets a ``per: budget`` rule counts (null for every
budget), and ``mode`` (``enforce`` or ``report``), and lets the ``per`` check
accept ``budget``. The constraint is replaced through a batch rebuild, since
SQLite cannot alter one in place.

Revision ID: bea0cc9abb34
Revises: c8e0f2a4b6d3
Create Date: 2026-10-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "bea0cc9abb34"
down_revision: str | Sequence[str] | None = "c8e0f2a4b6d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "rate_limit_rules"
_CHECK = "ck_rate_limit_rules_per"
_PER_BEFORE = "'deployment', 'key', 'user', 'model'"
_PER_AFTER = "'deployment', 'key', 'user', 'model', 'budget'"


def _table(per_values: str, *, with_budgets: bool) -> sa.Table:
    """``rate_limit_rules`` as it stands, for SQLite's batch rebuild."""
    columns: list[sa.Column[object]] = [
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("per", sa.String(), nullable=False),
        sa.Column("rpm", sa.Integer(), nullable=True),
        sa.Column("tpm", sa.Integer(), nullable=True),
        sa.Column("max_concurrent", sa.Integer(), nullable=True),
        sa.Column("lease_sec", sa.Float(), nullable=False, server_default="900"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("models", sa.JSON(), nullable=True),
        sa.Column("tpm_admission", sa.String(), nullable=False, server_default="used"),
    ]
    if with_budgets:
        columns.append(sa.Column("budgets", sa.JSON(), nullable=True))
        columns.append(sa.Column("mode", sa.String(), nullable=False, server_default="enforce"))
    return sa.Table(
        _TABLE,
        sa.MetaData(),
        *columns,
        sa.PrimaryKeyConstraint("name"),
        sa.CheckConstraint(f"per IN ({per_values})", name=_CHECK),
    )


def upgrade() -> None:
    with op.batch_alter_table(_TABLE, copy_from=_table(_PER_BEFORE, with_budgets=False)) as batch:
        batch.add_column(sa.Column("budgets", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("mode", sa.String(), nullable=False, server_default="enforce"))
        batch.drop_constraint(_CHECK, type_="check")
        batch.create_check_constraint(_CHECK, f"per IN ({_PER_AFTER})")


def downgrade() -> None:
    """A report rule comes back enforced, since the schema before this has no ``mode`` to keep it reporting."""
    op.execute(sa.text("DELETE FROM rate_limit_rules WHERE per = 'budget'"))
    with op.batch_alter_table(_TABLE, copy_from=_table(_PER_AFTER, with_budgets=True)) as batch:
        batch.drop_constraint(_CHECK, type_="check")
        batch.create_check_constraint(_CHECK, f"per IN ({_PER_BEFORE})")
        batch.drop_column("mode")
        batch.drop_column("budgets")
