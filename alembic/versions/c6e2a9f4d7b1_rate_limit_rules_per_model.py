"""Let a stored rate-limit rule be counted per model.

Adds ``models``, the ``instance:model`` names a ``per: model`` rule limits, and
lets the ``per`` check accept ``model``. The constraint is replaced through a
batch rebuild, since SQLite cannot alter one in place.

Revision ID: c6e2a9f4d7b1
Revises: 615fa323e931
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c6e2a9f4d7b1"
down_revision: str | Sequence[str] | None = "615fa323e931"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "rate_limit_rules"
_CHECK = "ck_rate_limit_rules_per"


def _table(per_values: str, *, with_models: bool) -> sa.Table:
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
    ]
    if with_models:
        columns.append(sa.Column("models", sa.JSON(), nullable=True))
    return sa.Table(
        _TABLE,
        sa.MetaData(),
        *columns,
        sa.PrimaryKeyConstraint("name"),
        sa.CheckConstraint(f"per IN ({per_values})", name=_CHECK),
    )


def upgrade() -> None:
    """Add ``models`` and accept ``per = 'model'``."""
    with op.batch_alter_table(_TABLE, copy_from=_table("'deployment', 'key', 'user'", with_models=False)) as batch:
        batch.add_column(sa.Column("models", sa.JSON(), nullable=True))
        batch.drop_constraint(_CHECK, type_="check")
        batch.create_check_constraint(_CHECK, "per IN ('deployment', 'key', 'user', 'model')")


def downgrade() -> None:
    """Drop the per-model rules, then ``models``; the rules they held stop applying."""
    op.execute(sa.text("DELETE FROM rate_limit_rules WHERE per = 'model'"))
    with op.batch_alter_table(
        _TABLE, copy_from=_table("'deployment', 'key', 'user', 'model'", with_models=True)
    ) as batch:
        batch.drop_constraint(_CHECK, type_="check")
        batch.create_check_constraint(_CHECK, "per IN ('deployment', 'key', 'user')")
        batch.drop_column("models")
