"""add usage provenance (source) and budget exclusion

Revision ID: b8c1d2e3f4a5
Revises: a9c1e3b5d7f9
Create Date: 2026-07-23 13:40:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8c1d2e3f4a5"
down_revision: str | Sequence[str] | None = "a9c1e3b5d7f9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add usage_logs provenance columns + api_keys.exclude_from_budget."""

    # New columns land nullable with a server_default so existing rows backfill on
    # ADD COLUMN (both SQLite and Postgres). The columns that must end up NOT NULL
    # are tightened inside the batch block below.
    op.add_column(
        "usage_logs",
        sa.Column("source", sa.String(), nullable=True, server_default="gateway"),
    )
    op.add_column("usage_logs", sa.Column("source_event_id", sa.String(), nullable=True))
    op.add_column("usage_logs", sa.Column("source_label", sa.String(), nullable=True))
    op.add_column(
        "usage_logs",
        sa.Column("counts_toward_budget", sa.Boolean(), nullable=True, server_default=sa.true()),
    )
    op.add_column(
        "api_keys",
        sa.Column("exclude_from_budget", sa.Boolean(), nullable=True, server_default=sa.false()),
    )

    # Defensive backfill (redundant with server_default, harmless): guarantee no NULLs
    # remain before we flip the columns to NOT NULL. sa.true()/sa.false() render the
    # dialect-correct boolean literal (Postgres rejects integer 1/0 for a boolean).
    conn = op.get_bind()
    conn.execute(sa.text("UPDATE usage_logs SET source = 'gateway' WHERE source IS NULL"))
    conn.execute(
        sa.update(sa.table("usage_logs", sa.column("counts_toward_budget", sa.Boolean())))
        .where(sa.column("counts_toward_budget").is_(None))
        .values(counts_toward_budget=True)
    )
    conn.execute(
        sa.update(sa.table("api_keys", sa.column("exclude_from_budget", sa.Boolean())))
        .where(sa.column("exclude_from_budget").is_(None))
        .values(exclude_from_budget=False)
    )

    # NOT NULL + unique constraint. batch_alter_table rebuilds the table on SQLite,
    # which is required for adding a UniqueConstraint and altering nullability there.
    with op.batch_alter_table("usage_logs") as batch_op:
        batch_op.alter_column("source", existing_type=sa.String(), nullable=False)
        batch_op.alter_column("counts_toward_budget", existing_type=sa.Boolean(), nullable=False)
        batch_op.create_unique_constraint(
            "uq_usage_logs_source_event",
            ["source", "source_event_id"],
        )

    with op.batch_alter_table("api_keys") as batch_op:
        batch_op.alter_column("exclude_from_budget", existing_type=sa.Boolean(), nullable=False)

    op.create_index("ix_usage_logs_source", "usage_logs", ["source"])


def downgrade() -> None:
    """Drop the provenance columns, budget-exclusion flag, and their constraints."""

    op.drop_index("ix_usage_logs_source", table_name="usage_logs")

    with op.batch_alter_table("usage_logs") as batch_op:
        batch_op.drop_constraint("uq_usage_logs_source_event", type_="unique")
        batch_op.drop_column("counts_toward_budget")
        batch_op.drop_column("source_label")
        batch_op.drop_column("source_event_id")
        batch_op.drop_column("source")

    with op.batch_alter_table("api_keys") as batch_op:
        batch_op.drop_column("exclude_from_budget")
