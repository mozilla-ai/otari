"""Add the rate-limit rules an operator manages from the dashboard.

Each row is one ``rate_limits`` rule, counted beside the rules in config.yml.
Nothing references the table, so it needs no foreign keys.

Revision ID: e5b9d3f7a1c2
Revises: a4d8e2f6b0c3
Create Date: 2026-10-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5b9d3f7a1c2"
down_revision: str | Sequence[str] | None = "a4d8e2f6b0c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create ``rate_limit_rules``."""
    op.create_table(
        "rate_limit_rules",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("per", sa.String(), nullable=False),
        sa.Column("rpm", sa.Integer(), nullable=True),
        sa.Column("tpm", sa.Integer(), nullable=True),
        sa.Column("max_concurrent", sa.Integer(), nullable=True),
        sa.Column("lease_sec", sa.Float(), nullable=False, server_default="900"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("name"),
        sa.CheckConstraint("per IN ('deployment', 'key', 'user')", name="ck_rate_limit_rules_per"),
    )


def downgrade() -> None:
    """Drop ``rate_limit_rules``; the rules it held stop applying."""
    op.drop_table("rate_limit_rules")
