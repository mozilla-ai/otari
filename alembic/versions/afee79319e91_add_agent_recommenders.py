"""Add the agent_recommenders table: the decision models an operator adds from the dashboard.

At most one row is switched on, which a partial unique index enforces. While
the table is empty, ``agent_recommender_model`` in the configuration answers.

Revision ID: afee79319e91
Revises: a8c2e4f6b0d1
Create Date: 2026-10-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "afee79319e91"
down_revision: str | Sequence[str] | None = "a8c2e4f6b0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RATE_TYPE = sa.Numeric(18, 8)


def upgrade() -> None:
    op.create_table(
        "agent_recommenders",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("api_base", sa.String(length=1024), nullable=True),
        sa.Column("encrypted_api_key", sa.Text(), nullable=True),
        sa.Column("api_key_last4", sa.String(length=4), nullable=True),
        sa.Column("input_price_per_million", _RATE_TYPE, server_default="0", nullable=False),
        sa.Column("output_price_per_million", _RATE_TYPE, server_default="0", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_agent_recommenders_enabled",
        "agent_recommenders",
        ["enabled"],
        unique=True,
        sqlite_where=sa.text("enabled"),
        postgresql_where=sa.text("enabled"),
    )


def downgrade() -> None:
    op.drop_index("uq_agent_recommenders_enabled", table_name="agent_recommenders")
    op.drop_table("agent_recommenders")
