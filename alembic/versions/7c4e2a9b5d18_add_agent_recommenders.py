"""Store the decision models that recommend a subagent's model.

``agent_recommenders`` holds the routers the Subagent router page adds: the
decision provider and model, its key encrypted, and the two token rates a
recommendation is charged at. A partial unique index lets one row be switched
on at a time.

Revision ID: 7c4e2a9b5d18
Revises: d9f1a3c5e7b2
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7c4e2a9b5d18"
down_revision: str | Sequence[str] | None = "d9f1a3c5e7b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_recommenders",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("api_base", sa.String(length=1024), nullable=True),
        sa.Column("encrypted_api_key", sa.Text(), nullable=True),
        sa.Column("api_key_last4", sa.String(length=4), nullable=True),
        sa.Column("input_price_per_million", sa.Numeric(18, 8), server_default="0", nullable=False),
        sa.Column("output_price_per_million", sa.Numeric(18, 8), server_default="0", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
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
