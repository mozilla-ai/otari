"""add routing_memory and router_preferences tables

Revision ID: d4e6f8a0b2c4
Revises: c3d5e7f9a1b3
Create Date: 2026-06-19 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e6f8a0b2c4"
down_revision: str | Sequence[str] | None = "c3d5e7f9a1b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the kNN router's vector store and preference-audit tables."""

    op.create_table(
        "routing_memory",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("quality", sa.Float(), nullable=False),
        sa.Column("cost", sa.Float(), nullable=False),
        sa.Column("task_id", sa.String(), nullable=True),
        sa.Column("label_source", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_routing_memory_tenant_id", "routing_memory", ["tenant_id"])
    op.create_index("ix_routing_memory_task_id", "routing_memory", ["task_id"])
    op.create_index("ix_routing_memory_created_at", "routing_memory", ["created_at"])
    op.create_index("ix_routing_memory_tenant_model", "routing_memory", ["tenant_id", "embedding_model"])
    op.create_index("ix_routing_memory_tenant_created", "routing_memory", ["tenant_id", "created_at"])

    op.create_table(
        "router_preferences",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("prompt", sa.String(), nullable=False),
        sa.Column("task_id", sa.String(), nullable=True),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("label_source", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_router_preferences_tenant_id", "router_preferences", ["tenant_id"])
    op.create_index("ix_router_preferences_created_at", "router_preferences", ["created_at"])
    op.create_index("ix_router_preferences_tenant_created", "router_preferences", ["tenant_id", "created_at"])


def downgrade() -> None:
    """Drop the router tables."""

    op.drop_index("ix_router_preferences_tenant_created", table_name="router_preferences")
    op.drop_index("ix_router_preferences_created_at", table_name="router_preferences")
    op.drop_index("ix_router_preferences_tenant_id", table_name="router_preferences")
    op.drop_table("router_preferences")

    op.drop_index("ix_routing_memory_tenant_created", table_name="routing_memory")
    op.drop_index("ix_routing_memory_tenant_model", table_name="routing_memory")
    op.drop_index("ix_routing_memory_created_at", table_name="routing_memory")
    op.drop_index("ix_routing_memory_task_id", table_name="routing_memory")
    op.drop_index("ix_routing_memory_tenant_id", table_name="routing_memory")
    op.drop_table("routing_memory")
