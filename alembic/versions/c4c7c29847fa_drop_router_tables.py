"""Drop the learned router's tables.

``routing_memory`` held prompt embeddings and ``router_preferences`` the scored
prompts behind them, both written only by the routing-memory routes. The router
and those routes are gone, so nothing reads or writes either table.

The downgrade recreates both empty, in the shape ``d2f5b8c0e4a7`` left them,
minus the ``workspace_id`` server default: that default was the deployment's
default workspace id, there only to backfill existing rows, and an empty table
has none.

Revision ID: c4c7c29847fa
Revises: 9964faffc86b
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4c7c29847fa"
down_revision: str | Sequence[str] | None = "9964faffc86b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEXES: dict[str, tuple[tuple[str, list[str]], ...]] = {
    "routing_memory": (
        ("ix_routing_memory_user_id", ["user_id"]),
        ("ix_routing_memory_task_id", ["task_id"]),
        ("ix_routing_memory_created_at", ["created_at"]),
        ("ix_routing_memory_workspace_id", ["workspace_id"]),
        ("ix_routing_memory_workspace_user_model", ["workspace_id", "user_id", "embedding_model"]),
        ("ix_routing_memory_workspace_user_created", ["workspace_id", "user_id", "created_at"]),
        (
            "ix_routing_memory_workspace_user_model_task",
            ["workspace_id", "user_id", "embedding_model", "task_id"],
        ),
    ),
    "router_preferences": (
        ("ix_router_preferences_user_id", ["user_id"]),
        ("ix_router_preferences_created_at", ["created_at"]),
        ("ix_router_preferences_workspace_id", ["workspace_id"]),
        ("ix_router_preferences_workspace_user_created", ["workspace_id", "user_id", "created_at"]),
    ),
}


def upgrade() -> None:
    for table, indexes in _INDEXES.items():
        for name, _columns in indexes:
            op.drop_index(name, table_name=table)
        op.drop_table(table)


def downgrade() -> None:
    op.create_table(
        "routing_memory",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=False),
        sa.Column("qualities", sa.JSON(), nullable=False),
        sa.Column("task_id", sa.String(), nullable=True),
        sa.Column("label_source", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspace.id"], name="fk_routing_memory_workspace_id", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "router_preferences",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("prompt", sa.String(), nullable=False),
        sa.Column("task_id", sa.String(), nullable=True),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("label_source", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspace.id"], name="fk_router_preferences_workspace_id", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    for table, indexes in _INDEXES.items():
        for name, columns in indexes:
            op.create_index(name, table, columns)
