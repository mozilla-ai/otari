"""Record each subagent model recommendation the gateway answers.

``subagent_decisions`` holds one row per answered recommendation: who asked,
the model the subagent would have started on, the model recommended and what
the decision call cost. The outcome columns are filled when the harness
reports what the subagent consumed. The task is never stored.

Revision ID: a7c9e1b3d5f8
Revises: 02003653d88c
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7c9e1b3d5f8"
down_revision: str | Sequence[str] | None = "02003653d88c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "subagent_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("harness", sa.String(length=16), nullable=False),
        sa.Column("agent_type", sa.String(length=128), nullable=False),
        sa.Column("requested_model", sa.String(length=256), nullable=True),
        sa.Column("parent_model", sa.String(length=256), nullable=False),
        sa.Column("baseline_model", sa.String(length=256), nullable=False),
        sa.Column("baseline_source", sa.String(length=16), nullable=False),
        sa.Column("recommended_model", sa.String(length=256), nullable=False),
        sa.Column("router_cost", sa.Numeric(18, 6), nullable=True),
        sa.Column("outcome_kind", sa.String(length=16), server_default="none", nullable=False),
        sa.Column("outcome_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ran_on", sa.String(length=256), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("input_tokens", sa.BigInteger(), nullable=True),
        sa.Column("output_tokens", sa.BigInteger(), nullable=True),
        sa.Column("cache_read_tokens", sa.BigInteger(), nullable=True),
        sa.Column("cache_write_tokens", sa.BigInteger(), nullable=True),
        sa.Column("baseline_model_key", sa.String(length=256), nullable=True),
        sa.Column("baseline_cost", sa.Numeric(18, 6), nullable=True),
        sa.Column("actual_cost", sa.Numeric(18, 6), nullable=True),
        sa.Column("unpriced_reason", sa.String(length=32), nullable=True),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_subagent_decisions_workspace_id_created_at", "subagent_decisions", ["workspace_id", "created_at"]
    )
    op.create_index("ix_subagent_decisions_user_id_created_at", "subagent_decisions", ["user_id", "created_at"])
    op.create_index("ix_subagent_decisions_api_key_id", "subagent_decisions", ["api_key_id"])


def downgrade() -> None:
    op.drop_index("ix_subagent_decisions_api_key_id", table_name="subagent_decisions")
    op.drop_index("ix_subagent_decisions_user_id_created_at", table_name="subagent_decisions")
    op.drop_index("ix_subagent_decisions_workspace_id_created_at", table_name="subagent_decisions")
    op.drop_table("subagent_decisions")
