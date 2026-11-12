"""Add the relevance check to a workspace's automatic recall.

- ``recall_rerank``: the deployment's rerank model scores the entries most alike to
  the turn, and only those relevant enough are handed over.
- ``recall_min_relevance``: the score an entry must reach, 0.05-0.99 (0.8).

Revision ID: e3a5c7f9b2d4
Revises: d8e2f4a6c1b3
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e3a5c7f9b2d4"
down_revision: str | Sequence[str] | None = "d8e2f4a6c1b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MEMORY_WORKSPACE_SETTINGS = "memory_workspace_settings"


def upgrade() -> None:
    with op.batch_alter_table(MEMORY_WORKSPACE_SETTINGS) as batch:
        batch.add_column(sa.Column("recall_rerank", sa.Boolean(), server_default=sa.false(), nullable=False))
        batch.add_column(sa.Column("recall_min_relevance", sa.Float(), server_default="0.8", nullable=False))
        batch.create_check_constraint(
            "check_memory_settings_recall_relevance", "recall_min_relevance BETWEEN 0.05 AND 0.99"
        )


def downgrade() -> None:
    with op.batch_alter_table(MEMORY_WORKSPACE_SETTINGS) as batch:
        batch.drop_constraint("check_memory_settings_recall_relevance", type_="check")
        batch.drop_column("recall_min_relevance")
        batch.drop_column("recall_rerank")
