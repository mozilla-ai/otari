"""Let a workspace skip the relevance check when an entry is unmistakably alike.

- ``recall_rerank_skip_similarity``: where the most alike entry reaches it,
  automatic recall hands over by similarity without calling the rerank model.
  0.3-0.99; null, the default, checks every turn.

Revision ID: 69080982e58d
Revises: f5b7d9e1a3c6
Create Date: 2026-11-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "69080982e58d"
down_revision: str | Sequence[str] | None = "f5b7d9e1a3c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MEMORY_WORKSPACE_SETTINGS = "memory_workspace_settings"


def upgrade() -> None:
    with op.batch_alter_table(MEMORY_WORKSPACE_SETTINGS) as batch:
        batch.add_column(sa.Column("recall_rerank_skip_similarity", sa.Float(), nullable=True))
        batch.create_check_constraint(
            "check_memory_settings_rerank_skip",
            "recall_rerank_skip_similarity IS NULL OR recall_rerank_skip_similarity BETWEEN 0.3 AND 0.99",
        )


def downgrade() -> None:
    with op.batch_alter_table(MEMORY_WORKSPACE_SETTINGS) as batch:
        batch.drop_constraint("check_memory_settings_rerank_skip", type_="check")
        batch.drop_column("recall_rerank_skip_similarity")
