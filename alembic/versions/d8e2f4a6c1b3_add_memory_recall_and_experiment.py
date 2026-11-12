"""Add automatic recall and the savings experiment to a workspace's memory switches.

- ``auto_recall``: the gateway searches memory for a request's newest user turn and
  hands the model what it finds, with no tool call.
- ``recall_min_similarity`` / ``recall_max_entries``: how alike an entry must be to
  be handed over, and how many at most.
- ``experiment_holdout_percent`` / ``experiment_started_at`` / ``experiment_ended_at``:
  the share of sessions served without memory, to measure what memory saves, and
  when that began and stopped. Kept once stopped, so its results still read.

Revision ID: d8e2f4a6c1b3
Revises: c4f7a9e2d1b8
Create Date: 2026-11-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d8e2f4a6c1b3"
down_revision: str | Sequence[str] | None = "c4f7a9e2d1b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MEMORY_WORKSPACE_SETTINGS = "memory_workspace_settings"


def upgrade() -> None:
    with op.batch_alter_table(MEMORY_WORKSPACE_SETTINGS) as batch:
        batch.add_column(sa.Column("auto_recall", sa.Boolean(), server_default=sa.false(), nullable=False))
        batch.add_column(sa.Column("recall_min_similarity", sa.Float(), server_default="0.65", nullable=False))
        batch.add_column(sa.Column("recall_max_entries", sa.Integer(), server_default="3", nullable=False))
        batch.add_column(sa.Column("experiment_holdout_percent", sa.Integer(), server_default="0", nullable=False))
        batch.add_column(sa.Column("experiment_started_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("experiment_ended_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_check_constraint(
            "check_memory_settings_recall_similarity", "recall_min_similarity BETWEEN 0.3 AND 0.99"
        )
        batch.create_check_constraint("check_memory_settings_recall_entries", "recall_max_entries BETWEEN 1 AND 5")
        batch.create_check_constraint(
            "check_memory_settings_experiment_holdout", "experiment_holdout_percent BETWEEN 0 AND 50"
        )


def downgrade() -> None:
    with op.batch_alter_table(MEMORY_WORKSPACE_SETTINGS) as batch:
        batch.drop_constraint("check_memory_settings_experiment_holdout", type_="check")
        batch.drop_constraint("check_memory_settings_recall_entries", type_="check")
        batch.drop_constraint("check_memory_settings_recall_similarity", type_="check")
        batch.drop_column("experiment_ended_at")
        batch.drop_column("experiment_started_at")
        batch.drop_column("experiment_holdout_percent")
        batch.drop_column("recall_max_entries")
        batch.drop_column("recall_min_similarity")
        batch.drop_column("auto_recall")
