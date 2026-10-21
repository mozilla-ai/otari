"""Add the update check state table.

One row, pinned by a check constraint, holding the install id, the operator's
answer and the schedule every replica's worker polls; the conditional update
that moves its ``next_check_at`` forward is what lets exactly one replica send
a check.

Revision ID: d7f1b3c5e9a2
Revises: e8b2d4f6a0c3
Create Date: 2026-10-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d7f1b3c5e9a2"
down_revision: str | Sequence[str] | None = "e8b2d4f6a0c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "update_check_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("install_id", sa.Uuid(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=True),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_check_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failures", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("checked_version", sa.String(), nullable=True),
        sa.Column("latest_version", sa.String(), nullable=True),
        sa.Column("release_url", sa.String(), nullable=True),
        sa.Column("advisories", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_update_check_state_one_row"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("update_check_state")
