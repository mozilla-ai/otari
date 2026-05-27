"""add project tags to usage logs

Revision ID: d9e0f1a2b3c4
Revises: c8d9e0f1a2b3
Create Date: 2026-05-27 14:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d9e0f1a2b3c4"
down_revision: str | Sequence[str] | None = "c8d9e0f1a2b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("usage_logs") as batch_op:
        batch_op.add_column(sa.Column("project_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("tags", sa.JSON(), nullable=True))
        batch_op.create_foreign_key(
            "fk_usage_logs_project_id_projects",
            "projects",
            ["project_id"],
            ["project_id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(op.f("ix_usage_logs_project_id"), ["project_id"], unique=False)
        batch_op.create_index(
            "ix_usage_logs_project_id_timestamp",
            ["project_id", "timestamp"],
            unique=False,
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("usage_logs") as batch_op:
        batch_op.drop_index("ix_usage_logs_project_id_timestamp")
        batch_op.drop_index(op.f("ix_usage_logs_project_id"))
        batch_op.drop_constraint("fk_usage_logs_project_id_projects", type_="foreignkey")
        batch_op.drop_column("tags")
        batch_op.drop_column("project_id")
