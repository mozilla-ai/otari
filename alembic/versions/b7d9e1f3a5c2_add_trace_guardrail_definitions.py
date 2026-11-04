"""Add trace_guardrail_definitions.

Each version of an agent guardrail a coding agent's hook checked, keyed by its
fingerprint, so a gate span can say what its gate is for. Definitions go with
their workspace.

Revision ID: b7d9e1f3a5c2
Revises: 7e02300efc98
Create Date: 2026-10-18 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7d9e1f3a5c2"
down_revision: str | Sequence[str] | None = "7e02300efc98"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "trace_guardrail_definitions",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("digest", sa.String(length=64), nullable=False),
        sa.Column("gates", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("workspace_id", "digest"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("trace_guardrail_definitions")
