"""Add ``workspace.trace_content_tool_output``, a workspace's own switch for keeping what tools returned.

On for every workspace, existing ones included: a workspace keeps tool output
only where it keeps content and the deployment's ``trace_content_tool_output``
allows it, so the switch can only narrow what the deployment keeps.

Revision ID: b3e5f7a9c1d4
Revises: d1145eb686f8
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b3e5f7a9c1d4"
down_revision: str | Sequence[str] | None = "d1145eb686f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workspace", sa.Column("trace_content_tool_output", sa.Boolean(), nullable=False, server_default=sa.true())
    )


def downgrade() -> None:
    op.drop_column("workspace", "trace_content_tool_output")
