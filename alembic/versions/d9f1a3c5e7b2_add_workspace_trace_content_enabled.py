"""Add the workspace's own switch for keeping trace content.

Off for every workspace, existing ones included: a workspace keeps what its
agent sessions were about only once its managers turn it on, and only where
the deployment's ``trace_content_enabled`` allows it.

Revision ID: d9f1a3c5e7b2
Revises: c8e0f2a4b6d3
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d9f1a3c5e7b2"
down_revision: str | Sequence[str] | None = "c8e0f2a4b6d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workspace", sa.Column("trace_content_enabled", sa.Boolean(), nullable=False, server_default=sa.false())
    )


def downgrade() -> None:
    op.drop_column("workspace", "trace_content_enabled")
