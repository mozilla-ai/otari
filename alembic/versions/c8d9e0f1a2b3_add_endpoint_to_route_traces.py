"""add endpoint to route traces

Revision ID: c8d9e0f1a2b3
Revises: b7c9d0e1f2a3
Create Date: 2026-05-27 13:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8d9e0f1a2b3"
down_revision: str | Sequence[str] | None = "b7c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "route_traces",
        sa.Column(
            "endpoint",
            sa.String(),
            nullable=False,
            server_default="/v1/chat/completions",
        ),
    )
    op.create_index(op.f("ix_route_traces_endpoint"), "route_traces", ["endpoint"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_route_traces_endpoint"), table_name="route_traces")
    op.drop_column("route_traces", "endpoint")
