"""Add the request id and the routing decision to usage_logs.

``request_id`` is the value a caller already receives as ``Otari-Request-ID``,
which until now named nothing a caller could point back at. Stored, it lets a
caller refer to a request after the fact, which is what rating a routed
response needs. Indexed, because that lookup is by request id.

``routing_backend`` and ``routing_decision_id`` record which learning router
backend decided the request and that backend's own token for the decision, so a
rating can be handed back to the backend that made it.

All nullable with no backfill: no earlier row has either, and null reads
correctly as "not recorded".

Revision ID: c3e5a7b9d1f2
Revises: b7d2e9f4a6c1
Create Date: 2026-10-01 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3e5a7b9d1f2"
down_revision: str | Sequence[str] | None = "b7d2e9f4a6c1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("usage_logs", sa.Column("request_id", sa.String(), nullable=True))
    op.add_column("usage_logs", sa.Column("routing_backend", sa.String(), nullable=True))
    op.add_column("usage_logs", sa.Column("routing_decision_id", sa.String(), nullable=True))
    op.create_index("ix_usage_logs_request_id", "usage_logs", ["request_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_usage_logs_request_id", table_name="usage_logs")
    op.drop_column("usage_logs", "routing_decision_id")
    op.drop_column("usage_logs", "routing_backend")
    op.drop_column("usage_logs", "request_id")
