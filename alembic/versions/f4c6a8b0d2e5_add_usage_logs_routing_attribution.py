"""Add routing attribution to usage_logs.

Makes a routing decision legible after the fact. Without these columns a
tier-down and a fallover are indistinguishable from an ordinary request, which
would leave the feature unfalsifiable: "which model actually served this, and
why?" could only be answered from logs.

``request_group_id`` correlates the rows belonging to one request. It is needed
because a request that fails over writes more than one row: the attempt that
served, plus one per absorbed failure. Those extra rows carry
``status='absorbed'``, deliberately not ``'error'``, because every error metric in
the product counts ``status == 'error'`` and a *working* fallback chain must not
read as an outage. ``request_count`` excludes them for the same reason: a request
that took two attempts is still one request.

All columns are nullable with no backfill. A row that predates routing has no
policy, and inventing one would be a lie; null reads correctly as "not routed
through a policy".

Revision ID: f4c6a8b0d2e5
Revises: e8b1d3f5a7c9
Create Date: 2026-08-04 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f4c6a8b0d2e5"
down_revision: str | Sequence[str] | None = "e8b1d3f5a7c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("usage_logs", sa.Column("policy_name", sa.String(), nullable=True))
    op.add_column("usage_logs", sa.Column("selection_reason", sa.String(), nullable=True))
    op.add_column("usage_logs", sa.Column("attempt_position", sa.Integer(), nullable=True))
    op.add_column("usage_logs", sa.Column("attempt_count", sa.Integer(), nullable=True))
    op.add_column("usage_logs", sa.Column("request_group_id", sa.String(), nullable=True))
    # Indexed because the dashboard filters and groups by policy, and correlates a
    # request's attempts. Both are read paths on a table that is otherwise
    # write-heavy, so they are the two worth paying for; `model` is deliberately
    # left unindexed for the same reason (high cardinality, see UsageLog).
    op.create_index("ix_usage_logs_policy_name", "usage_logs", ["policy_name"])
    op.create_index("ix_usage_logs_request_group_id", "usage_logs", ["request_group_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_usage_logs_request_group_id", table_name="usage_logs")
    op.drop_index("ix_usage_logs_policy_name", table_name="usage_logs")
    op.drop_column("usage_logs", "request_group_id")
    op.drop_column("usage_logs", "attempt_count")
    op.drop_column("usage_logs", "attempt_position")
    op.drop_column("usage_logs", "selection_reason")
    op.drop_column("usage_logs", "policy_name")
