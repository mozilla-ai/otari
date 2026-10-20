"""Count the errors that fail a trace's session apart from its other errors.

A tool's error goes back to the model, which usually recovers, so the session
series counts a trace as failed only for an error other than a tool's.
``failing_error_count`` is the part of ``error_count`` that fails the session,
backfilled from the spans each trace already holds.

Revision ID: a8c2e4f6b0d1
Revises: f0f9d05dafc6
Create Date: 2026-10-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a8c2e4f6b0d1"
down_revision: str | Sequence[str] | None = "f0f9d05dafc6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("traces", sa.Column("failing_error_count", sa.Integer(), server_default="0", nullable=False))
    op.execute(
        sa.text(
            "UPDATE traces SET failing_error_count = ("
            "SELECT COUNT(*) FROM trace_spans"
            " WHERE trace_spans.workspace_id = traces.workspace_id"
            " AND trace_spans.trace_id = traces.trace_id"
            " AND trace_spans.kind <> 'tool' AND trace_spans.outcome = 'error'"
            " AND NOT trace_spans.recovered"
            ") WHERE error_count > 0"
        )
    )


def downgrade() -> None:
    op.drop_column("traces", "failing_error_count")
