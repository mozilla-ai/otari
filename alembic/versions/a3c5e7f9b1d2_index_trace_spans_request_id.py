"""Index trace_spans by request id, so a usage row finds the session it was traced into.

Revision ID: a3c5e7f9b1d2
Revises: c4e8a2d6f0b3
Create Date: 2026-10-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a3c5e7f9b1d2"
down_revision: str | Sequence[str] | None = "c4e8a2d6f0b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_trace_spans_request_id", "trace_spans", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_trace_spans_request_id", table_name="trace_spans")
