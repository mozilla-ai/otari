"""Add trace_spans.content_sealed.

What an instrumented coding agent reports a span was about (the prompt that
opened a turn, a tool's command or file, what a guardrail gate checked),
encrypted with OTARI_SECRET_KEY and kept only where a deployment turns on
trace_content_enabled. Null everywhere else.

Revision ID: c8e0f2a4b6d3
Revises: c3e5a7b9d1f2
Create Date: 2026-10-18 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8e0f2a4b6d3"
down_revision: str | Sequence[str] | None = "c3e5a7b9d1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("trace_spans", sa.Column("content_sealed", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("trace_spans", "content_sealed")
