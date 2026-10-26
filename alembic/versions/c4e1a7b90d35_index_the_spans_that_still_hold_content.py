"""Index the spans of a workspace that still hold content.

``DELETE /api/v1/workspaces/{id}/trace-content`` clears content a batch at a
time, selecting on ``content_sealed IS NOT NULL``. The index is partial on the
same predicate, so it covers those rows alone and a cleared row leaves it:
each batch then costs what it clears rather than the workspace's whole history
of spans.

Revision ID: c4e1a7b90d35
Revises: d9f1a3c5e7b2
Create Date: 2026-10-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e1a7b90d35"
down_revision: str | Sequence[str] | None = "d9f1a3c5e7b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "trace_spans"
_INDEX = "ix_trace_spans_workspace_sealed"
_HELD = sa.text("content_sealed IS NOT NULL")


def upgrade() -> None:
    op.create_index(_INDEX, _TABLE, ["workspace_id"], postgresql_where=_HELD, sqlite_where=_HELD)


def downgrade() -> None:
    op.drop_index(_INDEX, table_name=_TABLE)
