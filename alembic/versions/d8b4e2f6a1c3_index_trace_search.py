"""Index the trace columns a session search matches a prefix of.

Search matches a prefix of a trace's id or of its name. On PostgreSQL a
``text_pattern_ops`` index serves a prefix ``LIKE`` whatever the database's
collation, and indexing both columns lets the planner answer the search's ``OR``
from the two together. SQLite ignores the operator class and indexes the columns
plainly.

Revision ID: d8b4e2f6a1c3
Revises: a8c2e4f6b0d1
Create Date: 2026-10-19
"""

from collections.abc import Sequence

from alembic import op

revision: str = "d8b4e2f6a1c3"
down_revision: str | Sequence[str] | None = "a8c2e4f6b0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_traces_workspace_trace_id_pattern",
        "traces",
        ["workspace_id", "trace_id"],
        postgresql_ops={"trace_id": "text_pattern_ops"},
    )
    op.create_index(
        "ix_traces_workspace_name", "traces", ["workspace_id", "name"], postgresql_ops={"name": "text_pattern_ops"}
    )


def downgrade() -> None:
    op.drop_index("ix_traces_workspace_name", table_name="traces")
    op.drop_index("ix_traces_workspace_trace_id_pattern", table_name="traces")
