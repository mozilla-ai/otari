"""Add traces and trace_spans.

Agent traces: one ``traces`` row per agent session, with the totals the session
list reads, and its ``trace_spans``. Both key on the workspace first, so an id a
client chose cannot reach another tenant's rows. A span cascades with its trace,
and a trace with its workspace or its owner, because a trace is a projection
rather than a billing record: ``usage_logs`` keeps the money.

Revision ID: 12d51fa2acc0
Revises: a9d3e5f7b1c2
Create Date: 2026-10-07 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "12d51fa2acc0"
down_revision: str | Sequence[str] | None = "a9d3e5f7b1c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ID = sa.String(64)
_COST_TYPE = sa.Numeric(18, 6)


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "traces",
        sa.Column("workspace_id", sa.Uuid(), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
        sa.Column("trace_id", _ID, nullable=False),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=True),
        sa.Column("api_key_id", sa.String(), sa.ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True),
        sa.Column("session_source", sa.String(), nullable=False),
        sa.Column("harness", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("step_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("span_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("input_tokens", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("cost_snapshot", _COST_TYPE, server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "trace_id"),
    )
    op.create_index("ix_traces_user_id", "traces", ["user_id"])
    op.create_index("ix_traces_api_key_id", "traces", ["api_key_id"])
    op.create_index("ix_traces_workspace_last_activity", "traces", ["workspace_id", "last_activity_at"])
    op.create_index("ix_traces_last_activity", "traces", ["last_activity_at"])

    op.create_table(
        "trace_spans",
        sa.Column("workspace_id", sa.Uuid(), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
        sa.Column("trace_id", _ID, nullable=False),
        sa.Column("span_id", _ID, nullable=False),
        sa.Column("parent_span_id", _ID, nullable=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("origin", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("operation", sa.String(), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("outcome", sa.String(), nullable=False),
        sa.Column("recovered", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("error_class", sa.String(), nullable=True),
        sa.Column("opens_turn", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_snapshot", _COST_TYPE, nullable=True),
        sa.Column("tool_name", sa.String(), nullable=True),
        sa.Column("tool_type", sa.String(), nullable=True),
        sa.Column("tool_call_id", sa.String(), nullable=True),
        sa.Column("request_id", _ID, nullable=True),
        sa.Column("otel_trace_id", _ID, nullable=True),
        sa.Column("otel_span_id", _ID, nullable=True),
        sa.Column("attributes", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "trace_id", "span_id"),
        sa.ForeignKeyConstraint(
            ["workspace_id", "trace_id"],
            ["traces.workspace_id", "traces.trace_id"],
            ondelete="CASCADE",
            name="fk_trace_spans_trace",
        ),
    )
    op.create_index("ix_trace_spans_trace_start", "trace_spans", ["workspace_id", "trace_id", "start_time"])
    op.create_index("ix_trace_spans_trace_tool_call", "trace_spans", ["workspace_id", "trace_id", "tool_call_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_trace_spans_trace_tool_call", table_name="trace_spans")
    op.drop_index("ix_trace_spans_trace_start", table_name="trace_spans")
    op.drop_table("trace_spans")
    op.drop_index("ix_traces_last_activity", table_name="traces")
    op.drop_index("ix_traces_workspace_last_activity", table_name="traces")
    op.drop_index("ix_traces_api_key_id", table_name="traces")
    op.drop_index("ix_traces_user_id", table_name="traces")
    op.drop_table("traces")
