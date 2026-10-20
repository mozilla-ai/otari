"""Add the health check settings, runs, results and recoveries tables.

The settings table holds one row, pinned by a check constraint, which every
replica's worker polls; the conditional update that moves its ``next_run_at``
forward is what lets exactly one replica start a run.

Revision ID: c4e6a8b0d2f4
Revises: cc92a10774d8
Create Date: 2026-10-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e6a8b0d2f4"
down_revision: str | Sequence[str] | None = "cc92a10774d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "health_check_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("schedule_enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("interval_minutes", sa.Integer(), server_default="60", nullable=False),
        sa.Column("prompt", sa.String(), nullable=False),
        sa.Column("max_tokens", sa.Integer(), server_default="16", nullable=False),
        sa.Column("reasoning_max_tokens", sa.Integer(), server_default="256", nullable=False),
        sa.Column("reasoning_every_runs", sa.Integer(), server_default="6", nullable=False),
        sa.Column("runs_started", sa.Integer(), server_default="0", nullable=False),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("run_requested", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("alert_emails", sa.JSON(), nullable=False),
        sa.Column("alerted_selectors", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_health_check_settings_one_row"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "health_check_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("included_reasoning", sa.Boolean(), nullable=False),
        sa.Column("deferred_selectors", sa.JSON(), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("passed", sa.Integer(), nullable=False),
        sa.Column("failed", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_health_check_runs_started_at"), "health_check_runs", ["started_at"])
    op.create_table(
        "health_check_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("selector", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('pass', 'fail', 'skip')", name="ck_health_check_results_status"),
        sa.ForeignKeyConstraint(["run_id"], ["health_check_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_health_check_results_run_id"), "health_check_results", ["run_id"])
    op.create_table(
        "health_check_recoveries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("selector", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("failing_since", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("failed_runs", sa.Integer(), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_health_check_recoveries_selector"), "health_check_recoveries", ["selector"])
    op.create_index(op.f("ix_health_check_recoveries_recovered_at"), "health_check_recoveries", ["recovered_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_health_check_recoveries_recovered_at"), table_name="health_check_recoveries")
    op.drop_index(op.f("ix_health_check_recoveries_selector"), table_name="health_check_recoveries")
    op.drop_table("health_check_recoveries")
    op.drop_index(op.f("ix_health_check_results_run_id"), table_name="health_check_results")
    op.drop_table("health_check_results")
    op.drop_index(op.f("ix_health_check_runs_started_at"), table_name="health_check_runs")
    op.drop_table("health_check_runs")
    op.drop_table("health_check_settings")
