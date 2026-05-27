"""add budget alerts

Revision ID: f3a4b5c6d7e8
Revises: f2a3b4c5d6e7
Create Date: 2026-05-27 16:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3a4b5c6d7e8"
down_revision: str | Sequence[str] | None = "f2a3b4c5d6e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("budgets") as batch_op:
        batch_op.add_column(sa.Column("alert_thresholds", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("alert_webhook_url", sa.String(), nullable=True))

    op.create_table(
        "budget_alerts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("budget_id", sa.String(), nullable=False),
        sa.Column("scope_type", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("spend", sa.Float(), nullable=False),
        sa.Column("max_budget", sa.Float(), nullable=False),
        sa.Column("budget_period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("webhook_url", sa.String(), nullable=True),
        sa.Column("delivery_status", sa.String(), nullable=False, server_default="not_configured"),
        sa.Column("delivery_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_delivery_status_code", sa.Integer(), nullable=True),
        sa.Column("last_delivery_error", sa.String(), nullable=True),
        sa.Column("last_delivery_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_delivery_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dead_lettered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["budget_id"], ["budgets.budget_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_budget_alerts_budget_id"), "budget_alerts", ["budget_id"], unique=False)
    op.create_index(
        op.f("ix_budget_alerts_delivery_status"),
        "budget_alerts",
        ["delivery_status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_budget_alerts_next_delivery_attempt_at"),
        "budget_alerts",
        ["next_delivery_attempt_at"],
        unique=False,
    )
    op.create_index(
        "ix_budget_alerts_budget_id_created_at",
        "budget_alerts",
        ["budget_id", "created_at"],
        unique=False,
    )
    op.create_index(op.f("ix_budget_alerts_scope_id"), "budget_alerts", ["scope_id"], unique=False)
    op.create_index(op.f("ix_budget_alerts_scope_type"), "budget_alerts", ["scope_type"], unique=False)
    op.create_index(
        "ix_budget_alerts_scope_created_at",
        "budget_alerts",
        ["scope_type", "scope_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_budget_alerts_scope_created_at", table_name="budget_alerts")
    op.drop_index(op.f("ix_budget_alerts_scope_type"), table_name="budget_alerts")
    op.drop_index(op.f("ix_budget_alerts_scope_id"), table_name="budget_alerts")
    op.drop_index("ix_budget_alerts_budget_id_created_at", table_name="budget_alerts")
    op.drop_index(op.f("ix_budget_alerts_next_delivery_attempt_at"), table_name="budget_alerts")
    op.drop_index(op.f("ix_budget_alerts_delivery_status"), table_name="budget_alerts")
    op.drop_index(op.f("ix_budget_alerts_budget_id"), table_name="budget_alerts")
    op.drop_table("budget_alerts")

    with op.batch_alter_table("budgets") as batch_op:
        batch_op.drop_column("alert_webhook_url")
        batch_op.drop_column("alert_thresholds")
