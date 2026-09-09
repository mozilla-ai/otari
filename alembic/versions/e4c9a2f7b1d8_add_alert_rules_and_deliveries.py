"""Add organization-scoped alert rules and the delivery dedupe ledger.

One ``alert_rules`` row is one destination an organization wants its budget
alerts sent to, plus the thresholds that reach it. The destination is an Apprise
URL, so Slack, Discord, PagerDuty, mail and a plain webhook are one column
rather than one column per vendor, and it is stored encrypted because such a URL
embeds a bot token or basic-auth credentials.

``alert_deliveries`` is what makes the feature send each alert once. A crossed
threshold stays crossed for the rest of the budget period and every refresher in
``gateway.main`` runs once per worker, so both a repeat over time and a repeat
across workers are settled by one unique constraint that a claim inserts against
before dispatching. ``period_start`` is part of the key, which is what re-arms an
alert when a budget period rolls without anything having to clear the table.

The partial unique index alongside the constraint covers a budget with no period
at all: PostgreSQL treats two NULL ``period_start`` values as distinct, so the
constraint alone would let a never-rolling ceiling alert on every tick.

``alert_rules.organization_id`` cascades, matching the other tenant-owned
configuration tables (``organization_guardrails``, ``workspace_mcp_servers``):
these rows are configuration, not durable request-plane history.
``scoped_budget_id`` is deliberately not a foreign key, matching
``scoped_budgets``' own columns, which declare none.

A deployment that creates no alert rules is unaffected: the evaluator returns on
its first query when the table is empty.

Revision ID: e4c9a2f7b1d8
Revises: d5b7f9a1c3e6
Create Date: 2026-09-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e4c9a2f7b1d8"
down_revision: str | Sequence[str] | None = "d5b7f9a1c3e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RULES_UNIQUE_NAME = "uq_alert_rules_org_name"
_RULES_WARN_CHECK = "ck_alert_rules_warn_percent_range"
_DELIVERIES_UNIQUE_NAME = "uq_alert_deliveries_rule_budget_period_kind"
_DELIVERIES_NO_PERIOD_INDEX = "uq_alert_deliveries_rule_budget_kind_no_period"
_DELIVERIES_LOOKUP_INDEX = "ix_alert_deliveries_rule_period"


def upgrade() -> None:
    op.create_table(
        "alert_rules",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("encrypted_destination", sa.Text(), nullable=False),
        sa.Column("redacted_destination", sa.String(), nullable=False),
        sa.Column("warn_at_percent", sa.Integer(), nullable=True),
        sa.Column("notify_on_exceeded", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", "name", name=_RULES_UNIQUE_NAME),
        sa.CheckConstraint(
            "warn_at_percent IS NULL OR (warn_at_percent > 0 AND warn_at_percent < 100)",
            name=_RULES_WARN_CHECK,
        ),
    )
    op.create_index(
        op.f("ix_alert_rules_organization_id"),
        "alert_rules",
        ["organization_id"],
    )

    op.create_table(
        "alert_deliveries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("alert_rule_id", sa.Uuid(), nullable=False),
        sa.Column("scoped_budget_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["alert_rule_id"], ["alert_rules.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "alert_rule_id",
            "scoped_budget_id",
            "period_start",
            "kind",
            name=_DELIVERIES_UNIQUE_NAME,
        ),
    )
    op.create_index(
        op.f("ix_alert_deliveries_alert_rule_id"),
        "alert_deliveries",
        ["alert_rule_id"],
    )
    op.create_index(
        _DELIVERIES_NO_PERIOD_INDEX,
        "alert_deliveries",
        ["alert_rule_id", "scoped_budget_id", "kind"],
        unique=True,
        sqlite_where=sa.text("period_start IS NULL"),
        postgresql_where=sa.text("period_start IS NULL"),
    )
    op.create_index(
        _DELIVERIES_LOOKUP_INDEX,
        "alert_deliveries",
        ["alert_rule_id", "period_start"],
    )


def downgrade() -> None:
    op.drop_index(_DELIVERIES_LOOKUP_INDEX, table_name="alert_deliveries")
    op.drop_index(_DELIVERIES_NO_PERIOD_INDEX, table_name="alert_deliveries")
    op.drop_index(op.f("ix_alert_deliveries_alert_rule_id"), table_name="alert_deliveries")
    op.drop_table("alert_deliveries")
    op.drop_index(op.f("ix_alert_rules_organization_id"), table_name="alert_rules")
    op.drop_table("alert_rules")
