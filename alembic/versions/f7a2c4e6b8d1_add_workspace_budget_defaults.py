"""Add the workspace-member-budget-default template table.

A workspace-level template for a per-member ``scoped_budgets`` ceiling. It
carries no counters and enforces nothing itself; creating one, or a member
joining a workspace that has one, materializes a ``scoped_budgets`` row per
member (see ``services/tenancy/workspace_budget_default_service.py``).

Two partial unique indexes rather than one plain UNIQUE over the pair, for the
same reason ``scoped_budgets`` has them: PostgreSQL and SQLite both treat NULLs
as distinct, so an index including the nullable ``provider_key_id`` would
enforce nothing on the aggregate (NULL-key) rows.

Unlike ``scoped_budgets.scope_id``, ``workspace_id`` here is a real foreign
key: a template belongs to exactly one workspace and nothing else names it, so
it rides the workspace's own delete rather than needing separate cleanup.

Revision ID: f7a2c4e6b8d1
Revises: c8f2a6b4e9d3
Create Date: 2026-08-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f7a2c4e6b8d1"
down_revision: str | Sequence[str] | None = "c8f2a6b4e9d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WITH_KEY = "uq_workspace_budget_defaults_with_key"
_NO_KEY = "uq_workspace_budget_defaults_no_key"


def upgrade() -> None:
    op.create_table(
        "workspace_budget_defaults",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("provider_key_id", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("max_budget", sa.Float(), nullable=True),
        sa.Column("budget_duration_sec", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
    )
    op.create_index(
        op.f("ix_workspace_budget_defaults_workspace_id"),
        "workspace_budget_defaults",
        ["workspace_id"],
    )
    op.create_index(
        _WITH_KEY,
        "workspace_budget_defaults",
        ["workspace_id", "provider_key_id"],
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NOT NULL"),
        sqlite_where=sa.text("provider_key_id IS NOT NULL"),
    )
    op.create_index(
        _NO_KEY,
        "workspace_budget_defaults",
        ["workspace_id"],
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NULL"),
        sqlite_where=sa.text("provider_key_id IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(_NO_KEY, table_name="workspace_budget_defaults")
    op.drop_index(_WITH_KEY, table_name="workspace_budget_defaults")
    op.drop_index(op.f("ix_workspace_budget_defaults_workspace_id"), table_name="workspace_budget_defaults")
    op.drop_table("workspace_budget_defaults")
