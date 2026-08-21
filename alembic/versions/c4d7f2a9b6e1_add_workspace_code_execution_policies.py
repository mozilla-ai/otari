"""Add the per-workspace code-execution policy table.

One row per workspace, or none. It carries no credential and no endpoint: the
sandbox stays deployment-wide, and this says which workspaces may reach it and
within which limits (see ``services/tenancy/workspace_code_execution_policy_service.py``).

``workspace_id`` is the primary key rather than a surrogate id, since a
workspace has exactly one policy, and a real foreign key with ``CASCADE`` like
``workspace_budget_defaults``: nothing else names the row, so it rides the
workspace's own delete.

No backfill. Absence of a row is the "no narrowing" state, so a deployment
upgrading onto this revision keeps behaving exactly as it did.

Revision ID: c4d7f2a9b6e1
Revises: c5e9a1d3f7b2
Create Date: 2026-08-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4d7f2a9b6e1"
down_revision: str | Sequence[str] | None = "c5e9a1d3f7b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "workspace_code_execution_policies"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("default_purpose_hint", sa.Text(), nullable=True),
        sa.Column("max_iterations", sa.Integer(), nullable=True),
        sa.Column("exec_timeout_s", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        # A ceiling of zero or less would floor the loop to nothing runnable
        # while reading as configured; the request schemas refuse it first.
        sa.CheckConstraint(
            "max_iterations IS NULL OR max_iterations > 0",
            name="ck_workspace_code_execution_policies_max_iterations_positive",
        ),
        sa.CheckConstraint(
            "exec_timeout_s IS NULL OR exec_timeout_s > 0",
            name="ck_workspace_code_execution_policies_exec_timeout_positive",
        ),
    )


def downgrade() -> None:
    op.drop_table(_TABLE)
