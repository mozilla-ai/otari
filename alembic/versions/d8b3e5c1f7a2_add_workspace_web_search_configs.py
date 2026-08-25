"""Add the per-workspace web-search configuration table.

One row per workspace, or none. It carries no credential and no endpoint: the
search backend stays deployment-wide, and this says which workspaces may reach
it and how their searches are constrained (see
``services/tenancy/workspace_web_search_service.py``).

``workspace_id`` is the primary key rather than a surrogate id, since a
workspace has exactly one configuration, and a real foreign key with ``CASCADE``
like ``workspace_code_execution_policies``: nothing else names the row, so it
rides the workspace's own delete.

No backfill. Absence of a row is the "no narrowing" state, so a deployment
upgrading onto this revision keeps behaving exactly as it did.

Revision ID: d8b3e5c1f7a2
Revises: c4d7f2a9b6e1
Create Date: 2026-08-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d8b3e5c1f7a2"
down_revision: str | Sequence[str] | None = "c4d7f2a9b6e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "workspace_web_search_configs"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        # The three NOT NULL columns carry server defaults so a row can be
        # inserted by something that is not this service's ORM mapping (a psql
        # session, a data migration) without naming them. `sa.func.now()` rather
        # than a dialect literal because the chain runs on SQLite too, which is
        # how `workspace` itself declares its own `created_at`.
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("max_results", sa.Integer(), nullable=True),
        sa.Column("purpose_hint", sa.Text(), nullable=True),
        sa.Column("allowed_domains", sa.JSON(), nullable=True),
        sa.Column("blocked_domains", sa.JSON(), nullable=True),
        sa.Column("provider_options", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("workspace_id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        # A ceiling of zero or less would ask for a search that can return
        # nothing while reading as configured; the request schema refuses it first.
        sa.CheckConstraint(
            "max_results IS NULL OR max_results > 0",
            name="ck_workspace_web_search_configs_max_results_positive",
        ),
    )


def downgrade() -> None:
    op.drop_table(_TABLE)
