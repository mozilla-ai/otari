"""Add the per-workspace hosted provider access and provider policy tables.

One row per (workspace, hosted provider) a workspace departs from, or none. A
row switches the provider off for the workspace or narrows it to a list of
models; it never grants, so the organization's own access to the provider is
still decided where it was.

``provider`` is the hosted provider's name rather than a foreign key, because a
hosted provider declared in configuration has no row to point at, the same
reason ``hosted_provider_models.provider`` is a name. ``workspace_id`` cascades:
the row means nothing without its workspace.

``workspace_provider_policies`` holds one row per workspace that requires
something of every provider serving it, zero data retention today, keyed by
the workspace and cascading with it.

No backfill. Absence of a row is the "inherit everything" state, so a deployment
upgrading onto this revision keeps behaving exactly as it did.

Revision ID: d8a3c5e7f9b1
Revises: b7e3c9d2f4a1
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d8a3c5e7f9b1"
down_revision: str | Sequence[str] | None = "b7e3c9d2f4a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "workspace_hosted_provider_access"
_POLICY_TABLE = "workspace_provider_policies"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("models", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("workspace_id", "provider", name="uq_workspace_hosted_provider_access_workspace_provider"),
    )
    op.create_index("ix_workspace_hosted_provider_access_workspace_id", _TABLE, ["workspace_id"], unique=False)
    op.create_table(
        _POLICY_TABLE,
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("require_zero_data_retention", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("workspace_id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    op.drop_table(_POLICY_TABLE)
    op.drop_index("ix_workspace_hosted_provider_access_workspace_id", table_name=_TABLE)
    op.drop_table(_TABLE)
