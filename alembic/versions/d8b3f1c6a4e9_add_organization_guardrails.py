"""Add the organization guardrail plane and its workspace scope.

One ``organization_guardrails`` row is one guardrail an organization mandates:
a profile on the guardrails service, an optional endpoint and credential of its
own, the two failure-handling modes, and the switch that decides whether it runs
in every workspace of the organization or only in the ones
``organization_guardrail_workspaces`` names.

``(organization_id, profile)`` is unique because the effective guardrail set on
the request path is keyed by profile: two rows of one profile could never both
run, so the second is refused at the write rather than silently ignored at
admission.

Both foreign keys cascade, matching the other tenant-owned configuration tables
(``workspace_mcp_servers``, ``workspace_code_execution_policies``): these rows
are configuration, not durable request-plane history.

The deployment-wide ``guardrails_url`` is untouched and stays in
``runtime_settings``; a deployment with no rows in either table behaves exactly
as it did before this migration.

Revision ID: d8b3f1c6a4e9
Revises: e9b3d7f1a5c2
Create Date: 2026-08-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d8b3f1c6a4e9"
down_revision: str | Sequence[str] | None = "e9b3d7f1a5c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UNIQUE_NAME = "uq_organization_guardrails_org_profile"


def upgrade() -> None:
    op.create_table(
        "organization_guardrails",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("profile", sa.String(), nullable=False),
        sa.Column("url", sa.String(), nullable=True),
        sa.Column("encrypted_credential", sa.Text(), nullable=True),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("on_unavailable", sa.String(), nullable=False),
        sa.Column("validate_kwargs", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("applies_to_all_workspaces", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", "profile", name=_UNIQUE_NAME),
    )
    op.create_index(
        op.f("ix_organization_guardrails_organization_id"),
        "organization_guardrails",
        ["organization_id"],
    )

    op.create_table(
        "organization_guardrail_workspaces",
        sa.Column("organization_guardrail_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("organization_guardrail_id", "workspace_id"),
        sa.ForeignKeyConstraint(["organization_guardrail_id"], ["organization_guardrails.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
    )
    # The request path looks a workspace up in this table on every request that
    # reaches a completion endpoint, so the workspace side is indexed as well as
    # the guardrail side the composite primary key already covers.
    op.create_index(
        op.f("ix_organization_guardrail_workspaces_workspace_id"),
        "organization_guardrail_workspaces",
        ["workspace_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_organization_guardrail_workspaces_workspace_id"),
        table_name="organization_guardrail_workspaces",
    )
    op.drop_table("organization_guardrail_workspaces")
    op.drop_index(
        op.f("ix_organization_guardrails_organization_id"),
        table_name="organization_guardrails",
    )
    op.drop_table("organization_guardrails")
