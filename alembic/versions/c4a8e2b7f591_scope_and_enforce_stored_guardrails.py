"""Say how a stored guardrail is enforced, and which workspaces it checks.

Three columns and one table, which together turn a stored definition from
something an operator can build into something that runs.

``mode`` and ``on_unavailable`` are the pair ``GuardrailConfig`` already carries,
so a definition answers the same two questions a caller's entry does: what to do
when the guardrail flags the input, and what to do when it could not answer at
all.

``applies_to_all_workspaces`` and ``guardrail_credential_workspaces`` scope it,
copying ``organization_guardrails`` and its scope table. The columns default to
``block`` and to unscoped, which is inert: a definition reaches no workspace
until one is named or the flag is set.

Revision ID: c4a8e2b7f591
Revises: d3f5a7c9e1b4
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4a8e2b7f591"
down_revision: str | Sequence[str] | None = "d3f5a7c9e1b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "guardrail_credentials",
        sa.Column("mode", sa.String(), nullable=False, server_default="block"),
    )
    op.add_column(
        "guardrail_credentials",
        sa.Column("on_unavailable", sa.String(), nullable=False, server_default="block"),
    )
    op.add_column(
        "guardrail_credentials",
        sa.Column("applies_to_all_workspaces", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_table(
        "guardrail_credential_workspaces",
        sa.Column("credential_name", sa.String(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["credential_name"], ["guardrail_credentials.name"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("credential_name", "workspace_id"),
    )
    op.create_index(
        op.f("ix_guardrail_credential_workspaces_workspace_id"),
        "guardrail_credential_workspaces",
        ["workspace_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_guardrail_credential_workspaces_workspace_id"),
        table_name="guardrail_credential_workspaces",
    )
    op.drop_table("guardrail_credential_workspaces")
    op.drop_column("guardrail_credentials", "applies_to_all_workspaces")
    op.drop_column("guardrail_credentials", "on_unavailable")
    op.drop_column("guardrail_credentials", "mode")
