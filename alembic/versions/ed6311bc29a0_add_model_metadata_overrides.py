"""Let an operator or an organization set what a model's server and models.dev do not say.

A self-hosted server can list a model with no output limit, no tool or
reasoning flags and no modalities, and models.dev may not know it at all. One
row per ``provider:model``, the deployment's or one organization's, whose
non-null fields win over every other source.

Revision ID: ed6311bc29a0
Revises: a4c8e2f6b1d9
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "ed6311bc29a0"
down_revision: str | Sequence[str] | None = "a4c8e2f6b1d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "model_metadata_overrides"


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        _TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=255), nullable=False),
        sa.Column("model", sa.String(length=512), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=True),
        sa.Column("context_window", sa.Integer(), nullable=True),
        sa.Column("max_output_tokens", sa.Integer(), nullable=True),
        sa.Column("tool_call", sa.Boolean(), nullable=True),
        sa.Column("reasoning", sa.Boolean(), nullable=True),
        sa.Column("structured_output", sa.Boolean(), nullable=True),
        sa.Column("input_modalities", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(f"ix_{_TABLE}_organization_id", _TABLE, ["organization_id"], unique=False)
    op.create_index(
        "uq_model_metadata_overrides_deployment",
        _TABLE,
        ["provider", "model"],
        unique=True,
        postgresql_where=sa.text("organization_id IS NULL"),
        sqlite_where=sa.text("organization_id IS NULL"),
    )
    op.create_index(
        "uq_model_metadata_overrides_organization",
        _TABLE,
        ["organization_id", "provider", "model"],
        unique=True,
        postgresql_where=sa.text("organization_id IS NOT NULL"),
        sqlite_where=sa.text("organization_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("uq_model_metadata_overrides_organization", table_name=_TABLE)
    op.drop_index("uq_model_metadata_overrides_deployment", table_name=_TABLE)
    op.drop_index(f"ix_{_TABLE}_organization_id", table_name=_TABLE)
    op.drop_table(_TABLE)
