"""Keep what a model server says about the models it serves.

vLLM and SGLang list each model with the context length they serve it at,
which can be shorter than models.dev's, and SGLang says whether it parses tool
calls and reasoning. One row per provider and model, replaced on each listing.
A row reached through an organization's own key names the key, since two
organizations can list different servers under the same prefix.

Revision ID: d5a1c7e3f9b2
Revises: c4f9e2a7d5b8
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d5a1c7e3f9b2"
down_revision: str | Sequence[str] | None = "c4f9e2a7d5b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "served_models"


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        _TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=255), nullable=False),
        sa.Column("model", sa.String(length=512), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=True),
        sa.Column("org_provider_key_id", sa.Uuid(), nullable=True),
        sa.Column("engine", sa.String(length=16), nullable=False),
        sa.Column("context_window", sa.Integer(), nullable=True),
        sa.Column("root", sa.String(length=512), nullable=True),
        sa.Column("settings_reported", sa.Boolean(), nullable=False),
        sa.Column("reasoning_parser", sa.String(length=64), nullable=True),
        sa.Column("tool_call_parser", sa.String(length=64), nullable=True),
        sa.Column("input_modalities", sa.JSON(), nullable=True),
        sa.Column("reported_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("engine IN ('vllm', 'sglang')", name="ck_served_models_engine"),
        sa.CheckConstraint(
            "(organization_id IS NULL) = (org_provider_key_id IS NULL)", name="ck_served_models_key_owner"
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["org_provider_key_id"], ["org_provider_keys.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_served_models_organization_id", _TABLE, ["organization_id"], unique=False)
    op.create_index("ix_served_models_org_provider_key_id", _TABLE, ["org_provider_key_id"], unique=False)
    op.create_index(
        "uq_served_models_deployment",
        _TABLE,
        ["provider", "model"],
        unique=True,
        postgresql_where=sa.text("org_provider_key_id IS NULL"),
        sqlite_where=sa.text("org_provider_key_id IS NULL"),
    )
    op.create_index(
        "uq_served_models_key",
        _TABLE,
        ["org_provider_key_id", "model"],
        unique=True,
        postgresql_where=sa.text("org_provider_key_id IS NOT NULL"),
        sqlite_where=sa.text("org_provider_key_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("uq_served_models_key", table_name=_TABLE)
    op.drop_index("uq_served_models_deployment", table_name=_TABLE)
    op.drop_index("ix_served_models_org_provider_key_id", table_name=_TABLE)
    op.drop_index("ix_served_models_organization_id", table_name=_TABLE)
    op.drop_table(_TABLE)
