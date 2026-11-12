"""Single-use OAuth app authorization grants.

Revision ID: d8a491bc02fa
Revises: c4b8e2a19d73
"""

import sqlalchemy as sa
from alembic import op

revision = "d8a491bc02fa"
down_revision = "c4b8e2a19d73"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "oauth_app_grants",
        sa.Column("code_hash", sa.String(64), primary_key=True),
        sa.Column("client_id", sa.String(128), nullable=False),
        sa.Column("redirect_uri", sa.String(2048), nullable=False),
        sa.Column("challenge", sa.String(43), nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_oauth_app_grants_user_id", "oauth_app_grants", ["user_id"])
    op.create_index("ix_oauth_app_grants_workspace_id", "oauth_app_grants", ["workspace_id"])
    op.create_index("ix_oauth_app_grants_expires_at", "oauth_app_grants", ["expires_at"])


def downgrade() -> None:
    op.drop_table("oauth_app_grants")
