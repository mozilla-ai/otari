"""Add self-registered OAuth clients, and let an approval redeem to an access token.

An MCP client registers itself (RFC 7591) and is approved like any OAuth app,
but its approval redeems to the person's read-only access token rather than an
API key, so ``oauth_app_grants`` records which it redeems to and, for an access
token, the organization it reads in place of a workspace. Existing grants are API
key grants.

Revision ID: 49e3ccf30619
Revises: 4ef53b7e9aac
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "49e3ccf30619"
down_revision: str | Sequence[str] | None = "4ef53b7e9aac"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "oauth_dynamic_client",
        sa.Column("client_id", sa.String(length=128), nullable=False),
        sa.Column("client_name", sa.String(length=100), nullable=False),
        sa.Column("redirect_uris", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("authorized_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("client_id"),
    )
    op.create_index(op.f("ix_oauth_dynamic_client_authorized_at"), "oauth_dynamic_client", ["authorized_at"])
    with op.batch_alter_table("oauth_app_grants") as batch:
        batch.add_column(sa.Column("credential", sa.String(length=16), nullable=False, server_default="api_key"))
        batch.add_column(sa.Column("organization_id", sa.Uuid(), nullable=True))
        batch.create_foreign_key(
            "fk_oauth_app_grants_organization_id", "organization", ["organization_id"], ["id"], ondelete="CASCADE"
        )
        batch.alter_column("workspace_id", existing_type=sa.Uuid(), nullable=True)


def downgrade() -> None:
    # A pending access token grant names no workspace; it lives two minutes, so dropping it loses nothing.
    op.execute("DELETE FROM oauth_app_grants WHERE workspace_id IS NULL")
    with op.batch_alter_table("oauth_app_grants") as batch:
        batch.alter_column("workspace_id", existing_type=sa.Uuid(), nullable=False)
        batch.drop_constraint("fk_oauth_app_grants_organization_id", type_="foreignkey")
        batch.drop_column("organization_id")
        batch.drop_column("credential")
    op.drop_index(op.f("ix_oauth_dynamic_client_authorized_at"), table_name="oauth_dynamic_client")
    op.drop_table("oauth_dynamic_client")
