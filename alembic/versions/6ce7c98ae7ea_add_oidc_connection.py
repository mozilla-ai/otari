"""Add the organization OIDC connection table.

One organization holds at most one connection, by the unique constraint on
``organization_id``. ``resolved_issuer`` carries an index but no unique
constraint: two organizations may configure one shared identity provider, so a
unique index there would refuse a legitimate second connection.

Revision ID: 6ce7c98ae7ea
Revises: 086a7895681c
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "6ce7c98ae7ea"
down_revision: str | Sequence[str] | None = "086a7895681c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "oidc_connection",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("issuer", sa.String(length=255), nullable=False),
        sa.Column("resolved_issuer", sa.String(length=255), nullable=False),
        sa.Column("client_id", sa.String(length=255), nullable=False),
        sa.Column("encrypted_client_secret", sa.Text(), nullable=False),
        sa.Column("client_secret_last4", sa.String(length=4), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=False),
        sa.Column("default_role", sa.String(length=32), server_default="member", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("discovery_document", sa.JSON(), nullable=False),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("first_sign_in_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", name="uq_oidc_connection_organization_id"),
    )
    op.create_index("ix_oidc_connection_resolved_issuer", "oidc_connection", ["resolved_issuer"])


def downgrade() -> None:
    op.drop_index("ix_oidc_connection_resolved_issuer", table_name="oidc_connection")
    op.drop_table("oidc_connection")
