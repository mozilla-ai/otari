"""Add the MCP server catalog, the organizations' featured shelves and the sync state.

The sync state table holds one row, pinned by a check constraint, which every
replica's worker polls; the conditional update that sets ``running`` is what
lets exactly one replica sync.

Revision ID: d7a1c3e5f9b2
Revises: b3e8d1f6a2c9
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d7a1c3e5f9b2"
down_revision: str | Sequence[str] | None = "b3e8d1f6a2c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mcp_catalog_servers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("requires_token", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("website_url", sa.String(), nullable=True),
        sa.Column("version", sa.String(), nullable=True),
        sa.Column("source", sa.String(), server_default="registry", nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index(op.f("ix_mcp_catalog_servers_title"), "mcp_catalog_servers", ["title"])
    op.create_index(op.f("ix_mcp_catalog_servers_last_seen_at"), "mcp_catalog_servers", ["last_seen_at"])
    op.create_table(
        "mcp_catalog_featured",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("requires_token", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("website_url", sa.String(), nullable=True),
        sa.Column("source", sa.String(), server_default="registry", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "name", name="uq_mcp_catalog_featured_organization_name"),
    )
    op.create_index(op.f("ix_mcp_catalog_featured_organization_id"), "mcp_catalog_featured", ["organization_id"])
    op.create_table(
        "mcp_catalog_sync_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sync_requested", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("running", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("crawl_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cursor", sa.String(), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("outcome", sa.String(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("pages", sa.Integer(), server_default="0", nullable=False),
        sa.Column("fetched", sa.Integer(), server_default="0", nullable=False),
        sa.Column("kept", sa.Integer(), server_default="0", nullable=False),
        sa.Column("removed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_mcp_catalog_sync_state_one_row"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("mcp_catalog_sync_state")
    op.drop_index(op.f("ix_mcp_catalog_featured_organization_id"), table_name="mcp_catalog_featured")
    op.drop_table("mcp_catalog_featured")
    op.drop_index(op.f("ix_mcp_catalog_servers_last_seen_at"), table_name="mcp_catalog_servers")
    op.drop_index(op.f("ix_mcp_catalog_servers_title"), table_name="mcp_catalog_servers")
    op.drop_table("mcp_catalog_servers")
