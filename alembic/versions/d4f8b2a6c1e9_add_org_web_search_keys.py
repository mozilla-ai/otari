"""Add organization web search keys and their workspace overrides.

An organization can bring its own key for a web search provider, and its
workspaces search with it rather than with the deployment's own search. The
shape follows the provider key tables (otari-ai#1748, mozilla-ai/otari#1724):
one live default per organization and provider, archival rather than a hard
delete of a key in use, and per-workspace overrides that pin a key or turn it
off.

The override table's foreign key to its key is composite on
``(organization_id, id)``, so an override can only name a key of its own
organization. Both tables cascade: a credential and its overrides mean nothing
once their organization or workspace is gone.

Revision ID: d4f8b2a6c1e9
Revises: e8b2d5f1a7c3
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4f8b2a6c1e9"
down_revision: str | Sequence[str] | None = "e8b2d5f1a7c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ORG_DEFAULT_INDEX = "uq_org_web_search_keys_org_default"
_LIVE_DEFAULT = "is_org_default AND archived_at IS NULL"


def upgrade() -> None:
    op.create_table(
        "org_web_search_keys",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=False),
        sa.Column("last4", sa.String(length=8), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_org_default", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", "provider", "name", name="uq_org_web_search_keys_org_provider_name"),
        sa.UniqueConstraint("organization_id", "id", name="uq_org_web_search_keys_org_id"),
    )
    op.create_index(op.f("ix_org_web_search_keys_organization_id"), "org_web_search_keys", ["organization_id"])
    op.create_index(
        _ORG_DEFAULT_INDEX,
        "org_web_search_keys",
        ["organization_id", "provider"],
        unique=True,
        postgresql_where=sa.text(_LIVE_DEFAULT),
        sqlite_where=sa.text(_LIVE_DEFAULT),
    )

    op.create_table(
        "workspace_web_search_key_overrides",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("org_web_search_key_id", sa.Uuid(), nullable=False),
        sa.Column("is_default", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("disabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["organization_id", "org_web_search_key_id"],
            ["org_web_search_keys.organization_id", "org_web_search_keys.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "workspace_id", "org_web_search_key_id", name="uq_workspace_web_search_key_overrides_ws_key"
        ),
    )
    op.create_index(
        op.f("ix_workspace_web_search_key_overrides_workspace_id"),
        "workspace_web_search_key_overrides",
        ["workspace_id"],
    )
    op.create_index(
        op.f("ix_workspace_web_search_key_overrides_org_web_search_key_id"),
        "workspace_web_search_key_overrides",
        ["org_web_search_key_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_workspace_web_search_key_overrides_org_web_search_key_id"),
        table_name="workspace_web_search_key_overrides",
    )
    op.drop_index(
        op.f("ix_workspace_web_search_key_overrides_workspace_id"), table_name="workspace_web_search_key_overrides"
    )
    op.drop_table("workspace_web_search_key_overrides")
    op.drop_index(_ORG_DEFAULT_INDEX, table_name="org_web_search_keys")
    op.drop_index(op.f("ix_org_web_search_keys_organization_id"), table_name="org_web_search_keys")
    op.drop_table("org_web_search_keys")
