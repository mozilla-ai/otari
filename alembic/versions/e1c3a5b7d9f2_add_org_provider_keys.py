"""Add organization-scoped provider keys.

Decided at otari-ai#1748 (mozilla-ai/otari#643): the platform's
``ProviderKey`` shape ports into otari as new, additive tables that become the
source of truth for organization-scoped BYO provider credentials.
``provider_credentials`` and its config.yml-merge overlay are untouched by
this revision; they stay the mechanism for config.yml-defined and legacy
deployment-global stored credentials.

Three tables, no legacy data to migrate (they are brand new here), so this is
a single forward migration with no expand/backfill/contract dance:

- ``org_provider_keys``: one BYO credential per organization+provider+name,
  with a partial unique index enforcing at most one ``is_org_default`` row per
  ``(organization_id, provider)`` among non-archived rows. The "managed
  bucket" dimension of the platform's equivalent index does not carry over:
  every otari-side key is BYO, so there is one default per provider, not one
  per bucket.
- ``workspace_provider_key_overrides``: a workspace's departure from its
  organization's default for one key (pin it as the workspace default, or
  disable it for that workspace).
- ``workspace_provider_model_restrictions``: a per-workspace, per-key model
  allow-list.

Rebased onto `f7a2c4e6b8d1` (workspace per-member budget defaults), the actual
head `main` had gained by the time this revision landed, rather than the
original `a3c7e1b9d5f2` this was first written against: `main` grew a new head
several times over the life of this PR, and each rebase re-parents here rather
than leaving a redundant merge revision beside `main`'s own.

Revision ID: e1c3a5b7d9f2
Revises: f7a2c4e6b8d1
Create Date: 2026-08-19 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1c3a5b7d9f2"
down_revision: str | Sequence[str] | None = "f7a2c4e6b8d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ORG_DEFAULT_INDEX = "uq_org_provider_keys_org_default"


def upgrade() -> None:
    op.create_table(
        "org_provider_keys",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("api_base", sa.String(length=1024), nullable=True),
        sa.Column("client_args", sa.JSON(), nullable=True),
        sa.Column("encrypted_api_key", sa.String(), nullable=True),
        sa.Column("last4", sa.String(length=8), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_org_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "provider", "name", name="uq_org_provider_keys_org_provider_name"),
        # Covers (organization_id, id) so the two link tables below can carry a
        # composite FK to it: Postgres requires the referenced columns of a
        # composite foreign key to be backed by a unique constraint, not only
        # ``id`` being the primary key on its own.
        sa.UniqueConstraint("organization_id", "id", name="uq_org_provider_keys_org_id"),
    )
    op.create_index(op.f("ix_org_provider_keys_organization_id"), "org_provider_keys", ["organization_id"])
    op.create_index(
        _ORG_DEFAULT_INDEX,
        "org_provider_keys",
        ["organization_id", "provider"],
        unique=True,
        postgresql_where=sa.text("is_org_default AND archived_at IS NULL"),
        sqlite_where=sa.text("is_org_default AND archived_at IS NULL"),
    )

    op.create_table(
        "workspace_provider_key_overrides",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        # Denormalized from the workspace's own organization (the service layer
        # only ever creates an override when the key and the workspace already
        # agree on organization), so the composite FK below can enforce that
        # invariant at the database rather than trusting every write path to
        # keep re-deriving it correctly.
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("org_provider_key_id", sa.Uuid(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("disabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        # Composite, not a plain FK on org_provider_key_id alone: pins the
        # referenced key to *this row's own* organization_id, so a cross-organization
        # override (a workspace in org A pointing at org B's key) is a foreign-key
        # violation rather than a silently-persisted row.
        sa.ForeignKeyConstraint(
            ["organization_id", "org_provider_key_id"],
            ["org_provider_keys.organization_id", "org_provider_keys.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "org_provider_key_id", name="uq_workspace_provider_key_overrides_ws_key"),
    )
    op.create_index(
        op.f("ix_workspace_provider_key_overrides_workspace_id"),
        "workspace_provider_key_overrides",
        ["workspace_id"],
    )
    op.create_index(
        op.f("ix_workspace_provider_key_overrides_org_provider_key_id"),
        "workspace_provider_key_overrides",
        ["org_provider_key_id"],
    )

    op.create_table(
        "workspace_provider_model_restrictions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        # Same denormalization and reasoning as the override table above.
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("org_provider_key_id", sa.Uuid(), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["organization_id", "org_provider_key_id"],
            ["org_provider_keys.organization_id", "org_provider_keys.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "workspace_id",
            "org_provider_key_id",
            "model",
            name="uq_workspace_provider_model_restrictions_ws_key_model",
        ),
    )
    op.create_index(
        op.f("ix_workspace_provider_model_restrictions_workspace_id"),
        "workspace_provider_model_restrictions",
        ["workspace_id"],
    )
    op.create_index(
        op.f("ix_workspace_provider_model_restrictions_org_provider_key_id"),
        "workspace_provider_model_restrictions",
        ["org_provider_key_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_workspace_provider_model_restrictions_org_provider_key_id"),
        table_name="workspace_provider_model_restrictions",
    )
    op.drop_index(
        op.f("ix_workspace_provider_model_restrictions_workspace_id"),
        table_name="workspace_provider_model_restrictions",
    )
    op.drop_table("workspace_provider_model_restrictions")

    op.drop_index(
        op.f("ix_workspace_provider_key_overrides_org_provider_key_id"),
        table_name="workspace_provider_key_overrides",
    )
    op.drop_index(
        op.f("ix_workspace_provider_key_overrides_workspace_id"), table_name="workspace_provider_key_overrides"
    )
    op.drop_table("workspace_provider_key_overrides")

    op.drop_index(_ORG_DEFAULT_INDEX, table_name="org_provider_keys")
    op.drop_index(op.f("ix_org_provider_keys_organization_id"), table_name="org_provider_keys")
    op.drop_table("org_provider_keys")
