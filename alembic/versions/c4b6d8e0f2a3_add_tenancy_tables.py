"""Add the reconciled control plane's tenancy tables.

Organizations, workspaces, identities, and the two membership tables, rehomed
from the platform's schema so the OSS control plane owns them (M5).

Two things this revision has to get right that autogenerate does not:

- **The circular foreign key.** ``user.active_organization_id`` references
  ``organization.id`` and ``organization.created_by_user_id`` references
  ``user.id``, so no creation order satisfies both inline. ``organization`` is
  created without its reference to ``user``, ``user`` is created with its
  reference to ``organization``, and the remaining constraint is added
  afterwards. PostgreSQL takes that as a plain ``ALTER TABLE``; SQLite has no
  ``ADD CONSTRAINT``, so ``batch_alter_table`` rebuilds the table. The rebuild
  runs here, before ``workspace`` and ``organization_member`` exist, so it has
  no dependents to invalidate, and ``copy_from`` states the table explicitly
  rather than trusting SQLite reflection to round-trip the unique constraint.
- **Dialect-neutral defaults.** ``sa.func.now()`` renders as ``now()`` on
  PostgreSQL and ``CURRENT_TIMESTAMP`` on SQLite. A literal from either dialect
  would break the other, and the shared chain has to run on both.

Revision ID: c4b6d8e0f2a3
Revises: b2d4f6a8c0e1
Create Date: 2026-08-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4b6d8e0f2a3"
down_revision: str | Sequence[str] | None = "b2d4f6a8c0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ORGANIZATION_USER_FK = "fk_organization_created_by_user_id"


def _organization_table(*, with_user_fk: bool) -> sa.Table:
    """Describe ``organization`` for ``batch_alter_table``'s ``copy_from``.

    SQLite's batch mode rebuilds the table, and it can only reproduce what it
    knows about. Passing the definition explicitly keeps the named unique
    constraint from degrading into a reflected index.
    """
    constraints: list[sa.SchemaItem] = [
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_organization_slug"),
    ]
    if with_user_fk:
        constraints.append(
            sa.ForeignKeyConstraint(
                ["created_by_user_id"],
                ["user.id"],
                name=_ORGANIZATION_USER_FK,
                ondelete="SET NULL",
            )
        )
    return sa.Table(
        "organization",
        sa.MetaData(),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        *constraints,
    )


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "organization",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_organization_slug"),
    )

    op.create_table(
        "user",
        sa.Column("id", sa.Uuid(), nullable=False),
        # Nullable: a standalone operator identity, and every gateway user M4
        # re-parents, is an operator-defined label with no sign-in address.
        # Both engines allow repeated NULLs in a unique index, so those rows
        # coexist without weakening uniqueness for real addresses.
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_superuser", sa.Boolean(), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("active_organization_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        # No ondelete: an organization with members must not be deletable out
        # from under them. The delete path repoints its members first.
        sa.ForeignKeyConstraint(
            ["active_organization_id"],
            ["organization.id"],
            name="fk_user_active_organization_id",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_user_active_organization_id"), "user", ["active_organization_id"], unique=False)
    op.create_index(op.f("ix_user_email"), "user", ["email"], unique=True)

    with op.batch_alter_table("organization", copy_from=_organization_table(with_user_fk=False)) as batch_op:
        batch_op.create_foreign_key(
            _ORGANIZATION_USER_FK,
            "user",
            ["created_by_user_id"],
            ["id"],
            ondelete="SET NULL",
        )

    op.create_table(
        "organization_member",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "user_id", name="uq_organization_member_organization_user"),
    )
    op.create_index(
        op.f("ix_organization_member_organization_id"), "organization_member", ["organization_id"], unique=False
    )
    op.create_index(op.f("ix_organization_member_user_id"), "organization_member", ["user_id"], unique=False)

    op.create_table(
        "workspace",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=True),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        # Edition-invariant schema: the OSS control plane never writes this, but
        # the overlay contributes no tables of its own, so the column the hosted
        # activation surface reads has to exist in the one schema both boot.
        sa.Column("activation_classification", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "activation_classification IN ('eligible', 'internal', 'automated', 'migrated', 'enterprise_assisted')",
            name="check_workspace_activation_classification",
        ),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "name", name="uq_workspace_organization_name"),
    )
    op.create_index(op.f("ix_workspace_organization_id"), "workspace", ["organization_id"], unique=False)

    op.create_table(
        "workspace_member",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member_workspace_user"),
    )
    op.create_index(op.f("ix_workspace_member_user_id"), "workspace_member", ["user_id"], unique=False)
    op.create_index(op.f("ix_workspace_member_workspace_id"), "workspace_member", ["workspace_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_workspace_member_workspace_id"), table_name="workspace_member")
    op.drop_index(op.f("ix_workspace_member_user_id"), table_name="workspace_member")
    op.drop_table("workspace_member")

    op.drop_index(op.f("ix_workspace_organization_id"), table_name="workspace")
    op.drop_table("workspace")

    op.drop_index(op.f("ix_organization_member_user_id"), table_name="organization_member")
    op.drop_index(op.f("ix_organization_member_organization_id"), table_name="organization_member")
    op.drop_table("organization_member")

    # Break the cycle in the other direction before either table goes: dropping
    # "user" while "organization" still references it fails on PostgreSQL.
    with op.batch_alter_table("organization", copy_from=_organization_table(with_user_fk=True)) as batch_op:
        batch_op.drop_constraint(_ORGANIZATION_USER_FK, type_="foreignkey")

    op.drop_index(op.f("ix_user_email"), table_name="user")
    op.drop_index(op.f("ix_user_active_organization_id"), table_name="user")
    op.drop_table("user")

    op.drop_table("organization")
