"""Widen alias and routing-policy uniqueness to the workspace.

``model_aliases`` and ``routing_policies`` have carried a ``workspace_id`` since
``d5e7f1a2b3c4``, but their uniqueness stayed at ``(name, user_id)``: resolution
read a process-wide cache keyed on name alone, so a second workspace's "fast"
would have silently shadowed the first at request time. ``alias_service`` and
``policy_store`` key that cache by workspace as of this change, so the
constraints can finally say what the column already implies.

Both tables get the same pair they had before, one column wider:

* ``uq_<table>_workspace_name_user`` over ``(workspace_id, name, user_id)``.
* ``uq_<table>_workspace_global_name``, a partial unique index over
  ``(workspace_id, name)`` where ``user_id IS NULL``. The composite constraint
  cannot cover the global rows, because SQLite and PostgreSQL both treat NULLs
  in a unique index as distinct.

Widening only ever admits rows, never rejects one: every pair that was unique
under ``(name, user_id)`` is still unique under ``(workspace_id, name,
user_id)``. The downgrade is the direction that can fail, and deliberately does
rather than deleting rows: two workspaces each holding a "fast" alias have no
representation in the narrower constraint, and dropping one of them is not a
choice a migration gets to make silently.

The partial indexes are dropped before the batch rebuild and recreated after.
SQLite has no ``ALTER TABLE ... DROP CONSTRAINT``, so the batch block rebuilds
the table, and a partial index carried through that rebuild by reflection is the
part most likely to come back subtly wrong.

Revision ID: c1e4a7b9d3f6
Revises: c7a1e4d8f3b6
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c1e4a7b9d3f6"
down_revision: str | Sequence[str] | None = "c7a1e4d8f3b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ALIAS_UQ_OLD = "uq_model_aliases_name_user"
_ALIAS_UQ_NEW = "uq_model_aliases_workspace_name_user"
_ALIAS_IX_OLD = "uq_model_aliases_global_name"
_ALIAS_IX_NEW = "uq_model_aliases_workspace_global_name"

_POLICY_UQ_OLD = "uq_routing_policies_name_user"
_POLICY_UQ_NEW = "uq_routing_policies_workspace_name_user"
_POLICY_IX_OLD = "uq_routing_policies_global_name"
_POLICY_IX_NEW = "uq_routing_policies_workspace_global_name"


def _model_aliases(*, workspace_scoped: bool) -> sa.Table:
    """``model_aliases`` as it stands on one side of this revision.

    Described explicitly rather than reflected, for the reason ``c3f7a9d1e5b8``
    gives: SQLite reflection does not reliably name a unique constraint, so
    ``drop_constraint`` could not address it and the batch recreate would swap it
    silently on behavior SQLAlchemy has reserved the right to turn into an error.
    """
    meta = sa.MetaData()
    return sa.Table(
        "model_aliases",
        meta,
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("target", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="model_aliases_pkey"),
        (
            sa.UniqueConstraint("workspace_id", "name", "user_id", name=_ALIAS_UQ_NEW)
            if workspace_scoped
            else sa.UniqueConstraint("name", "user_id", name=_ALIAS_UQ_OLD)
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.user_id"], name="fk_model_aliases_user_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspace.id"], name="fk_model_aliases_workspace_id", ondelete="RESTRICT"
        ),
        # Declared because ``copy_from`` replaces reflection wholesale: an index
        # left out here is an index SQLite's table rebuild drops on the floor.
        # The two partial unique indexes are deliberately absent, being dropped
        # before the rebuild and recreated after it.
        sa.Index("ix_model_aliases_user_id", "user_id"),
        sa.Index("ix_model_aliases_workspace_id", "workspace_id"),
    )


def _routing_policies(*, workspace_scoped: bool) -> sa.Table:
    """``routing_policies`` as it stands on one side of this revision."""
    meta = sa.MetaData()
    return sa.Table(
        "routing_policies",
        meta,
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="routing_policies_pkey"),
        (
            sa.UniqueConstraint("workspace_id", "name", "user_id", name=_POLICY_UQ_NEW)
            if workspace_scoped
            else sa.UniqueConstraint("name", "user_id", name=_POLICY_UQ_OLD)
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.user_id"], name="fk_routing_policies_user_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspace.id"], name="fk_routing_policies_workspace_id", ondelete="RESTRICT"
        ),
        sa.Index("ix_routing_policies_user_id", "user_id"),
        sa.Index("ix_routing_policies_workspace_id", "workspace_id"),
    )


def _global_index(name: str, table: str, columns: list[str]) -> None:
    op.create_index(
        name,
        table,
        columns,
        unique=True,
        sqlite_where=sa.text("user_id IS NULL"),
        postgresql_where=sa.text("user_id IS NULL"),
    )


def upgrade() -> None:
    op.drop_index(_ALIAS_IX_OLD, table_name="model_aliases")
    with op.batch_alter_table(
        "model_aliases", copy_from=_model_aliases(workspace_scoped=False)
    ) as batch:
        batch.drop_constraint(_ALIAS_UQ_OLD, type_="unique")
        batch.create_unique_constraint(_ALIAS_UQ_NEW, ["workspace_id", "name", "user_id"])
    _global_index(_ALIAS_IX_NEW, "model_aliases", ["workspace_id", "name"])

    op.drop_index(_POLICY_IX_OLD, table_name="routing_policies")
    with op.batch_alter_table(
        "routing_policies", copy_from=_routing_policies(workspace_scoped=False)
    ) as batch:
        batch.drop_constraint(_POLICY_UQ_OLD, type_="unique")
        batch.create_unique_constraint(_POLICY_UQ_NEW, ["workspace_id", "name", "user_id"])
    _global_index(_POLICY_IX_NEW, "routing_policies", ["workspace_id", "name"])


def downgrade() -> None:
    """Narrow both constraints back, failing loudly on rows that need the wider one.

    No row is deleted here. A name held by two workspaces is exactly the state
    the widening exists to allow, and the narrower constraint has no room for it,
    so the integrity error the recreate raises is the honest outcome: an operator
    rolling back has to decide which row survives.
    """
    op.drop_index(_ALIAS_IX_NEW, table_name="model_aliases")
    with op.batch_alter_table(
        "model_aliases", copy_from=_model_aliases(workspace_scoped=True)
    ) as batch:
        batch.drop_constraint(_ALIAS_UQ_NEW, type_="unique")
        batch.create_unique_constraint(_ALIAS_UQ_OLD, ["name", "user_id"])
    _global_index(_ALIAS_IX_OLD, "model_aliases", ["name"])

    op.drop_index(_POLICY_IX_NEW, table_name="routing_policies")
    with op.batch_alter_table(
        "routing_policies", copy_from=_routing_policies(workspace_scoped=True)
    ) as batch:
        batch.drop_constraint(_POLICY_UQ_NEW, type_="unique")
        batch.create_unique_constraint(_POLICY_UQ_OLD, ["name", "user_id"])
    _global_index(_POLICY_IX_OLD, "routing_policies", ["name"])
