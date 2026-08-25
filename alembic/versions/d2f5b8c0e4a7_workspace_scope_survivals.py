"""Scope routing memory, router preferences and files to a workspace.

``d5e7f1a2b3c4`` scoped the four request-plane tables the dashboard bills and
lists by. These three are the rest of what stays in the gateway rather than
moving to the platform (otari-ai#1643): the learned router's example store, its
preference audit trail, and uploaded file metadata. Same column, same shape, and
the same backfill onto the deployment's default workspace, so a standalone
gateway upgrades with nothing to re-issue and nothing to move.

NOT NULL and ``RESTRICT``, matching ``api_keys.workspace_id``: "no workspace" is
never a real state, and deleting a workspace must not silently take a user's
uploads or a router's training data with it.

``user_id`` is untouched on all three. The workspace is a second axis, not a
replacement: the router still reads one user's examples, and a file still belongs
to the user who uploaded it. What the workspace adds is that a user holding keys
in two workspaces no longer has one workspace's rows visible from the other.

The composite indexes on the two router tables are renamed rather than extended
in place, because every read now leads with the workspace: they are dropped
before the column arrives and recreated after it, so SQLite's table rebuild never
has a stale one to carry through reflection.

The ``server_default`` stays for the reason ``d5e7f1a2b3c4`` gives: the backfill
needs it, and it protects no writer afterwards, since SQLAlchemy sends an
explicit NULL for a mapped column it was given no value for.

Revision ID: d2f5b8c0e4a7
Revises: c1e4a7b9d3f6
Create Date: 2026-08-25
"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d2f5b8c0e4a7"
down_revision: str | Sequence[str] | None = "c1e4a7b9d3f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must match provisioning_service and d5e7f1a2b3c4, which look the default up by
# these values and only create one when the lookup misses.
DEFAULT_ORGANIZATION_NAME = "Default organization"
DEFAULT_ORGANIZATION_SLUG = "default"
DEFAULT_WORKSPACE_NAME = "Default workspace"

SCOPED_TABLES = ("routing_memory", "router_preferences", "file_objects")

# Old name -> (new name, columns). Renamed because the workspace now leads every
# read of these tables.
RENAMED_INDEXES: dict[str, tuple[str, list[str]]] = {
    "ix_routing_memory_user_model": (
        "ix_routing_memory_workspace_user_model",
        ["workspace_id", "user_id", "embedding_model"],
    ),
    "ix_routing_memory_user_created": (
        "ix_routing_memory_workspace_user_created",
        ["workspace_id", "user_id", "created_at"],
    ),
    "ix_routing_memory_user_model_task": (
        "ix_routing_memory_workspace_user_model_task",
        ["workspace_id", "user_id", "embedding_model", "task_id"],
    ),
    "ix_router_preferences_user_created": (
        "ix_router_preferences_workspace_user_created",
        ["workspace_id", "user_id", "created_at"],
    ),
}

_INDEX_TABLE = {
    "ix_routing_memory_user_model": "routing_memory",
    "ix_routing_memory_user_created": "routing_memory",
    "ix_routing_memory_user_model_task": "routing_memory",
    "ix_router_preferences_user_created": "router_preferences",
}

_OLD_INDEX_COLUMNS: dict[str, list[str]] = {
    "ix_routing_memory_user_model": ["user_id", "embedding_model"],
    "ix_routing_memory_user_created": ["user_id", "created_at"],
    "ix_routing_memory_user_model_task": ["user_id", "embedding_model", "task_id"],
    "ix_router_preferences_user_created": ["user_id", "created_at"],
}


def _uuid_literal(bind: sa.engine.Connection, value: uuid.UUID) -> str:
    """Render a UUID the way this dialect stores one.

    ``sa.Uuid`` is native on PostgreSQL and CHAR(32) hex on SQLite, and a
    ``server_default`` is raw SQL rather than a bound parameter, so the literal
    has to match the storage form or every backfilled row reads back as a value
    that joins to nothing.
    """
    return value.hex if bind.dialect.name == "sqlite" else str(value)


def _default_workspace_id(bind: sa.engine.Connection) -> uuid.UUID:
    """The workspace existing rows are backfilled onto.

    ``d5e7f1a2b3c4`` runs before this and seeds the default when tenancy was
    never touched, so the first lookup answers on every migrated database. The
    fallbacks are here for the database an operator has since renamed or reshaped
    by hand: adopt the oldest workspace before minting a second default, matching
    ``services/workspace_scope.default_workspace_id``.
    """
    organization_id = bind.execute(
        sa.text("SELECT id FROM organization WHERE slug = :slug"),
        {"slug": DEFAULT_ORGANIZATION_SLUG},
    ).scalar()
    if organization_id is not None:
        workspace_id = bind.execute(
            sa.text("SELECT id FROM workspace WHERE organization_id = :org AND name = :name"),
            {"org": _uuid_literal(bind, uuid.UUID(str(organization_id))), "name": DEFAULT_WORKSPACE_NAME},
        ).scalar()
        if workspace_id is not None:
            return uuid.UUID(str(workspace_id))

    # The id breaks a ``created_at`` tie so this and the runtime resolve the same
    # row when one transaction created two workspaces.
    workspace_id = bind.execute(
        sa.text("SELECT id FROM workspace ORDER BY created_at, id LIMIT 1")
    ).scalar()
    if workspace_id is not None:
        return uuid.UUID(str(workspace_id))

    return _seed_default_workspace(bind)


def _seed_default_workspace(bind: sa.engine.Connection) -> uuid.UUID:
    """Create the default organization and workspace, returning the workspace id.

    Only reachable on a database whose tenancy rows were removed after
    ``d5e7f1a2b3c4`` seeded them, which the ``RESTRICT`` foreign keys make
    possible exactly while no request-plane row references them. Seeding rather
    than failing keeps that database upgradable, and the slug and name match what
    provisioning looks up, so a later first boot adopts these rows.
    """
    organization_id = bind.execute(
        sa.text("SELECT id FROM organization WHERE slug = :slug"),
        {"slug": DEFAULT_ORGANIZATION_SLUG},
    ).scalar()
    if organization_id is None:
        organization_id = uuid.uuid4()
        bind.execute(
            sa.text(
                "INSERT INTO organization (id, name, slug, created_by_user_id, created_at) "
                "VALUES (:id, :name, :slug, NULL, CURRENT_TIMESTAMP)"
            ),
            {
                "id": _uuid_literal(bind, organization_id),
                "name": DEFAULT_ORGANIZATION_NAME,
                "slug": DEFAULT_ORGANIZATION_SLUG,
            },
        )
    else:
        organization_id = uuid.UUID(str(organization_id))

    workspace_id = uuid.uuid4()
    bind.execute(
        sa.text(
            "INSERT INTO workspace "
            "(id, organization_id, name, description, created_by_user_id, "
            " activation_classification, created_at) "
            "VALUES (:id, :org, :name, NULL, NULL, 'eligible', CURRENT_TIMESTAMP)"
        ),
        {
            "id": _uuid_literal(bind, workspace_id),
            "org": _uuid_literal(bind, organization_id),
            "name": DEFAULT_WORKSPACE_NAME,
        },
    )
    return workspace_id


def upgrade() -> None:
    bind = op.get_bind()
    literal = _uuid_literal(bind, _default_workspace_id(bind))

    for old_name in RENAMED_INDEXES:
        op.drop_index(old_name, table_name=_INDEX_TABLE[old_name])

    for table in SCOPED_TABLES:
        op.add_column(
            table,
            sa.Column("workspace_id", sa.Uuid(), nullable=False, server_default=sa.text(f"'{literal}'")),
        )
        op.create_index(op.f(f"ix_{table}_workspace_id"), table, ["workspace_id"], unique=False)
        # SQLite has no ``ALTER TABLE ... ADD CONSTRAINT``, so this rebuilds the
        # table there. Reflection carries the remaining single-column indexes
        # through; the composite ones were dropped above and are recreated below,
        # so none of them depends on how well SQLite reflects a multi-column index.
        with op.batch_alter_table(table) as batch:
            batch.create_foreign_key(
                f"fk_{table}_workspace_id",
                "workspace",
                ["workspace_id"],
                ["id"],
                ondelete="RESTRICT",
            )

    for old_name, (new_name, columns) in RENAMED_INDEXES.items():
        op.create_index(new_name, _INDEX_TABLE[old_name], columns)


def downgrade() -> None:
    for old_name, (new_name, _columns) in RENAMED_INDEXES.items():
        op.drop_index(new_name, table_name=_INDEX_TABLE[old_name])

    for table in SCOPED_TABLES:
        with op.batch_alter_table(table) as batch:
            batch.drop_constraint(f"fk_{table}_workspace_id", type_="foreignkey")
        # Dropped before the column: SQLite refuses ``DROP COLUMN`` while an index
        # covers it.
        op.drop_index(op.f(f"ix_{table}_workspace_id"), table_name=table)
        op.drop_column(table, "workspace_id")

    for old_name, columns in _OLD_INDEX_COLUMNS.items():
        op.create_index(old_name, _INDEX_TABLE[old_name], columns)
