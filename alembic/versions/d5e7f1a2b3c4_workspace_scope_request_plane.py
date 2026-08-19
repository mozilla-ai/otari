"""Scope the request plane to a workspace.

Adds ``workspace_id`` to the four request-plane tables the dashboard scopes by,
backfilled onto the deployment's default workspace so an upgraded gateway keeps
working unchanged.

NOT NULL on purpose. A workspace is the unit the dashboard scopes by, so "no
workspace" is never a real state and the dashboard never has to tell an
unscoped row from an unmigrated one.

That forces this migration to seed the default organization and workspace when
tenancy was never touched: provisioning is lazy (the first master-key request to
a tenancy route does it), so a gateway that has only ever served completions has
neither, and there would be nothing to backfill onto. The seed uses the same
slug and workspace name ``provisioning_service`` looks up, so a later first-boot
adopts these rows instead of creating a second default. It deliberately creates
no identity: ``organization.created_by_user_id`` is nullable, and provisioning
fills it in when it runs.

The column keeps its ``server_default``, and not for the reason a first draft of
this docstring gave: the ``batch_alter_table`` below already rebuilds all four
tables, since SQLite has no ``ALTER TABLE ADD CONSTRAINT``, so a rebuild is
paid for either way. It stays because the backfill needs it, and for nothing
after that. It protects no writer: it covers only an INSERT that omits the
column, and the ORM never does, since SQLAlchemy sends an explicit NULL for a
mapped column it was given no value for. A write path that has not learned to
carry a workspace therefore still fails its NOT NULL, on a migrated database as
much as on any other, which is the loud failure rather than a row quietly filed
under the wrong workspace. Dropping the default belongs with the change that
finishes threading the workspace through every writer.

Revision ID: d5e7f1a2b3c4
Revises: c4b6d8e0f2a3
Create Date: 2026-08-18
"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d5e7f1a2b3c4"
down_revision: str | None = "c4b6d8e0f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must match provisioning_service, which looks the default up by these values
# and only creates one when the lookup misses.
DEFAULT_ORGANIZATION_NAME = "Default organization"
DEFAULT_ORGANIZATION_SLUG = "default"
DEFAULT_WORKSPACE_NAME = "Default workspace"

SCOPED_TABLES = ("api_keys", "usage_logs", "model_aliases", "routing_policies")


def _uuid_literal(bind: sa.engine.Connection, value: uuid.UUID) -> str:
    """Render a UUID the way this dialect stores one.

    ``sa.Uuid`` is native on PostgreSQL and CHAR(32) hex on SQLite, and a
    ``server_default`` is raw SQL rather than a bound parameter, so the literal
    has to match the storage form or every backfilled row reads back as a value
    that joins to nothing.
    """
    return value.hex if bind.dialect.name == "sqlite" else str(value)


def _ensure_default_workspace(bind: sa.engine.Connection) -> uuid.UUID:
    """Return the default workspace id, seeding the tenancy root if it is absent."""
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

    workspace_id = bind.execute(
        sa.text("SELECT id FROM workspace WHERE organization_id = :org AND name = :name"),
        {"org": _uuid_literal(bind, organization_id), "name": DEFAULT_WORKSPACE_NAME},
    ).scalar()
    if workspace_id is not None:
        return uuid.UUID(str(workspace_id))

    # Adopt the organization's oldest workspace before minting one, matching the
    # fallback in `services/workspace_scope.default_workspace_id`. An operator who
    # renamed the default would otherwise get a second workspace here, every
    # existing key and usage row backfilled into it, and no member on it, since
    # provisioning's marker already resolves and it will not revisit the
    # organization. The id breaks a `created_at` tie so the migration and the
    # runtime resolve the same row.
    workspace_id = bind.execute(
        sa.text("SELECT id FROM workspace WHERE organization_id = :org ORDER BY created_at, id LIMIT 1"),
        {"org": _uuid_literal(bind, organization_id)},
    ).scalar()
    if workspace_id is not None:
        return uuid.UUID(str(workspace_id))

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
    default_workspace_id = _ensure_default_workspace(bind)
    literal = _uuid_literal(bind, default_workspace_id)

    for table in SCOPED_TABLES:
        # Added NOT NULL with the default in one statement, which both engines
        # accept on a populated table, so there is no add-then-backfill-then-
        # tighten dance. It does not avoid SQLite's table rebuild, as an earlier
        # draft of this comment claimed: the foreign key below needs one anyway
        # (see the docstring). The partial indexes on the alias and policy tables
        # survive that rebuild, which `test_tenancy_schema_chain` pins.
        op.add_column(
            table,
            sa.Column("workspace_id", sa.Uuid(), nullable=False, server_default=sa.text(f"'{literal}'")),
        )
        op.create_index(op.f(f"ix_{table}_workspace_id"), table, ["workspace_id"], unique=False)
        with op.batch_alter_table(table) as batch:
            batch.create_foreign_key(
                f"fk_{table}_workspace_id",
                "workspace",
                ["workspace_id"],
                ["id"],
                ondelete="RESTRICT",
            )


def downgrade() -> None:
    for table in SCOPED_TABLES:
        with op.batch_alter_table(table) as batch:
            batch.drop_constraint(f"fk_{table}_workspace_id", type_="foreignkey")
        op.drop_index(op.f(f"ix_{table}_workspace_id"), table_name=table)
        op.drop_column(table, "workspace_id")

    # The seeded organization and workspace stay. They are ordinary tenancy rows
    # by now: an operator may have renamed them, added members, or created keys
    # against them through a later upgrade, and a downgrade of this migration is
    # not a statement about tenancy.
