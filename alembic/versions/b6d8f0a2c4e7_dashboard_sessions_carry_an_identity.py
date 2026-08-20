"""Give a dashboard session the identity it resolves to.

Adds ``user_id`` to ``dashboard_sessions``, so a session answers "who is
calling" rather than only "the master key was presented once". It names a
tenancy identity, and that identity's ``active_organization_id`` is the
organization the session acts in.

NOT NULL, because a session naming nobody is exactly the state the column
exists to remove. Existing rows are bound to the identity master-key auth
already resolves to, the bootstrap operator named by the
``tenancy_bootstrap_user_id`` marker, so a signed-in operator stays signed in
across the upgrade. A deployment that has never served a tenancy request has no
such identity, and provisioning is lazy, so there is nothing to attribute its
sessions to: those rows are deleted and the operator signs in once more. That
is the same outcome a master-key rotation already produces, and it is preferred
over seeding an identity here, which would duplicate provisioning (organization,
workspace, memberships, attribution user, marker) in SQL and drift from it.

CASCADE on the foreign key: deleting an identity revokes its sessions instead of
leaving a live cookie pointing at a row that is gone.

Revision ID: b6d8f0a2c4e7
Revises: 7ff4e082eb0c
Create Date: 2026-08-20 00:00:00.000000

"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b6d8f0a2c4e7"
down_revision: str | Sequence[str] | None = "7ff4e082eb0c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must match provisioning_service, which anchors the operator identity here.
BOOTSTRAP_IDENTITY_KEY = "tenancy_bootstrap_user_id"

_USER_ID_INDEX = "ix_dashboard_sessions_user_id"
_USER_ID_FK = "fk_dashboard_sessions_user_id"


def _dashboard_sessions(*, with_user_id: bool, user_id_nullable: bool = True) -> sa.Table:
    """The table as it stands at each ``batch_alter_table`` below.

    ``copy_from`` is what keeps SQLite's table rebuild from dropping what
    reflection could not see, so the expiry index is declared here rather than
    left to be rediscovered. The foreign key is not declared: a rebuild that
    omits ``user_id`` drops it either way, which is all the downgrade needs.
    """
    columns = [
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    ]
    if with_user_id:
        columns.insert(1, sa.Column("user_id", sa.Uuid(), nullable=user_id_nullable))
    table = sa.Table(
        "dashboard_sessions",
        sa.MetaData(),
        *columns,
        sa.PrimaryKeyConstraint("token_hash"),
    )
    sa.Index("ix_dashboard_sessions_expires_at", table.c.expires_at)
    return table


def _bootstrap_identity_id(bind: sa.engine.Connection) -> uuid.UUID | None:
    """The identity every master-key request already resolves to, if one exists.

    Both halves are checked. The marker can name an identity that is gone (the
    runtime treats that as "not provisioned yet" and re-provisions), and binding
    a session to it would fail the foreign key added below.
    """
    marker = bind.execute(
        sa.text("SELECT value FROM runtime_settings WHERE key = :key"),
        {"key": BOOTSTRAP_IDENTITY_KEY},
    ).scalar()
    if marker is None:
        return None
    try:
        identity = uuid.UUID(str(marker))
    except ValueError:
        return None
    # A bound ``sa.Uuid`` parameter rather than a rendered literal: the type is
    # native on PostgreSQL and CHAR(32) hex on SQLite, and a hand-written
    # literal that matches one engine joins to nothing on the other.
    found = bind.execute(
        sa.text('SELECT id FROM "user" WHERE id = :id').bindparams(sa.bindparam("id", identity, type_=sa.Uuid())),
    ).scalar()
    return identity if found is not None else None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    op.add_column("dashboard_sessions", sa.Column("user_id", sa.Uuid(), nullable=True))

    operator = _bootstrap_identity_id(bind)
    if operator is not None:
        bind.execute(
            sa.text("UPDATE dashboard_sessions SET user_id = :owner WHERE user_id IS NULL").bindparams(
                sa.bindparam("owner", operator, type_=sa.Uuid())
            )
        )
    bind.execute(sa.text("DELETE FROM dashboard_sessions WHERE user_id IS NULL"))

    with op.batch_alter_table("dashboard_sessions", copy_from=_dashboard_sessions(with_user_id=True)) as batch:
        batch.alter_column("user_id", existing_type=sa.Uuid(), nullable=False)
        batch.create_foreign_key(_USER_ID_FK, "user", ["user_id"], ["id"], ondelete="CASCADE")
    op.create_index(op.f(_USER_ID_INDEX), "dashboard_sessions", ["user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # The index goes first: SQLite refuses to drop a column an index covers, and
    # the rebuild below is driven by a definition that no longer mentions it.
    op.drop_index(op.f(_USER_ID_INDEX), table_name="dashboard_sessions")
    with op.batch_alter_table(
        "dashboard_sessions",
        copy_from=_dashboard_sessions(with_user_id=True, user_id_nullable=False),
    ) as batch:
        batch.drop_column("user_id")
