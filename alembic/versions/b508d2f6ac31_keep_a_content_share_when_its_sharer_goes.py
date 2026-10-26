"""Keep a content share, and the reads recorded against it, when the account that made it is deleted.

A read cascades from its share, so the share has to outlive the identity that
made it for the session owner's record of who opened their content to survive.
``trace_content_shares.shared_by_user_id`` therefore becomes nullable with
SET NULL, as ``reader_user_id`` already is beside a stored ``reader_email``.

Nobody reads through such a share: a read checks the session owner's
membership first, and a deleted account is no member.

Revision ID: b508d2f6ac31
Revises: f1a93c7e5d28
Create Date: 2026-10-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b508d2f6ac31"
down_revision: str | Sequence[str] | None = "f1a93c7e5d28"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "trace_content_shares"
_COLUMN = "shared_by_user_id"
_FK = "fk_trace_content_shares_shared_by_user"
_RUNNING = sa.text("revoked_at IS NULL")


def _named(fallback: str) -> str:
    """The name the live schema gives the foreign key on ``_COLUMN``, or ``fallback``.

    The constraint this replaces was created without a name, so its name is
    whatever the engine chose. Reflection answers that; a convention guessed
    from the table and column would fail the migration wherever it is wrong.
    """
    found = sa.inspect(op.get_bind()).get_foreign_keys(_TABLE)
    for key in found:
        if key["constrained_columns"] == [_COLUMN] and key.get("name"):
            return str(key["name"])
    return fallback


def _shares(*, nullable: bool) -> sa.Table:
    """The table as it stands, minus the constraint being replaced.

    SQLite rebuilds the table from this, so every index belongs here or the rebuild
    drops it, and the old foreign key does not, or the rebuild keeps it beside the
    new one and the stricter of the two decides.
    """
    return sa.Table(
        _TABLE,
        sa.MetaData(),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("audience", sa.String(length=32), nullable=False),
        sa.Column("note", sa.String(length=500), nullable=True),
        sa.Column("shared_by_user_id", sa.Uuid(), nullable=nullable),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "audience IN ('organization_admins', 'deployment_operator')", name="check_trace_content_shares_audience"
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "trace_id"],
            ["traces.workspace_id", "traces.trace_id"],
            ondelete="CASCADE",
            name="fk_trace_content_shares_trace",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("ix_trace_content_shares_trace", "workspace_id", "trace_id"),
        sa.Index("ix_trace_content_shares_shared_by_user_id", "shared_by_user_id"),
        sa.Index(
            "uq_trace_content_shares_running",
            "workspace_id",
            "trace_id",
            "audience",
            unique=True,
            postgresql_where=_RUNNING,
            sqlite_where=_RUNNING,
        ),
    )


def _repoint(*, nullable: bool, ondelete: str) -> None:
    # Read before the block opens: inside it the table may already be a copy.
    drop = _named(_FK)
    with op.batch_alter_table(_TABLE, copy_from=_shares(nullable=not nullable)) as batch:
        batch.alter_column(_COLUMN, existing_type=sa.Uuid(), nullable=nullable)
        if op.get_bind().dialect.name != "sqlite":
            # SQLite's rebuild leaves the old constraint behind with the table; every
            # other engine alters in place and has to be told to drop it.
            batch.drop_constraint(drop, type_="foreignkey")
        batch.create_foreign_key(_FK, "user", [_COLUMN], ["id"], ondelete=ondelete)


def upgrade() -> None:
    _repoint(nullable=True, ondelete="SET NULL")


def downgrade() -> None:
    # A share whose account is already gone has nobody to point at, so it goes.
    op.execute(sa.text(f"DELETE FROM {_TABLE} WHERE {_COLUMN} IS NULL"))
    _repoint(nullable=False, ondelete="CASCADE")
