"""Point a workspace budget default at a ``budgets`` row instead of restating one.

A default used to carry its own ``name``, ``max_budget`` and
``budget_duration_sec``, which made it a third thing spelled "budget": the
Budgets page could not say that a limit was a workspace's default, because the
two were unrelated rows that merely held equal numbers. It now names a budget,
so assigning one on a workspace and reading it back on the budget are the same
fact.

``budget_id`` is NOT NULL: a default with no budget is a template for nothing.
``RESTRICT`` rather than cascade, because deleting a budget that a workspace
hands to every member should be refused and explained, not silently withdraw
every materialized ceiling's limit.

The narrowing stays on the default: ``provider_key_id`` says which provider this
workspace applies the budget to, which is a property of the assignment rather
than of the budget, and the same budget may be a default for two workspaces that
narrow it differently.

Backfill: every existing default mints a ``budgets`` row carrying the numbers it
used to hold, so no workspace loses a limit. The downgrade copies them back and
leaves the minted rows behind rather than guessing which budgets were only ever
a default.

Revision ID: e2f4a6c8b0d3
Revises: a7c3e5d9b1f4
Create Date: 2026-08-21
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision: str = "e2f4a6c8b0d3"
down_revision: str | Sequence[str] | None = "a7c3e5d9b1f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WITH_KEY = "uq_workspace_budget_defaults_with_key"
_NO_KEY = "uq_workspace_budget_defaults_no_key"
_WORKSPACE_INDEX = "ix_workspace_budget_defaults_workspace_id"
_BUDGET_FK = "fk_workspace_budget_defaults_budget_id"


def _defaults_table(*, with_budget_id: bool, with_inline: bool, budget_id_nullable: bool = True) -> sa.Table:
    """The table as it stands at each ``batch_alter_table`` below.

    ``copy_from`` is what stops SQLite's rebuild from dropping what reflection
    could not see, so the three indexes are declared here, partial clauses
    included. The foreign keys are declared for the same reason: a rebuild that
    omits them drops the workspace one too.
    """
    columns = [
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("provider_key_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]
    if with_budget_id:
        columns.insert(2, sa.Column("budget_id", sa.String(), nullable=budget_id_nullable))
    if with_inline:
        columns.extend(
            [
                sa.Column("name", sa.String(), nullable=True),
                sa.Column("max_budget", sa.Float(), nullable=True),
                sa.Column("budget_duration_sec", sa.Integer(), nullable=True),
            ]
        )
    table = sa.Table(
        "workspace_budget_defaults",
        sa.MetaData(),
        *columns,
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
    )
    sa.Index(_WORKSPACE_INDEX, table.c.workspace_id)
    sa.Index(
        _WITH_KEY,
        table.c.workspace_id,
        table.c.provider_key_id,
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NOT NULL"),
        sqlite_where=sa.text("provider_key_id IS NOT NULL"),
    )
    sa.Index(
        _NO_KEY,
        table.c.workspace_id,
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NULL"),
        sqlite_where=sa.text("provider_key_id IS NULL"),
    )
    return table


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    op.add_column("workspace_budget_defaults", sa.Column("budget_id", sa.String(), nullable=True))

    # One budget per existing default, carrying the numbers that default held.
    # Named after the workspace where the default had no name of its own, so the
    # Budgets page does not fill with untitled rows.
    rows = bind.execute(
        sa.text(
            "SELECT d.id, d.name, d.max_budget, d.budget_duration_sec, w.name AS workspace_name "
            "FROM workspace_budget_defaults d JOIN workspace w ON w.id = d.workspace_id"
        )
    ).fetchall()
    now = datetime.now(UTC)
    for row in rows:
        budget_id = str(uuid.uuid4())
        bind.execute(
            sa.text(
                "INSERT INTO budgets (budget_id, name, max_budget, budget_duration_sec, created_at, updated_at) "
                "VALUES (:budget_id, :name, :max_budget, :duration, :created_at, :updated_at)"
            ),
            {
                "budget_id": budget_id,
                "name": row.name or f"{row.workspace_name} member default",
                "max_budget": row.max_budget,
                "duration": row.budget_duration_sec,
                "created_at": now,
                "updated_at": now,
            },
        )
        bind.execute(
            sa.text("UPDATE workspace_budget_defaults SET budget_id = :budget_id WHERE id = :id"),
            {"budget_id": budget_id, "id": row.id},
        )

    with op.batch_alter_table(
        "workspace_budget_defaults",
        copy_from=_defaults_table(with_budget_id=True, with_inline=True),
    ) as batch:
        batch.alter_column("budget_id", existing_type=sa.String(), nullable=False)
        batch.create_foreign_key(_BUDGET_FK, "budgets", ["budget_id"], ["budget_id"], ondelete="RESTRICT")
        batch.drop_column("name")
        batch.drop_column("max_budget")
        batch.drop_column("budget_duration_sec")


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    with op.batch_alter_table(
        "workspace_budget_defaults",
        copy_from=_defaults_table(with_budget_id=True, with_inline=False, budget_id_nullable=False),
    ) as batch:
        batch.add_column(sa.Column("name", sa.String(), nullable=True))
        batch.add_column(sa.Column("max_budget", sa.Float(), nullable=True))
        batch.add_column(sa.Column("budget_duration_sec", sa.Integer(), nullable=True))

    # Copy the numbers back off the budget each default names. The minted budgets
    # stay: nothing records which of them existed only to back a default, and
    # deleting a row a user may since have assigned to an identity would lose a
    # limit rather than restore one.
    bind.execute(
        sa.text(
            "UPDATE workspace_budget_defaults SET "
            "name = (SELECT b.name FROM budgets b WHERE b.budget_id = workspace_budget_defaults.budget_id), "
            "max_budget = (SELECT b.max_budget FROM budgets b WHERE b.budget_id = workspace_budget_defaults.budget_id), "
            "budget_duration_sec = (SELECT b.budget_duration_sec FROM budgets b "
            "WHERE b.budget_id = workspace_budget_defaults.budget_id)"
        )
    )

    with op.batch_alter_table(
        "workspace_budget_defaults",
        copy_from=_defaults_table(with_budget_id=True, with_inline=True, budget_id_nullable=False),
    ) as batch:
        # No `drop_constraint` for the budget foreign key: `copy_from` above does
        # not declare it, so SQLite's rebuild drops it with the column either
        # way, and naming it here fails on a table the rebuild does not believe
        # has one.
        batch.drop_column("budget_id")
