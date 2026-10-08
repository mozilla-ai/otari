"""Let a scoped ceiling narrow to one model of a provider.

``scoped_budgets`` had one resource axis, ``provider_key_id``. ``model`` joins it
as a second, so a budget can apply to "gpt-4o on openai" as well as to "openai".

**A model needs a provider.** Model ids are only unique per provider, and
pricing keys them ``provider:model``, so the CHECK refuses a model with no
provider rather than guessing which provider an unqualified id means.

**One unique index replaces two partial ones.** Each entity carries at most one
budget, and an entity is the scope plus both resource axes, NULL reading as
"all". PostgreSQL and SQLite treat NULLs as distinct in a plain UNIQUE, which is
why the old pair was partial; a second nullable axis would have taken three.
``COALESCE(..., '')`` makes "all" a value that collides with itself, so one
index covers every shape on both dialects. ``''`` is free to mean "all" because
the API refuses a blank provider or model.

No backfill: every existing row is provider-wide or aggregate, which is what
``model IS NULL`` already says. ``downgrade()`` deletes model-narrowed ceilings,
since the old indexes cannot hold two ceilings that differ only by model.

Revision ID: a7c3e9f1b5d2
Revises: d4b8e2f6a917
Create Date: 2026-10-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7c3e9f1b5d2"
down_revision: str | Sequence[str] | None = "d4b8e2f6a917"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "scoped_budgets"
_ENTITY_INDEX = "uq_scoped_budgets_entity"
_ENTITY_EXPRESSIONS = ("scope_type", "scope_id", "coalesce(provider_key_id, '')", "coalesce(model, '')")
_WITH_KEY = "uq_scoped_budgets_scope_with_key"
_NO_KEY = "uq_scoped_budgets_scope_no_key"
_CHECK = "ck_scoped_budgets_model_needs_provider"
_CHECK_SQL = "model IS NULL OR provider_key_id IS NOT NULL"


def _scoped_budgets() -> sa.Table:
    """The table with ``model``, for SQLite's rebuild.

    ``copy_from`` is what the rebuild recreates the table from, so the FK is
    named and every index is declared: what is not here is dropped.
    """
    table = sa.Table(
        _TABLE,
        sa.MetaData(),
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("scope_type", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("provider_key_id", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column(
            "budget_id",
            sa.String(),
            sa.ForeignKey("budgets.budget_id", name="fk_scoped_budgets_budget_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("current_spend", sa.Numeric(18, 6), nullable=False, server_default="0"),
        sa.Column("reserved_spend", sa.Numeric(18, 6), nullable=False, server_default="0"),
        sa.Column("current_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("reserved_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("current_requests", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("reserved_requests", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(_CHECK_SQL, name=_CHECK),
    )
    blank = sa.literal_column("''")
    sa.Index(
        _ENTITY_INDEX,
        table.c.scope_type,
        table.c.scope_id,
        sa.func.coalesce(table.c.provider_key_id, blank),
        sa.func.coalesce(table.c.model, blank),
        unique=True,
    )
    sa.Index("ix_scoped_budgets_scope", table.c.scope_type, table.c.scope_id)
    sa.Index("ix_scoped_budgets_budget_id", table.c.budget_id)
    return table


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("model", sa.String(), nullable=True))
    op.drop_index(_WITH_KEY, table_name=_TABLE)
    op.drop_index(_NO_KEY, table_name=_TABLE)
    op.create_index(_ENTITY_INDEX, _TABLE, [sa.text(expression) for expression in _ENTITY_EXPRESSIONS], unique=True)
    with op.batch_alter_table(_TABLE, copy_from=_scoped_budgets()) as batch:
        batch.create_check_constraint(_CHECK, _CHECK_SQL)


def downgrade() -> None:
    op.execute(sa.text("DELETE FROM scoped_budgets WHERE model IS NOT NULL"))
    with op.batch_alter_table(_TABLE, copy_from=_scoped_budgets()) as batch:
        batch.drop_constraint(_CHECK, type_="check")
        batch.drop_index(_ENTITY_INDEX)
        batch.drop_column("model")
    op.create_index(
        _WITH_KEY,
        _TABLE,
        ["scope_type", "scope_id", "provider_key_id"],
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NOT NULL"),
        sqlite_where=sa.text("provider_key_id IS NOT NULL"),
    )
    op.create_index(
        _NO_KEY,
        _TABLE,
        ["scope_type", "scope_id"],
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NULL"),
        sqlite_where=sa.text("provider_key_id IS NULL"),
    )
