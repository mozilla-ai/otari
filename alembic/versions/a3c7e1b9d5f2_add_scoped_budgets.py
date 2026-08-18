"""Add the tenancy-scoped budget table.

A second, fully enforced budget mechanism beside ``budgets``, which is left
exactly as it is. ``budgets`` is many-to-one from ``users`` and is checked
against ``users.spend + users.reserved``, so N users sharing one budget each get
the full limit; moving counters onto that row would silently convert it into a
pooled cap. This table carries its own counters and its own period window
instead, keyed on ``(scope_type, scope_id)`` plus an optional provider narrowing.

Two partial unique indexes rather than one plain UNIQUE over the triple:
PostgreSQL treats NULLs as distinct, so an index that included the nullable
``provider_key_id`` would enforce nothing at all on the aggregate rows. The
narrowed rows are unique on the triple; the aggregate rows are unique on the
identity alone.

Nothing here is a foreign key. A scope names a row in one of four tables
depending on ``scope_type``, and a provider instance may be configured only in
``config.yml`` and have no row anywhere.

Revision ID: a3c7e1b9d5f2
Revises: d5e7f1a2b3c4
Create Date: 2026-08-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3c7e1b9d5f2"
down_revision: str | Sequence[str] | None = "d5e7f1a2b3c4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WITH_KEY = "uq_scoped_budgets_scope_with_key"
_NO_KEY = "uq_scoped_budgets_scope_no_key"
_SCOPE = "ix_scoped_budgets_scope"


def upgrade() -> None:
    op.create_table(
        "scoped_budgets",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("scope_type", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("provider_key_id", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("max_budget", sa.Float(), nullable=True),
        sa.Column("current_spend", sa.Float(), nullable=False, server_default="0"),
        sa.Column("reserved_spend", sa.Float(), nullable=False, server_default="0"),
        sa.Column("budget_duration_sec", sa.Integer(), nullable=True),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        _WITH_KEY,
        "scoped_budgets",
        ["scope_type", "scope_id", "provider_key_id"],
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NOT NULL"),
        sqlite_where=sa.text("provider_key_id IS NOT NULL"),
    )
    op.create_index(
        _NO_KEY,
        "scoped_budgets",
        ["scope_type", "scope_id"],
        unique=True,
        postgresql_where=sa.text("provider_key_id IS NULL"),
        sqlite_where=sa.text("provider_key_id IS NULL"),
    )
    op.create_index(_SCOPE, "scoped_budgets", ["scope_type", "scope_id"])


def downgrade() -> None:
    op.drop_index(_SCOPE, table_name="scoped_budgets")
    op.drop_index(_NO_KEY, table_name="scoped_budgets")
    op.drop_index(_WITH_KEY, table_name="scoped_budgets")
    op.drop_table("scoped_budgets")
