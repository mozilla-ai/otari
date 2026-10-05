"""Let a service key bill end users of its own.

A key marked as a service key may name an end user in a request's ``user``
field. The end user is a ``users`` row owned by the key's user, created on
first use with the key's end-user budget, so each end user has a budget of its
own while the key's own ceiling pools them.

The two foreign keys are added inline on SQLite, which has ``ADD COLUMN ...
REFERENCES`` but no ``ADD CONSTRAINT``, so neither table is rebuilt: a rebuild
reflects ``api_keys``, and its partial index is what reflection loses.

Revision ID: a4d8e2f6b0c3
Revises: c4e8a2f6b1d3
Create Date: 2026-09-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4d8e2f6b0c3"
down_revision: str | Sequence[str] | None = "c4e8a2f6b1d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BUDGET_FK = "fk_api_keys_end_user_budget_id"
_PARENT_FK = "fk_users_parent_user_id"
_END_USER_INDEX = "uq_users_parent_external_id"


def _add_reference(table: str, column: str, target: str, *, name: str, ondelete: str) -> None:
    """Add a nullable string column holding a foreign key to ``target`` (``table.column``)."""
    if op.get_bind().dialect.name == "sqlite":
        target_table, target_column = target.split(".")
        op.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} VARCHAR "
            f"CONSTRAINT {name} REFERENCES {target_table} ({target_column}) ON DELETE {ondelete}"
        )
        return
    op.add_column(table, sa.Column(column, sa.String(), nullable=True))
    target_table, target_column = target.split(".")
    op.create_foreign_key(name, table, target_table, [column], [target_column], ondelete=ondelete)


def upgrade() -> None:
    """Add the service-key settings and the end-user owner."""
    op.add_column("api_keys", sa.Column("is_service_key", sa.Boolean(), nullable=False, server_default=sa.false()))
    _add_reference("api_keys", "end_user_budget_id", "budgets.budget_id", name=_BUDGET_FK, ondelete="SET NULL")
    op.create_index(op.f("ix_api_keys_end_user_budget_id"), "api_keys", ["end_user_budget_id"], unique=False)

    _add_reference("users", "parent_user_id", "users.user_id", name=_PARENT_FK, ondelete="CASCADE")
    op.add_column("users", sa.Column("external_id", sa.String(), nullable=True))
    op.create_index(op.f("ix_users_parent_user_id"), "users", ["parent_user_id"], unique=False)
    op.create_index(_END_USER_INDEX, "users", ["parent_user_id", "external_id"], unique=True)


def downgrade() -> None:
    """Drop the end users' owner and the service-key settings.

    End users are left in place as ordinary users: their spend and usage rows
    stay, and nothing can name them by their external id any more.
    """
    sqlite = op.get_bind().dialect.name == "sqlite"

    op.drop_index(_END_USER_INDEX, table_name="users")
    op.drop_index(op.f("ix_users_parent_user_id"), table_name="users")
    with op.batch_alter_table("users") as batch:
        if not sqlite:
            batch.drop_constraint(_PARENT_FK, type_="foreignkey")
        batch.drop_column("external_id")
        batch.drop_column("parent_user_id")

    op.drop_index(op.f("ix_api_keys_end_user_budget_id"), table_name="api_keys")
    with op.batch_alter_table("api_keys") as batch:
        if not sqlite:
            batch.drop_constraint(_BUDGET_FK, type_="foreignkey")
        batch.drop_column("end_user_budget_id")
        batch.drop_column("is_service_key")
