"""Scope model aliases to users.

Adds a nullable ``user_id`` to ``model_aliases`` (NULL = global, the scope every
existing row keeps) and moves the primary key off ``name`` onto a surrogate
``id``, because the natural key is now (name, user_id) and a primary key cannot
contain a nullable column.

Revision ID: c3f7a9d1e5b8
Revises: b8c1d2e3f4a5
Create Date: 2026-07-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3f7a9d1e5b8"
down_revision: str | Sequence[str] | None = "b8c1d2e3f4a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    is_sqlite = conn.dialect.name == "sqlite"

    op.add_column("model_aliases", sa.Column("id", sa.String(), nullable=True))
    op.add_column("model_aliases", sa.Column("user_id", sa.String(), nullable=True))
    # Existing rows are global and their names are unique (name was the primary
    # key), so the name doubles as a back-filled surrogate id.
    conn.execute(sa.text("UPDATE model_aliases SET id = name WHERE id IS NULL"))

    with op.batch_alter_table("model_aliases") as batch_op:
        batch_op.alter_column("id", existing_type=sa.String(), nullable=False)
        if not is_sqlite:
            batch_op.drop_constraint("model_aliases_pkey", type_="primary")
        batch_op.create_primary_key("model_aliases_pkey", ["id"])
        batch_op.create_unique_constraint("uq_model_aliases_name_user", ["name", "user_id"])
        batch_op.create_foreign_key(
            "fk_model_aliases_user_id_users",
            "users",
            ["user_id"],
            ["user_id"],
            ondelete="CASCADE",
        )

    op.create_index("ix_model_aliases_name", "model_aliases", ["name"])
    op.create_index("ix_model_aliases_user_id", "model_aliases", ["user_id"])
    # A composite unique constraint cannot enforce this: both dialects treat NULL
    # user_ids as distinct, so it would allow two global rows with one name.
    op.create_index(
        "uq_model_aliases_global_name",
        "model_aliases",
        ["name"],
        unique=True,
        sqlite_where=sa.text("user_id IS NULL"),
        postgresql_where=sa.text("user_id IS NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    conn = op.get_bind()
    is_sqlite = conn.dialect.name == "sqlite"

    # User-scoped aliases have no representation in the old schema, and keeping
    # them would break the name primary key with duplicates.
    conn.execute(sa.text("DELETE FROM model_aliases WHERE user_id IS NOT NULL"))

    op.drop_index("uq_model_aliases_global_name", table_name="model_aliases")
    op.drop_index("ix_model_aliases_user_id", table_name="model_aliases")
    op.drop_index("ix_model_aliases_name", table_name="model_aliases")

    with op.batch_alter_table("model_aliases") as batch_op:
        if not is_sqlite:
            batch_op.drop_constraint("fk_model_aliases_user_id_users", type_="foreignkey")
            batch_op.drop_constraint("uq_model_aliases_name_user", type_="unique")
            batch_op.drop_constraint("model_aliases_pkey", type_="primary")
        batch_op.create_primary_key("model_aliases_pkey", ["name"])
        batch_op.drop_column("user_id")
        batch_op.drop_column("id")
