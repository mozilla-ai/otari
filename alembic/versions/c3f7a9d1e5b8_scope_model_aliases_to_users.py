"""Scope model aliases to users.

Adds a nullable ``user_id`` to ``model_aliases`` (NULL = global, the scope every
existing row keeps) and moves the primary key off ``name`` onto a surrogate
``id``, because the natural key is now (name, user_id) and a primary key cannot
contain a nullable column.

Both batch blocks pass ``copy_from`` rather than letting Alembic reflect the
table. Reflection on SQLite yields an unnamed primary key, so ``drop_constraint``
could not address it by name and the old key had to be left in place; the batch
recreate then swapped it silently and SQLAlchemy warned that it "may become an
exception in a future release". Describing the table explicitly names the
constraint, so the drop is a real, dialect-independent operation and the swap no
longer rests on behavior SQLAlchemy has reserved the right to turn into an error.

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

PK = "model_aliases_pkey"
UQ = "uq_model_aliases_name_user"
FK = "fk_model_aliases_user_id_users"


def _table(*, scoped: bool) -> sa.Table:
    """The table as it stands at a given point, for ``copy_from``.

    ``scoped`` selects the post-upgrade shape (surrogate key, ``user_id``, and
    its constraints); otherwise the pre-upgrade shape keyed on ``name``. Both
    include ``id``/``user_id`` as plain columns because the upgrade adds them
    before its batch block, so it can back-fill ``id`` while it is still
    nullable.
    """
    meta = sa.MetaData()
    columns = [
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("target", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.String(), nullable=not scoped),
        sa.Column("user_id", sa.String(), nullable=True),
    ]
    constraints: list[sa.schema.SchemaItem] = []
    if scoped:
        constraints = [
            sa.PrimaryKeyConstraint("id", name=PK),
            sa.UniqueConstraint("name", "user_id", name=UQ),
            sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], name=FK, ondelete="CASCADE"),
        ]
    else:
        constraints = [sa.PrimaryKeyConstraint("name", name=PK)]
    return sa.Table("model_aliases", meta, *columns, *constraints)


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()

    op.add_column("model_aliases", sa.Column("id", sa.String(), nullable=True))
    op.add_column("model_aliases", sa.Column("user_id", sa.String(), nullable=True))
    # Existing rows are global and their names are unique (name was the primary
    # key), so the name doubles as a back-filled surrogate id.
    conn.execute(sa.text("UPDATE model_aliases SET id = name WHERE id IS NULL"))

    with op.batch_alter_table("model_aliases", copy_from=_table(scoped=False), recreate="always") as batch_op:
        batch_op.alter_column("id", existing_type=sa.String(), nullable=False)
        batch_op.drop_constraint(PK, type_="primary")
        batch_op.create_primary_key(PK, ["id"])
        batch_op.create_unique_constraint(UQ, ["name", "user_id"])
        batch_op.create_foreign_key(FK, "users", ["user_id"], ["user_id"], ondelete="CASCADE")

    op.create_index("ix_model_aliases_user_id", "model_aliases", ["user_id"])
    # A composite unique constraint cannot enforce this: both dialects treat NULL
    # user_ids as distinct, so it would allow two global rows with one name.
    # No plain index on `name`: uq_model_aliases_name_user already leads with it.
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

    # User-scoped aliases have no representation in the old schema, and keeping
    # them would break the name primary key with duplicates.
    conn.execute(sa.text("DELETE FROM model_aliases WHERE user_id IS NOT NULL"))

    op.drop_index("uq_model_aliases_global_name", table_name="model_aliases")
    op.drop_index("ix_model_aliases_user_id", table_name="model_aliases")

    with op.batch_alter_table("model_aliases", copy_from=_table(scoped=True), recreate="always") as batch_op:
        batch_op.drop_constraint(FK, type_="foreignkey")
        batch_op.drop_constraint(UQ, type_="unique")
        batch_op.drop_constraint(PK, type_="primary")
        batch_op.create_primary_key(PK, ["name"])
        batch_op.drop_column("user_id")
        batch_op.drop_column("id")
