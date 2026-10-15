"""Let a service key mint keys bound to one of its end users.

Adds ``api_keys.parent_key_id``, set on a key a service key minted, with
``ON DELETE CASCADE`` so the minted keys go with the service key;
``api_keys.max_child_ttl_sec``, the longest such a key may live, null on every
existing key so none of them mints; and ``api_keys.tags``, written on every
usage row a key writes.

The foreign key is added inline on SQLite, which has ``ADD COLUMN ...
REFERENCES`` but no ``ADD CONSTRAINT``, so ``api_keys`` is not rebuilt: a
rebuild reflects it, and its partial index is what reflection loses.

Revision ID: c9e1a3f5b7d2
Revises: b7d9f1a3c5e8
Create Date: 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c9e1a3f5b7d2"
down_revision: str | Sequence[str] | None = "b7d9f1a3c5e8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PARENT_FK = "fk_api_keys_parent_key_id"
_PARENT_INDEX = "ix_api_keys_parent_key_id"


def upgrade() -> None:
    """Add the issuing key, the mint limit and the key's tags."""
    if op.get_bind().dialect.name == "sqlite":
        op.execute(
            "ALTER TABLE api_keys ADD COLUMN parent_key_id VARCHAR "
            f"CONSTRAINT {_PARENT_FK} REFERENCES api_keys (id) ON DELETE CASCADE"
        )
    else:
        op.add_column("api_keys", sa.Column("parent_key_id", sa.String(), nullable=True))
        op.create_foreign_key(_PARENT_FK, "api_keys", "api_keys", ["parent_key_id"], ["id"], ondelete="CASCADE")
    op.create_index(_PARENT_INDEX, "api_keys", ["parent_key_id"], unique=False)
    op.add_column("api_keys", sa.Column("max_child_ttl_sec", sa.Integer(), nullable=True))
    op.add_column("api_keys", sa.Column("tags", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Drop the minted keys, then the three columns.

    A minted key cannot outlive the column that marks it as one: left in place it
    would authenticate as an ordinary key of its end user, with no expiry bound to
    a service key any more.
    """
    sqlite = op.get_bind().dialect.name == "sqlite"
    op.execute("DELETE FROM api_keys WHERE parent_key_id IS NOT NULL")
    op.drop_index(_PARENT_INDEX, table_name="api_keys")
    with op.batch_alter_table("api_keys") as batch:
        if not sqlite:
            batch.drop_constraint(_PARENT_FK, type_="foreignkey")
        batch.drop_column("tags")
        batch.drop_column("max_child_ttl_sec")
        batch.drop_column("parent_key_id")
