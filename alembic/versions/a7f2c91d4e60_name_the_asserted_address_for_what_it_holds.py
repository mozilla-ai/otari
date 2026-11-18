"""Name the provider-asserted address columns for what they now hold.

``email_at_link`` and ``email_assertion_at_link`` were written once, when the link was made.
They are now taken again on every sign-in, because the deployment's email blocklist matches an
account through them and a stale address would match the wrong one. The names outlived that.

``batch_alter_table`` is used because SQLite cannot rename a column in place on every version
it ships; it recreates the table there and renders a plain ``ALTER COLUMN`` on PostgreSQL.

Revision ID: a7f2c91d4e60
Revises: 4eee3a7ee90f
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7f2c91d4e60"
down_revision: str | Sequence[str] | None = "4eee3a7ee90f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "oauth_identity_link"
_RENAMES = (
    ("email_at_link", "asserted_email", sa.String(length=255)),
    ("email_assertion_at_link", "email_assertion", sa.String(length=16)),
)


def upgrade() -> None:
    with op.batch_alter_table(_TABLE) as batch_op:
        for old, new, kind in _RENAMES:
            batch_op.alter_column(old, new_column_name=new, existing_type=kind)


def downgrade() -> None:
    with op.batch_alter_table(_TABLE) as batch_op:
        for old, new, kind in _RENAMES:
            batch_op.alter_column(new, new_column_name=old, existing_type=kind)
