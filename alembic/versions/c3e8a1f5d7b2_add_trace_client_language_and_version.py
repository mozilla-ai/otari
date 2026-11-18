"""Add the client's language and version to each trace.

Both are read from the request headers that open a session, beside ``harness``:
the language from an SDK's own header or an HTTP library's ``User-Agent``, the
version from the ``User-Agent``. Nullable, and left empty on existing traces.

Revision ID: c3e8a1f5d7b2
Revises: b7e2c9d4f1a6
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3e8a1f5d7b2"
down_revision: str | Sequence[str] | None = "b7e2c9d4f1a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("traces") as batch_op:
        batch_op.add_column(sa.Column("client_language", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("client_version", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("traces") as batch_op:
        batch_op.drop_column("client_version")
        batch_op.drop_column("client_language")
