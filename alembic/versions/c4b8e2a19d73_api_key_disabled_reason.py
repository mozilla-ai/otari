"""Replace the API key operator flag with the reason the key is disabled.

``operator_disabled`` recorded that an operator had disabled a key. Callers
need the reason the owner cannot enable it, which also makes the illegal
``is_active`` and ``operator_disabled`` pair unrepresentable.
``disabled_reason`` is NULL for a key its owner can enable, ``operator`` for a
reason only an operator can clear, and ``revoked`` for a credential no caller
can enable.

Every ``operator_disabled`` row becomes ``operator``. The downgrade sets the
flag for any reason, because every reason is one the owner cannot clear.

``batch_alter_table`` is used because SQLite cannot drop a column in place on
every version it ships; it recreates the table there and renders a plain
``ALTER TABLE DROP COLUMN`` on PostgreSQL.

Revision ID: c4b8e2a19d73
Revises: a7f2c91d4e60
Create Date: 2026-11-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4b8e2a19d73"
down_revision: str | Sequence[str] | None = "a7f2c91d4e60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "api_keys"
# Spelled here rather than imported, so a later change to the model's
# vocabulary cannot move what this revision writes.
_OPERATOR = "operator"
_api_keys = sa.table(
    _TABLE,
    sa.column("disabled_reason", sa.String()),
    sa.column("operator_disabled", sa.Boolean()),
)


def upgrade() -> None:
    """Add the reason column, carry each flagged key over as ``operator``, then drop the flag."""
    op.add_column(_TABLE, sa.Column("disabled_reason", sa.String(length=16), nullable=True))
    op.execute(_api_keys.update().where(_api_keys.c.operator_disabled.is_(True)).values(disabled_reason=_OPERATOR))
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.drop_column("operator_disabled")


def downgrade() -> None:
    """Add the flag, set it for a key disabled under any reason, then drop the reason column."""
    op.add_column(_TABLE, sa.Column("operator_disabled", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.execute(_api_keys.update().where(_api_keys.c.disabled_reason.is_not(None)).values(operator_disabled=True))
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.drop_column("disabled_reason")
