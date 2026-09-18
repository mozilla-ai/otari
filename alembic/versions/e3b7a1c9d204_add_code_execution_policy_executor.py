"""Add the executor pin to the workspace code-execution policy.

One nullable column, ``executor``, on ``workspace_code_execution_policies``. It
names who runs a provider-native code-execution declaration for the workspace
(``auto``, ``otari`` or ``provider``), over the deployment's default and over
the request's own header.

No backfill: NULL is "no pin", the state every existing row is in, so a
deployment upgrading onto this revision keeps deciding exactly as it did.

Revision ID: e3b7a1c9d204
Revises: d5f8b2a4c6e9
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e3b7a1c9d204"
down_revision: str | Sequence[str] | None = "d5f8b2a4c6e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "workspace_code_execution_policies"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("executor", sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column(_TABLE, "executor")
