"""Add the sandbox image and exposed tool set to the workspace code-execution policy.

Two columns, both nullable, on ``workspace_code_execution_policies``:

* ``image`` names the sandbox image the workspace's code runs in, chosen from
  the allow-list an operator curated (``sandbox_allowed_session_images``). A workspace
  image is a supply-chain surface rather than a free string, which is why the
  service refuses one that is not on that list.
* ``tools`` names which code-execution tool kinds the policy exposes. A stored
  list narrows the set this deployment's sandbox backend would otherwise offer;
  it can never add one the backend does not serve.

No backfill, and both stay NULL for every existing row. NULL is the "no
narrowing" state that #725 established for the columns beside them, so a
deployment upgrading onto this revision keeps behaving exactly as it did.

Revision ID: a7f3c9e2b481
Revises: d8b3f1c6a4e9
Create Date: 2026-08-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7f3c9e2b481"
down_revision: str | Sequence[str] | None = "d8b3f1c6a4e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "workspace_code_execution_policies"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("image", sa.String(length=255), nullable=True))
    op.add_column(_TABLE, sa.Column("tools", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column(_TABLE, "tools")
    op.drop_column(_TABLE, "image")
