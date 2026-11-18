"""Let a workspace pin who runs a provider's own web search and web fetch keywords.

- ``workspace_web_search_configs.search_executor`` and ``fetch_executor``: NULL for
  every row, existing ones included, which keeps the deployment's interception and
  the request's headers deciding exactly as before. A stored value is one of
  ``CodeExecutor``'s, as ``workspace_code_execution_policies.executor`` is.

Revision ID: a3d5f7b9c1e4
Revises: 7c4e2a9b5d18
Create Date: 2026-11-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3d5f7b9c1e4"
down_revision: str | Sequence[str] | None = "7c4e2a9b5d18"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("workspace_web_search_configs", sa.Column("search_executor", sa.String(length=16), nullable=True))
    op.add_column("workspace_web_search_configs", sa.Column("fetch_executor", sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column("workspace_web_search_configs", "fetch_executor")
    op.drop_column("workspace_web_search_configs", "search_executor")
