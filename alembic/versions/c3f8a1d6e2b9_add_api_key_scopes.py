"""Give each API key the scopes that say which endpoints it may reach.

Every existing key is given the scopes that cover what it could reach before
scopes existed, so no client breaks: all of them for a service key, and all but
the end-user scopes for any other key, which has no end users. The column's
default is no scope at all, so a row inserted without naming any reaches
nothing but the open catalog reads.

The list is written out here rather than imported, so this revision keeps
granting what it granted when it was written whatever the enum later holds.

Revision ID: c3f8a1d6e2b9
Revises: c6e8a0b2d4f1
Create Date: 2026-11-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3f8a1d6e2b9"
down_revision: str | Sequence[str] | None = "c6e8a0b2d4f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KEY_SCOPES = [
    "inference",
    "files:read",
    "files:write",
    "batches:read",
    "batches:write",
    "web",
    "code:execute",
    "mcp:read",
    "mcp:execute",
    "telemetry:write",
    "agents",
    "usage:read",
    "usage:write",
]
_SERVICE_KEY_SCOPES = [*_KEY_SCOPES, "end_users:read", "end_users:write", "end_users:delete"]


def upgrade() -> None:
    """Add the scopes column and grant each existing key what it could already reach."""
    op.add_column("api_keys", sa.Column("scopes", sa.JSON(), nullable=False, server_default="[]"))
    api_keys = sa.table("api_keys", sa.column("scopes", sa.JSON()), sa.column("is_service_key", sa.Boolean()))
    for is_service_key, scopes in ((False, _KEY_SCOPES), (True, _SERVICE_KEY_SCOPES)):
        op.execute(api_keys.update().where(api_keys.c.is_service_key.is_(is_service_key)).values(scopes=scopes))


def downgrade() -> None:
    """Drop the scopes column; every key reaches every endpoint it could before."""
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("scopes")
