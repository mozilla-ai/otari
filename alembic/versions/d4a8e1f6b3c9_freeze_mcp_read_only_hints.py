"""Keep the read-only hints of a workspace MCP server's first tool listing, so a later relabel cannot widen ``@read``.

``workspace_mcp_servers.read_tools_hinted`` holds the tools the server marked ``readOnlyHint`` the
first time its tools were listed, and nothing it says later changes it. Until a manager edits
``read_tools``, ``@read`` stands for this list rather than for the server's latest listing.

A server already listed takes the hints its stored health holds now. One never listed stays null
and takes them at its first listing.

Revision ID: d4a8e1f6b3c9
Revises: c7e2a4f9d1b8
Create Date: 2026-11-17
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4a8e1f6b3c9"
down_revision: str | Sequence[str] | None = "c7e2a4f9d1b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SERVERS = "workspace_mcp_servers"


def _listed_hints(health: object) -> list[str] | None:
    """The read-only hints of a stored health that holds a tool listing, else None."""
    if isinstance(health, str):
        try:
            health = json.loads(health)
        except ValueError:
            return None
    if not isinstance(health, dict) or health.get("status") != "healthy" or health.get("tool_count") is None:
        return None
    hinted = health.get("read_only_hinted_tools")
    return [name for name in hinted if isinstance(name, str)] if isinstance(hinted, list) else []


def upgrade() -> None:
    with op.batch_alter_table(_SERVERS) as batch:
        batch.add_column(sa.Column("read_tools_hinted", sa.JSON(none_as_null=True), nullable=True))
    servers = sa.table(
        _SERVERS,
        sa.column("id", sa.Uuid()),
        sa.column("health", sa.JSON()),
        sa.column("read_tools_hinted", sa.JSON(none_as_null=True)),
    )
    bind = op.get_bind()
    for server_id, health in bind.execute(sa.select(servers.c.id, servers.c.health)).all():
        hinted = _listed_hints(health)
        if hinted is not None:
            bind.execute(servers.update().where(servers.c.id == server_id).values(read_tools_hinted=hinted))


def downgrade() -> None:
    with op.batch_alter_table(_SERVERS) as batch:
        batch.drop_column("read_tools_hinted")
