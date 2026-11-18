"""Data access for ``mcp_server_aliases``: the ids of folded servers, and the server each now names.

The only module that constructs or queries the table. Writes flush and leave
the commit to the caller.
"""

import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import McpServerAlias
from gateway.repositories.base_repository import BaseRepository


class McpServerAliasRepository(BaseRepository[McpServerAlias, Never, Never]):
    """Record an old id, and read which server old ids name in a workspace."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, McpServerAlias)

    async def add(self, old_id: uuid.UUID, *, workspace_id: uuid.UUID, server_id: uuid.UUID) -> None:
        """Stage ``old_id`` naming ``server_id`` for callers in ``workspace_id``."""
        self.db.add(McpServerAlias(old_id=old_id, workspace_id=workspace_id, server_id=server_id))
        await self.db.flush()

    async def targets(self, workspace_id: uuid.UUID, ids: Collection[uuid.UUID]) -> dict[uuid.UUID, uuid.UUID]:
        """The server each of these ids now names in a workspace, by old id; an id that is no alias there is absent."""
        if not ids:
            return {}
        rows = await self.db.execute(
            select(McpServerAlias.old_id, McpServerAlias.server_id).where(
                McpServerAlias.workspace_id == workspace_id, McpServerAlias.old_id.in_(list(ids))
            )
        )
        return {row.old_id: row.server_id for row in rows}
