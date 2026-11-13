"""Data access for ``workspace_mcp_server_logos``. Flushes, never commits."""

import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import WorkspaceMcpServerLogo
from gateway.repositories.base_repository import BaseRepository


class WorkspaceMcpServerLogoRepository(BaseRepository[WorkspaceMcpServerLogo, Never, Never]):
    """How a workspace shows its MCP servers: an admin's uploaded logo and tile choice."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, WorkspaceMcpServerLogo)

    async def find(self, server_id: uuid.UUID) -> WorkspaceMcpServerLogo | None:
        return await self.db.get(WorkspaceMcpServerLogo, server_id, populate_existing=True)

    async def for_servers(self, server_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, WorkspaceMcpServerLogo]:
        if not server_ids:
            return {}
        rows = await self.db.execute(
            select(WorkspaceMcpServerLogo).where(WorkspaceMcpServerLogo.server_id.in_(list(server_ids)))
        )
        return {row.server_id: row for row in rows.scalars().all()}

    async def save(
        self,
        server_id: uuid.UUID,
        *,
        tile: str,
        content: bytes | None,
        etag: str | None,
        luminance: float | None,
        replace_content: bool,
    ) -> WorkspaceMcpServerLogo:
        """Set the tile, and the uploaded image too when ``replace_content``. The row is created on first use."""
        row = await self.find(server_id)
        if row is None:
            row = WorkspaceMcpServerLogo(server_id=server_id)
            self.db.add(row)
        row.tile = tile
        if replace_content:
            row.content = content
            row.etag = etag
            row.luminance = luminance
        await self.db.flush()
        return row
