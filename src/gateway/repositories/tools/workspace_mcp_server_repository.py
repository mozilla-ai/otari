"""Data access for ``workspace_mcp_servers``.

New reads of the table go here. The tenancy and Playground services that read and write it directly predate this
module and are on the architecture check's baseline until they move.
"""

import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import WorkspaceMcpServer
from gateway.repositories.base_repository import BaseRepository


class WorkspaceMcpServerRepository(BaseRepository[WorkspaceMcpServer, Never, Never]):
    """Read stored MCP server rows."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, WorkspaceMcpServer)

    async def ids_in_workspace(self, workspace_id: uuid.UUID) -> list[uuid.UUID]:
        """The ids of a workspace's stored servers."""
        return list(
            (
                await self.db.execute(
                    select(WorkspaceMcpServer.id).where(WorkspaceMcpServer.workspace_id == workspace_id)
                )
            )
            .scalars()
            .all()
        )

    async def ids_in_workspaces(self, workspace_ids: Collection[uuid.UUID]) -> list[uuid.UUID]:
        """The ids of these workspaces' stored servers."""
        if not workspace_ids:
            return []
        return list(
            (
                await self.db.execute(
                    select(WorkspaceMcpServer.id).where(WorkspaceMcpServer.workspace_id.in_(list(workspace_ids)))
                )
            )
            .scalars()
            .all()
        )
