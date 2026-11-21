"""Data access for ``workspace_mcp_server_overrides``: each workspace's settings on an organization's MCP servers.

The only module that constructs or queries the table. Writes flush and leave
the commit to the caller.
"""

import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import WorkspaceMcpServer, WorkspaceMcpServerOverride
from gateway.repositories.base_repository import BaseRepository


class WorkspaceMcpServerOverrideRepository(BaseRepository[WorkspaceMcpServerOverride, Never, Never]):
    """Read and stage override rows, always addressed by workspace and server."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, WorkspaceMcpServerOverride)

    async def find(self, *, workspace_id: uuid.UUID, server_id: uuid.UUID) -> WorkspaceMcpServerOverride | None:
        """A workspace's override of one server, or ``None`` where it inherits."""
        found = await self.db.execute(
            select(WorkspaceMcpServerOverride).where(
                WorkspaceMcpServerOverride.workspace_id == workspace_id,
                WorkspaceMcpServerOverride.server_id == server_id,
            )
        )
        return found.scalar_one_or_none()

    async def for_workspace(
        self, workspace_id: uuid.UUID, server_ids: Collection[uuid.UUID]
    ) -> dict[uuid.UUID, WorkspaceMcpServerOverride]:
        """A workspace's overrides of these servers, by server id; a server it inherits has none."""
        if not server_ids:
            return {}
        rows = await self.db.execute(
            select(WorkspaceMcpServerOverride).where(
                WorkspaceMcpServerOverride.workspace_id == workspace_id,
                WorkspaceMcpServerOverride.server_id.in_(list(server_ids)),
            )
        )
        return {override.server_id: override for override in rows.scalars().all()}

    async def for_server(self, server_id: uuid.UUID) -> list[WorkspaceMcpServerOverride]:
        """Every workspace's override of one server, oldest first."""
        rows = await self.db.execute(
            select(WorkspaceMcpServerOverride)
            .where(WorkspaceMcpServerOverride.server_id == server_id)
            .order_by(WorkspaceMcpServerOverride.created_at, WorkspaceMcpServerOverride.id)
        )
        return list(rows.scalars().all())

    async def attaching_workspace_ids(self) -> frozenset[uuid.UUID]:
        """Every workspace whose own settings attach one of its organization's servers to every request."""
        found = await self.db.execute(
            select(WorkspaceMcpServerOverride.workspace_id)
            .where(WorkspaceMcpServerOverride.attach_by_default.is_(True))
            .distinct()
        )
        return frozenset(found.scalars())

    async def save(
        self,
        override: WorkspaceMcpServerOverride | None,
        *,
        workspace_id: uuid.UUID,
        server: WorkspaceMcpServer,
        organization_tools: list[str] | None,
        enabled: bool,
        everyone_tools: list[str] | None,
        attach_by_default: bool | None,
    ) -> WorkspaceMcpServerOverride | None:
        """Stage a workspace's settings on an organization's ``server``: a new override, or ``override`` changed.

        Settings that are all the defaults are no override: the one passed in is removed, and ``None`` returned.
        The keys refuse a workspace of another organization than the server's.

        Raises:
            ValueError: ``server`` is one workspace's own, which no other workspace reaches, or ``override`` is
                another workspace's or another server's.
        """
        if server.workspace_id is not None:
            raise ValueError(f"MCP server {server.id} is a workspace's own; only an organization's takes overrides")
        if override is not None and (override.workspace_id != workspace_id or override.server_id != server.id):
            raise ValueError("The override passed in is another workspace's or another server's")
        if organization_tools is None and enabled and everyone_tools is None and attach_by_default is None:
            if override is not None:
                await self.remove(override)
            return None
        if override is None:
            override = WorkspaceMcpServerOverride(
                workspace_id=workspace_id, organization_id=server.organization_id, server_id=server.id
            )
            self.db.add(override)
        override.organization_tools = organization_tools
        override.enabled = enabled
        override.everyone_tools = everyone_tools
        override.attach_by_default = attach_by_default
        await self.db.flush()
        return override

    async def remove(self, override: WorkspaceMcpServerOverride) -> None:
        """Stage an override's removal."""
        await self.db.delete(override)
        await self.db.flush()
