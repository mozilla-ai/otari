"""Data access for ``workspace_mcp_servers``.

New reads of the table go here. The tenancy and Playground services that read and write it directly predate this
module and are on the architecture check's baseline until they move.
"""

import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import ColumnElement, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import WorkspaceMcpServer
from gateway.repositories.base_repository import BaseRepository


class WorkspaceMcpServerRepository(BaseRepository[WorkspaceMcpServer, Never, Never]):
    """Read stored MCP server rows."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, WorkspaceMcpServer)

    async def find(self, server_id: uuid.UUID, *, lock: bool = False) -> WorkspaceMcpServer | None:
        """One stored server by id, or ``None``; ``lock`` holds the row until the caller commits."""
        query = select(WorkspaceMcpServer).where(WorkspaceMcpServer.id == server_id)
        if lock:
            query = query.with_for_update().execution_options(populate_existing=True)
        return (await self.db.execute(query)).scalar_one_or_none()

    async def find_many(self, server_ids: Collection[uuid.UUID]) -> list[WorkspaceMcpServer]:
        """The stored servers these ids name."""
        if not server_ids:
            return []
        rows = await self.db.execute(select(WorkspaceMcpServer).where(WorkspaceMcpServer.id.in_(list(server_ids))))
        return list(rows.scalars().all())

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

    async def for_request(
        self, workspace_id: uuid.UUID, server_ids: Collection[uuid.UUID], *, attached: bool
    ) -> list[WorkspaceMcpServer]:
        """A workspace's servers with these ids, and with ``attached`` every one attached by default, oldest first."""
        wanted: ColumnElement[bool] = WorkspaceMcpServer.id.in_(list(server_ids))
        if attached:
            wanted = or_(wanted, WorkspaceMcpServer.attach_by_default.is_(True))
        query = (
            select(WorkspaceMcpServer)
            .where(WorkspaceMcpServer.workspace_id == workspace_id, wanted)
            .order_by(WorkspaceMcpServer.created_at, WorkspaceMcpServer.id)
        )
        return list((await self.db.execute(query)).scalars().all())

    async def any_attached(self, workspace_id: uuid.UUID) -> bool:
        """Whether a workspace has an enabled server attached by default."""
        found = await self.db.execute(
            select(WorkspaceMcpServer.id)
            .where(
                WorkspaceMcpServer.workspace_id == workspace_id,
                WorkspaceMcpServer.attach_by_default.is_(True),
                WorkspaceMcpServer.enabled.is_(True),
            )
            .limit(1)
        )
        return found.first() is not None

    async def has_url(self, workspace_id: uuid.UUID, url: str) -> bool:
        """Whether a workspace stores a server at ``url``, a trailing slash and surrounding spaces aside."""
        bare = url.strip().rstrip("/")
        rows = await self.db.execute(
            select(WorkspaceMcpServer.id)
            .where(
                WorkspaceMcpServer.workspace_id == workspace_id,
                func.rtrim(func.trim(WorkspaceMcpServer.url), "/") == bare,
            )
            .limit(1)
        )
        return rows.first() is not None

    async def urls_in_workspace(self, workspace_id: uuid.UUID) -> dict[uuid.UUID, str]:
        """Each of a workspace's stored servers' URL, by server id."""
        rows = await self.db.execute(
            select(WorkspaceMcpServer.id, WorkspaceMcpServer.url).where(WorkspaceMcpServer.workspace_id == workspace_id)
        )
        return {row.id: row.url for row in rows}
