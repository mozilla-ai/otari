"""Where a workspace's stored MCP servers are, for a domain that lists servers the workspace could add."""

import uuid

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.tools import WorkspaceMcpServerRepository


class WorkspaceMcpServerUrls:
    """The URLs of a workspace's stored MCP servers, read inside the caller's Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        self._servers = WorkspaceMcpServerRepository(uow)

    async def urls_in_workspace(self, workspace_id: uuid.UUID) -> dict[uuid.UUID, str]:
        """Each of the workspace's stored servers' URL, by server id."""
        return await self._servers.urls_in_workspace(workspace_id)


__all__ = ["WorkspaceMcpServerUrls"]
