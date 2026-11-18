"""Where a workspace's stored MCP servers are, for a domain that lists servers the workspace could add."""

import uuid
from collections.abc import Awaitable, Callable

from gateway.repositories.tools import WorkspaceMcpServerRepository


class WorkspaceMcpServerUrls:
    """The URLs of the MCP servers a workspace reaches, its organization's too, read in the caller's Unit of Work."""

    def __init__(
        self,
        servers: WorkspaceMcpServerRepository,
        organization_of: Callable[[uuid.UUID], Awaitable[uuid.UUID | None]],
    ) -> None:
        """``organization_of`` names a workspace's organization, whose servers it reaches too."""
        self._servers = servers
        self._organization_of = organization_of

    async def urls_in_workspace(self, workspace_id: uuid.UUID) -> dict[uuid.UUID, str]:
        """Each server's URL the workspace reaches, by server id."""
        return await self._servers.urls_in_workspace(workspace_id, await self._organization_of(workspace_id))


__all__ = ["WorkspaceMcpServerUrls"]
