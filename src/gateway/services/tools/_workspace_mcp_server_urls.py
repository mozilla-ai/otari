"""Where a workspace's stored MCP servers are, for a domain that lists servers the workspace could add."""

import uuid
from collections.abc import Awaitable, Callable

from gateway.repositories.tools import WorkspaceMcpServerRepository
from gateway.services.tools._mcp_workspace_servers import McpWorkspaceServers


class WorkspaceMcpServerUrls:
    """The URLs of the MCP servers a workspace reaches, its organization's too, read in the caller's Unit of Work.

    An organization's server turned off for the workspace is one it does not reach, as a request finds it.
    """

    def __init__(
        self,
        servers: WorkspaceMcpServerRepository,
        views: McpWorkspaceServers,
        organization_of: Callable[[uuid.UUID], Awaitable[uuid.UUID | None]],
    ) -> None:
        """``organization_of`` names a workspace's organization, whose servers it reaches too."""
        self._servers = servers
        self._views = views
        self._organization_of = organization_of

    async def urls_in_workspace(self, workspace_id: uuid.UUID) -> dict[uuid.UUID, str]:
        """Each server's URL the workspace reaches, by server id."""
        organization_id = await self._organization_of(workspace_id)
        if organization_id is None:
            return {}
        return {view.id: view.row.url for view in await self._views.every(workspace_id, organization_id)}

    async def has_url(self, workspace_id: uuid.UUID, url: str, organization_id: uuid.UUID) -> bool:
        """Whether the workspace reaches a server at ``url``, its path's trailing slashes and whitespace aside."""
        rows = await self._servers.at_url(workspace_id, url, organization_id)
        return bool(await self._views.seen_from(workspace_id, rows))


__all__ = ["WorkspaceMcpServerUrls"]
