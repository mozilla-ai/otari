"""Where a deployment's MCP servers come from: the rows this deployment holds.

A caller connects to a returned server directly; nothing here proxies MCP traffic.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.exceptions.tools_exceptions import McpResolutionFailure, McpServerResolutionFailedError
from gateway.models.mcp import McpServerConfig, ResolvedMcpServer
from gateway.ports.mcp_server_port import McpServerPort, McpServerScope
from gateway.services.tenancy.workspace_mcp_server_service import (
    resolve_workspace_mcp_server,
    resolve_workspace_mcp_servers,
)


class LocalMcpServers(McpServerPort):
    """The servers stored in this deployment's own database."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def resolve_many(self, scope: McpServerScope, server_ids: list[uuid.UUID]) -> list[McpServerConfig]:
        if scope.workspace_id is None:
            raise McpServerResolutionFailedError(McpResolutionFailure.NO_WORKSPACE)
        return await resolve_workspace_mcp_servers(
            self._session, workspace_id=scope.workspace_id, server_ids=server_ids
        )

    async def resolve_one(self, scope: McpServerScope, server_id: uuid.UUID) -> ResolvedMcpServer | None:
        if scope.workspace_id is None:
            raise McpServerResolutionFailedError(McpResolutionFailure.NO_WORKSPACE)
        return await resolve_workspace_mcp_server(self._session, workspace_id=scope.workspace_id, server_id=server_id)
