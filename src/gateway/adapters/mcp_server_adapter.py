"""The two places a deployment's MCP servers come from.

``LocalMcpServers`` reads the rows this deployment holds.
``RemoteMcpServers`` asks a peer, speaking `docs/hybrid-mode-protocol.md`.

Both return a server a caller connects to directly.
Neither proxies MCP traffic.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.deployment import Plane, deployment_for
from gateway.exceptions.tools_exceptions import McpServerResolutionFailedError
from gateway.models.mcp import McpServerConfig, ResolvedMcpServer
from gateway.ports.mcp_server_port import McpServerPort, McpServerScope
from gateway.services.control_plane import ResolveEndpoint, resolve
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
            raise McpServerResolutionFailedError
        return await resolve_workspace_mcp_servers(
            self._session, workspace_id=scope.workspace_id, server_ids=server_ids
        )

    async def resolve_one(self, scope: McpServerScope, server_id: uuid.UUID) -> ResolvedMcpServer | None:
        if scope.workspace_id is None:
            raise McpServerResolutionFailedError
        return await resolve_workspace_mcp_server(self._session, workspace_id=scope.workspace_id, server_id=server_id)


class RemoteMcpServers(McpServerPort):
    """The servers the control plane holds for this deployment's workspaces."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    async def _ask(self, scope: McpServerScope, server_ids: list[uuid.UUID]) -> list[Any]:
        """The peer's ``servers`` list.

        IDs are de-duplicated with their order kept, because the protocol does not define a repeated ID.
        An empty list resolves to no servers, and an answer omitting the key is refused.
        """
        if not scope.user_token:
            raise McpServerResolutionFailedError
        payload = await resolve(
            self._config,
            user_token=scope.user_token,
            endpoint=ResolveEndpoint.MCP_SERVERS,
            body={"mcp_server_ids": [str(uid) for uid in dict.fromkeys(server_ids)]},
        )
        if not isinstance(payload, dict):
            raise McpServerResolutionFailedError
        servers = payload.get("servers")
        if not isinstance(servers, list):
            raise McpServerResolutionFailedError
        return servers

    async def resolve_many(self, scope: McpServerScope, server_ids: list[uuid.UUID]) -> list[McpServerConfig]:
        entries = await self._ask(scope, server_ids)
        try:
            return [
                McpServerConfig(
                    name=entry["name"],
                    url=entry["url"],
                    authorization_token=entry.get("authorization_token"),
                    purpose_hint=entry.get("purpose_hint"),
                    allowed_tools=entry.get("allowed_tools"),
                )
                for entry in entries
            ]
        except (ValidationError, KeyError, TypeError):
            # The underlying error quotes the answer, which carries the stored
            # URL and credential, so it reaches neither the caller nor the log.
            raise McpServerResolutionFailedError from None

    async def resolve_one(self, scope: McpServerScope, server_id: uuid.UUID) -> ResolvedMcpServer | None:
        servers = await self._ask(scope, [server_id])
        if not servers:
            # An older peer omits a disabled server rather than reporting one.
            # Both mean the caller cannot reach it.
            return None
        if len(servers) != 1:
            raise McpServerResolutionFailedError

        entry = servers[0]
        if isinstance(entry, dict):
            entry = dict(entry)
            entry.setdefault("id", server_id)
            entry.setdefault("enabled", True)
        try:
            resolved = ResolvedMcpServer.model_validate(entry)
        except ValidationError:
            # The validation error quotes the answer, which carries the stored
            # URL and credential.
            raise McpServerResolutionFailedError from None
        if resolved.id != server_id:
            raise McpServerResolutionFailedError
        return resolved


def build_mcp_server_port(config: GatewayConfig, session: AsyncSession | None) -> McpServerPort:
    """The implementation for the planes this deployment serves.

    A deployment serving a control plane holds the rows and needs a session.
    One that does not has a peer holding them, and needs none.
    """
    if deployment_for(config).supports(Plane.CONTROL):
        if session is None:
            msg = "a session is required when this deployment holds its own MCP server rows"
            raise ValueError(msg)
        return LocalMcpServers(session)
    return RemoteMcpServers(config)
