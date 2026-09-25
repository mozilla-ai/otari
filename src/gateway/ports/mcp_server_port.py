"""Where a deployment's MCP servers come from.

A workspace stores the servers it may reach.
One deployment holds those rows itself and another asks a peer that holds them for it.
Both answer the same questions, so a caller resolves servers without knowing which answered.

A caller connects to a resolved server directly.
Nothing here proxies MCP traffic.
"""

import uuid
from dataclasses import dataclass
from typing import Protocol

from gateway.exceptions.tools_exceptions import McpServerResolutionFailedError, WorkspaceMcpServerNotFoundError
from gateway.models.mcp import McpServerConfig, ResolvedMcpServer


@dataclass(frozen=True)
class McpServerScope:
    """Whose servers to resolve, in both the terms a deployment may keep them in.

    A workspace is named by its id where this deployment holds the rows, and by
    the caller's token where a peer holds them.
    A caller supplies both and chooses neither, so one is always unread.
    """

    workspace_id: uuid.UUID | None = None
    user_token: str | None = None


class McpServerPort(Protocol):
    """The MCP servers a workspace may reach."""

    async def resolve_many(self, scope: McpServerScope, server_ids: list[uuid.UUID]) -> list[McpServerConfig]:
        """The configs for ``server_ids``, de-duplicated with their order kept.

        This skips a disabled server rather than refusing it, so one
        decommissioned server does not break a caller whose list still names it.

        Raises:
            WorkspaceMcpServerNotFoundError: an id names no server this scope reaches.
            McpServerResolutionFailedError: the answer could not be read.
        """
        ...

    async def resolve_one(self, scope: McpServerScope, server_id: uuid.UUID) -> ResolvedMcpServer | None:
        """The one server ``server_id`` names, or ``None`` where it names none.

        A disabled server is returned carrying its state rather than omitted, so
        a caller can tell it apart from an id that reaches nothing.

        Raises:
            McpServerResolutionFailedError: the answer could not be read.
        """
        ...


__all__ = ["McpServerResolutionFailedError", "McpServerPort", "McpServerScope", "WorkspaceMcpServerNotFoundError"]
