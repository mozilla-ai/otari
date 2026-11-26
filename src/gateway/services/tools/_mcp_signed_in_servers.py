"""Which per-user MCP servers a user has signed in to, for a domain that lists servers a user may pick."""

import uuid
from collections.abc import Collection

from gateway.repositories.tools import McpUserConnectionRepository


class McpSignedInServers:
    """Reads a user's connections to per-user servers."""

    def __init__(self, connections: McpUserConnectionRepository) -> None:
        self._connections = connections

    async def signed_in(self, user_id: str, server_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        """Which of these servers ``user_id`` holds a connection to."""
        return await self._connections.connected_server_ids(user_id, server_ids)


__all__ = ["McpSignedInServers"]
