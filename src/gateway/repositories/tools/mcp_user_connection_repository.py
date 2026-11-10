"""Data access for ``mcp_user_connections``: each user's own tokens for a per-user MCP server.

The only module that constructs or queries the table. Writes flush and leave
the commit to the caller.
"""

import uuid
from collections.abc import Collection
from datetime import datetime
from typing import Never

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import McpUserConnection
from gateway.repositories.base_repository import BaseRepository


class McpUserConnectionRepository(BaseRepository[McpUserConnection, Never, Never]):
    """Read and stage connection rows."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, McpUserConnection)

    async def find(self, server_id: uuid.UUID, user_id: str, *, lock: bool = False) -> McpUserConnection | None:
        """A user's connection to a server, or ``None``.

        The row is read as it is committed, not as this session loaded it earlier, and ``lock`` holds it
        until the caller commits, so a caller that waited on another's refresh sees the tokens it stored.
        """
        query = (
            select(McpUserConnection)
            .where(McpUserConnection.server_id == server_id, McpUserConnection.user_id == user_id)
            .execution_options(populate_existing=True)
        )
        if lock:
            query = query.with_for_update()
        return (await self.db.execute(query)).scalar_one_or_none()

    async def connected_server_ids(self, user_id: str, server_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        """Which of these servers the user has a connection to."""
        if not server_ids:
            return set()
        rows = await self.db.execute(
            select(McpUserConnection.server_id).where(
                McpUserConnection.user_id == user_id, McpUserConnection.server_id.in_(list(server_ids))
            )
        )
        return set(rows.scalars().all())

    async def for_user(self, user_id: str, server_ids: Collection[uuid.UUID]) -> list[McpUserConnection]:
        """A user's connections to these servers, oldest first."""
        if not server_ids:
            return []
        rows = await self.db.execute(
            select(McpUserConnection)
            .where(McpUserConnection.user_id == user_id, McpUserConnection.server_id.in_(list(server_ids)))
            .order_by(McpUserConnection.created_at, McpUserConnection.id)
        )
        return list(rows.scalars().all())

    async def for_server(self, server_id: uuid.UUID, *, skip: int, limit: int) -> list[McpUserConnection]:
        """A page of a server's connections, oldest first."""
        rows = await self.db.execute(
            select(McpUserConnection)
            .where(McpUserConnection.server_id == server_id)
            .order_by(McpUserConnection.created_at, McpUserConnection.id)
            .offset(skip)
            .limit(limit)
        )
        return list(rows.scalars().all())

    async def count_for_server(self, server_id: uuid.UUID) -> int:
        """How many users are connected to a server."""
        counted = select(func.count()).select_from(McpUserConnection).where(McpUserConnection.server_id == server_id)
        return (await self.db.execute(counted)).scalar_one()

    async def put(
        self,
        server_id: uuid.UUID,
        user_id: str,
        *,
        encrypted_access_token: str,
        encrypted_refresh_token: str | None,
        expires_at: datetime | None,
        scope: str | None,
        client_id: str,
        client_auth_method: str,
        token_endpoint: str,
        revocation_endpoint: str | None,
        resource: str | None,
    ) -> McpUserConnection:
        """Stage a user's connection to a server, replacing the tokens of one they already hold.

        Two sign-ins by one user can finish together, and the unique constraint refuses the second
        insert. The insert runs in a SAVEPOINT, so losing rolls back this insert alone, and the
        winner's row then takes these tokens: the later sign-in is the one the user finished last.
        """
        fields = {
            "encrypted_access_token": encrypted_access_token,
            "encrypted_refresh_token": encrypted_refresh_token,
            "expires_at": expires_at,
            "scope": scope,
            "client_id": client_id,
            "client_auth_method": client_auth_method,
            "token_endpoint": token_endpoint,
            "revocation_endpoint": revocation_endpoint,
            "resource": resource,
        }
        existing = await self.find(server_id, user_id, lock=True)
        if existing is None:
            try:
                async with self.db.begin_nested():
                    created = McpUserConnection(server_id=server_id, user_id=user_id, **fields)
                    self.db.add(created)
                    await self.db.flush()
                return created
            except IntegrityError:
                existing = await self.find(server_id, user_id, lock=True)
                if existing is None:
                    raise
        for field, value in fields.items():
            setattr(existing, field, value)
        await self.db.flush()
        return existing

    async def store_tokens(
        self,
        connection: McpUserConnection,
        *,
        encrypted_access_token: str,
        encrypted_refresh_token: str | None,
        expires_at: datetime | None,
        scope: str | None,
    ) -> None:
        """Stage the tokens a refresh returned on a connection."""
        connection.encrypted_access_token = encrypted_access_token
        connection.encrypted_refresh_token = encrypted_refresh_token
        connection.expires_at = expires_at
        connection.scope = scope
        await self.db.flush()

    async def remove(self, connection: McpUserConnection) -> None:
        """Stage one connection's removal."""
        await self.db.delete(connection)
        await self.db.flush()

    async def remove_for_user(self, user_id: str) -> None:
        """Stage the removal of every connection a user holds, on any server.

        A user is soft-deleted, so the foreign key's cascade never fires; a user recreated under the
        same ID must not find the deleted one's sign-ins waiting for it.
        """
        await self.db.execute(delete(McpUserConnection).where(McpUserConnection.user_id == user_id))

    async def remove_for_user_on_servers(self, user_id: str, server_ids: Collection[uuid.UUID]) -> None:
        """Stage the removal of a user's connections to these servers."""
        if not server_ids:
            return
        await self.db.execute(
            delete(McpUserConnection).where(
                McpUserConnection.user_id == user_id, McpUserConnection.server_id.in_(list(server_ids))
            )
        )

    async def remove_for_server(self, server_id: uuid.UUID) -> None:
        """Stage the removal of every connection to a server."""
        await self.db.execute(delete(McpUserConnection).where(McpUserConnection.server_id == server_id))
