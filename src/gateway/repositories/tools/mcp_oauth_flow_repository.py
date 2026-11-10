"""Data access for ``mcp_oauth_flows``: sign-ins to a per-user MCP server that have not come back yet.

The only module that constructs or queries the table. Writes flush and leave
the commit to the caller.
"""

import uuid
from collections.abc import Collection
from datetime import datetime
from typing import Never

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import McpOAuthFlow
from gateway.repositories.base_repository import BaseRepository


class McpOAuthFlowRepository(BaseRepository[McpOAuthFlow, Never, Never]):
    """Read, stage and consume pending sign-ins."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, McpOAuthFlow)

    async def add(
        self,
        *,
        state_hash: str,
        server_id: uuid.UUID,
        user_id: str,
        identity_id: uuid.UUID | None,
        encrypted_code_verifier: str,
        client_id: str,
        client_auth_method: str,
        token_endpoint: str,
        revocation_endpoint: str | None,
        resource: str | None,
        redirect_uri: str,
        scope: str | None,
        issuer: str | None,
        requires_iss: bool,
        return_to: str | None,
        expires_at: datetime,
    ) -> None:
        """Stage a sign-in that is about to be sent to its authorization server."""
        self.db.add(
            McpOAuthFlow(
                state_hash=state_hash,
                server_id=server_id,
                user_id=user_id,
                identity_id=identity_id,
                encrypted_code_verifier=encrypted_code_verifier,
                client_id=client_id,
                client_auth_method=client_auth_method,
                token_endpoint=token_endpoint,
                revocation_endpoint=revocation_endpoint,
                resource=resource,
                redirect_uri=redirect_uri,
                scope=scope,
                issuer=issuer,
                requires_iss=requires_iss,
                return_to=return_to,
                expires_at=expires_at,
            )
        )
        await self.db.flush()

    async def peek(self, state_hash: str) -> McpOAuthFlow | None:
        """The pending sign-in a state names, left in place."""
        return (
            await self.db.execute(select(McpOAuthFlow).where(McpOAuthFlow.state_hash == state_hash))
        ).scalar_one_or_none()

    async def consume(self, state_hash: str) -> McpOAuthFlow | None:
        """Take the pending sign-in a state names, so that no second callback finds it.

        One ``DELETE ... RETURNING``: of two callbacks carrying the same state, the database hands
        the row to one.
        """
        taken = delete(McpOAuthFlow).where(McpOAuthFlow.state_hash == state_hash).returning(McpOAuthFlow)
        return (await self.db.execute(taken)).scalar_one_or_none()

    async def purge_expired(self, now: datetime) -> None:
        """Stage the removal of every sign-in past its deadline."""
        await self.db.execute(delete(McpOAuthFlow).where(McpOAuthFlow.expires_at <= now))

    async def remove_for_server(self, server_id: uuid.UUID) -> None:
        """Stage the removal of every pending sign-in to a server."""
        await self.db.execute(delete(McpOAuthFlow).where(McpOAuthFlow.server_id == server_id))

    async def remove_for_user(self, user_id: str, server_ids: Collection[uuid.UUID] | None = None) -> None:
        """Stage the removal of a user's pending sign-ins, on these servers or on every one."""
        query = delete(McpOAuthFlow).where(McpOAuthFlow.user_id == user_id)
        if server_ids is not None:
            if not server_ids:
                return
            query = query.where(McpOAuthFlow.server_id.in_(list(server_ids)))
        await self.db.execute(query)
