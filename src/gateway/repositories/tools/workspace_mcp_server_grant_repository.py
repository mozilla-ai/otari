"""Data access for ``workspace_mcp_server_grants``: a stored MCP server's exceptions for keys and users.

The only module that constructs or queries the table. Writes flush and leave
the commit to the caller.
"""

import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import WorkspaceMcpServerGrant
from gateway.repositories.base_repository import BaseRepository


class WorkspaceMcpServerGrantRepository(BaseRepository[WorkspaceMcpServerGrant, Never, Never]):
    """Read and stage grant rows."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, WorkspaceMcpServerGrant)

    async def for_caller(self, api_key_id: str, user_id: str | None) -> list[WorkspaceMcpServerGrant]:
        """Every grant naming this key or its owner, on any server."""
        subject = WorkspaceMcpServerGrant.api_key_id == api_key_id
        if user_id is not None:
            subject = or_(subject, WorkspaceMcpServerGrant.user_id == user_id)
        return list((await self.db.execute(select(WorkspaceMcpServerGrant).where(subject))).scalars().all())

    async def for_server(self, server_id: uuid.UUID) -> list[WorkspaceMcpServerGrant]:
        """A server's grants, oldest first."""
        return list(
            (
                await self.db.execute(
                    select(WorkspaceMcpServerGrant)
                    .where(WorkspaceMcpServerGrant.server_id == server_id)
                    .order_by(WorkspaceMcpServerGrant.created_at, WorkspaceMcpServerGrant.id)
                )
            )
            .scalars()
            .all()
        )

    async def count_for_server(self, server_id: uuid.UUID) -> int:
        """How many grants a server holds."""
        return int(
            (
                await self.db.execute(select(func.count()).where(WorkspaceMcpServerGrant.server_id == server_id))
            ).scalar_one()
        )

    async def get_on_server(self, server_id: uuid.UUID, grant_id: uuid.UUID) -> WorkspaceMcpServerGrant | None:
        """One grant of a server, or ``None`` when the id names none on it."""
        grant = await self.db.get(WorkspaceMcpServerGrant, grant_id)
        return grant if grant is not None and grant.server_id == server_id else None

    async def put(
        self,
        server_id: uuid.UUID,
        *,
        api_key_ids: Collection[str],
        user_ids: Collection[str],
        tools: list[str],
    ) -> None:
        """Stage one grant per key and per user with ``tools``, replacing the tools of any they already hold."""
        existing = {
            (grant.api_key_id, grant.user_id): grant
            for grant in await self.for_server(server_id)
            if grant.api_key_id in api_key_ids or grant.user_id in user_ids
        }
        subjects = [(key_id, None) for key_id in api_key_ids] + [(None, user_id) for user_id in user_ids]
        for subject in subjects:
            grant = existing.get(subject)
            if grant is None:
                self.db.add(
                    WorkspaceMcpServerGrant(server_id=server_id, api_key_id=subject[0], user_id=subject[1], tools=tools)
                )
            else:
                grant.tools = tools
        await self.db.flush()

    async def set_tools(self, grant: WorkspaceMcpServerGrant, tools: list[str]) -> None:
        """Stage new tools on a grant."""
        grant.tools = tools
        await self.db.flush()

    async def remove(self, grant: WorkspaceMcpServerGrant) -> None:
        """Stage a grant's removal."""
        await self.db.delete(grant)
        await self.db.flush()
