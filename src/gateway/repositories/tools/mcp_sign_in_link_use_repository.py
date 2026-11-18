"""Data access for ``mcp_sign_in_link_uses``: the sign-in links that have started their sign-in.

The only module that constructs or queries the table. Writes flush and leave
the commit to the caller.
"""

from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import McpSignInLinkUse
from gateway.repositories.base_repository import BaseRepository


class McpSignInLinkUseRepository(BaseRepository[McpSignInLinkUse, Never, Never]):
    """Claim a sign-in link once, and forget the claims of links that have expired anyway."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, McpSignInLinkUse)

    async def claim(self, ticket_hash: str, *, expires_at: datetime) -> bool:
        """Record a link as used, returning False when it already was.

        ``ON CONFLICT DO NOTHING`` rather than catching the unique violation: of two requests
        claiming one link at once, the database lets one insert, the other waits for it and then
        inserts nothing, and the loser's transaction stays usable.
        """
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        statement = (
            insert(McpSignInLinkUse)
            .values(ticket_hash=ticket_hash, expires_at=expires_at)
            .on_conflict_do_nothing(index_elements=[McpSignInLinkUse.ticket_hash])
        )
        result = cast("CursorResult[Any]", await self.db.execute(statement))
        await self.db.flush()
        return bool(result.rowcount)

    async def used(self, ticket_hash: str) -> bool:
        """Whether a link has started its sign-in already."""
        found = await self.db.execute(
            select(McpSignInLinkUse.ticket_hash).where(McpSignInLinkUse.ticket_hash == ticket_hash)
        )
        return found.scalar_one_or_none() is not None

    async def purge_expired(self, now: datetime) -> None:
        """Stage the removal of every claim whose link has expired by now."""
        await self.db.execute(delete(McpSignInLinkUse).where(McpSignInLinkUse.expires_at <= now))
