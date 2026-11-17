"""Data access for ``mcp_logos``. Flushes, never commits."""

from collections.abc import Collection
from datetime import datetime
from typing import NamedTuple, Never

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.mcp_catalog import LOGO_MISSING, LOGO_OK, McpLogo
from gateway.repositories.base_repository import BaseRepository


class StoredLogoState(NamedTuple):
    """What a list needs of a stored logo: everything but the image."""

    status: str
    etag: str | None
    luminance: float | None
    retry_at: datetime


class McpLogoRepository(BaseRepository[McpLogo, Never, Never]):
    """Repository for the logos the gateway fetched."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, McpLogo)

    async def find(self, key: str) -> McpLogo | None:
        return await self.db.get(McpLogo, key, populate_existing=True)

    async def states(self, keys: Collection[str]) -> dict[str, StoredLogoState]:
        """The stored logos for ``keys``, by key, without their images."""
        if not keys:
            return {}
        rows = await self.db.execute(
            select(McpLogo.key, McpLogo.status, McpLogo.etag, McpLogo.luminance, McpLogo.retry_at).where(
                McpLogo.key.in_(set(keys))
            )
        )
        return {row.key: StoredLogoState(row.status, row.etag, row.luminance, row.retry_at) for row in rows}

    async def store_found(
        self,
        key: str,
        *,
        content: bytes,
        content_type: str,
        etag: str,
        luminance: float | None,
        source_url: str,
        fetched_at: datetime,
        retry_at: datetime,
    ) -> McpLogo:
        row = await self._row(key, fetched_at)
        row.status = LOGO_OK
        row.content = content
        row.content_type = content_type
        row.etag = etag
        row.luminance = luminance
        row.source_url = source_url
        row.fetched_at = fetched_at
        row.retry_at = retry_at
        await self.db.flush()
        return row

    async def store_not_found(self, key: str, *, fetched_at: datetime, retry_at: datetime) -> McpLogo:
        """Record that no logo was found. A logo found before is kept, and only looked for again later."""
        row = await self._row(key, fetched_at)
        row.retry_at = retry_at
        if row.status != LOGO_OK:
            row.fetched_at = fetched_at
        await self.db.flush()
        return row

    async def _row(self, key: str, fetched_at: datetime) -> McpLogo:
        """The row for ``key``, inserted as missing when there is none.

        Two replicas can look for one logo at once. The insert skips a key that is already
        there, so the one that loses the race inserts nothing and updates the winner's row.
        """
        row = await self.find(key)
        if row is not None:
            return row
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        await self.db.execute(
            insert(McpLogo)
            .values(key=key, status=LOGO_MISSING, fetched_at=fetched_at, retry_at=fetched_at)
            .on_conflict_do_nothing(index_elements=[McpLogo.key])
        )
        row = await self.find(key)
        if row is None:
            raise LookupError("the MCP logo row is missing after its insert")
        return row
