"""Data access for ``mcp_catalog_servers``. Flushes, never commits."""

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Never

from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.sql import LIKE_ESCAPE, contains_pattern
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.mcp_catalog import McpCatalogServer
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.mcp_catalog import McpCatalogEntry


def _is_outage(exc: Exception) -> bool:
    """Whether ``exc`` is the database being unreachable, as opposed to it refusing a row."""
    return isinstance(exc, OperationalError | InterfaceError | TimeoutError)


class McpCatalogServerRepository(BaseRepository[McpCatalogServer, McpCatalogEntry, Never]):
    """Repository for the catalog's servers."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, McpCatalogServer)

    async def search(self, query: str, *, skip: int, limit: int) -> tuple[Sequence[McpCatalogServer], int]:
        """A page of the servers whose name, title or description contains ``query``, and how many match in all.

        A title that starts with the query comes first, then a title that contains it, then a match in
        the name or description alone; by title within each. An empty query matches every server.
        """
        statement = select(McpCatalogServer)
        count = select(func.count()).select_from(McpCatalogServer)
        order = [func.lower(McpCatalogServer.title), McpCatalogServer.name]
        term = query.strip()
        if term:
            pattern = contains_pattern(term)
            title = func.lower(McpCatalogServer.title)
            in_title = title.like(pattern, escape=LIKE_ESCAPE)
            matches = or_(
                in_title,
                func.lower(McpCatalogServer.name).like(pattern, escape=LIKE_ESCAPE),
                func.lower(func.coalesce(McpCatalogServer.description, "")).like(pattern, escape=LIKE_ESCAPE),
            )
            statement = statement.where(matches)
            count = count.where(matches)
            starts_title = title.like(pattern.removeprefix("%"), escape=LIKE_ESCAPE)
            order.insert(0, case((starts_title, 0), (in_title, 1), else_=2))
        rows = (await self.db.execute(statement.order_by(*order).offset(skip).limit(limit))).scalars().all()
        return rows, (await self.db.execute(count)).scalar_one()

    async def listed_by_name(self, names: Sequence[str]) -> dict[str, tuple[uuid.UUID, str]]:
        """The id and URL the catalog holds today for each of ``names`` it still lists."""
        if not names:
            return {}
        result = await self.db.execute(
            select(McpCatalogServer.name, McpCatalogServer.id, McpCatalogServer.url).where(
                McpCatalogServer.name.in_(names)
            )
        )
        return {name: (server_id, url) for name, server_id, url in result.all()}

    async def upsert(self, entries: Sequence[McpCatalogEntry], *, seen_at: datetime) -> int:
        """Store ``entries`` by name, each marked as seen by the sync that began at ``seen_at``.

        Returns how many the database refused. The entries are read from a public
        listing, so one the database will not take (a value its encoding cannot
        hold) is left out and the rest are stored: the page is tried whole in a
        SAVEPOINT, then entry by entry. A failure to reach the database is raised.
        """
        if not entries:
            return 0
        try:
            async with self.db.begin_nested():
                await self._upsert(entries, seen_at)
        except (SQLAlchemyError, ValueError) as exc:
            if _is_outage(exc):
                raise
        else:
            return 0
        refused = 0
        for entry in entries:
            try:
                async with self.db.begin_nested():
                    await self._upsert([entry], seen_at)
            except (SQLAlchemyError, ValueError) as exc:
                if _is_outage(exc):
                    raise
                refused += 1
        return refused

    async def _upsert(self, entries: Sequence[McpCatalogEntry], seen_at: datetime) -> None:
        names = [entry.name for entry in entries]
        existing = {
            row.name: row
            for row in (await self.db.execute(select(McpCatalogServer).where(McpCatalogServer.name.in_(names))))
            .scalars()
            .all()
        }
        for entry in entries:
            values = entry.model_dump()
            row = existing.get(entry.name)
            if row is None:
                row = McpCatalogServer(**values, last_seen_at=seen_at)
                existing[entry.name] = row
                self.db.add(row)
                continue
            for field, value in values.items():
                setattr(row, field, value)
            row.last_seen_at = seen_at
        await self.db.flush()

    async def count_unseen(self, before: datetime) -> int:
        """How many servers no sync has listed since ``before``."""
        result = await self.db.execute(
            select(func.count()).select_from(McpCatalogServer).where(McpCatalogServer.last_seen_at < before)
        )
        return result.scalar_one()

    async def delete_unseen(self, before: datetime) -> int:
        """Delete the servers no sync has listed since ``before``. Returns how many."""
        result = await self.db.execute(
            delete(McpCatalogServer)
            .where(McpCatalogServer.last_seen_at < before)
            .execution_options(synchronize_session=False)
        )
        return int(getattr(result, "rowcount", 0) or 0)
