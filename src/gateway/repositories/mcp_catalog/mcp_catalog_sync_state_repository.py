"""Data access for the one MCP catalog sync state row. Flushes, never commits."""

from datetime import datetime
from typing import Never

from sqlalchemy import and_, false, or_, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.mcp_catalog import SYNC_STATE_ROW_ID, McpCatalogSyncState
from gateway.repositories.base_repository import BaseRepository


class McpCatalogSyncStateRepository(BaseRepository[McpCatalogSyncState, Never, Never]):
    """Repository for the ``mcp_catalog_sync_state`` row."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, McpCatalogSyncState)

    async def get_or_create(self) -> McpCatalogSyncState:
        """The state row, created with its defaults the first time it is read.

        Every replica's poll and a page load can find no row at once. The insert
        skips a row that is already there, so the one that loses the race inserts
        nothing and reads the winner's.
        """
        row = await self.db.get(McpCatalogSyncState, SYNC_STATE_ROW_ID, populate_existing=True)
        if row is not None:
            return row
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        await self.db.execute(
            insert(McpCatalogSyncState)
            .values(id=SYNC_STATE_ROW_ID)
            .on_conflict_do_nothing(index_elements=[McpCatalogSyncState.id])
        )
        row = await self.db.get(McpCatalogSyncState, SYNC_STATE_ROW_ID, populate_existing=True)
        if row is None:
            raise LookupError("the MCP catalog sync state row is missing after its insert")
        return row

    async def claim(self, now: datetime, *, scheduled: bool, stale_before: datetime, next_sync_at: datetime) -> bool:
        """Mark a sync as running if one is due and none is, answering whether this caller now owns it.

        Due means asked for, or scheduled and past ``next_sync_at``. A running row whose
        heartbeat is older than ``stale_before`` was left by a replica that died, and is
        taken over. One statement, so of two replicas polling at once only one sees its
        update match the row.
        """
        abandoned = and_(McpCatalogSyncState.running.is_(True), McpCatalogSyncState.heartbeat_at < stale_before)
        on_schedule = (
            or_(McpCatalogSyncState.next_sync_at.is_(None), McpCatalogSyncState.next_sync_at <= now)
            if scheduled
            else false()
        )
        due = and_(
            McpCatalogSyncState.running.is_(False), or_(McpCatalogSyncState.sync_requested.is_(True), on_schedule)
        )
        result = await self.db.execute(
            update(McpCatalogSyncState)
            .where(McpCatalogSyncState.id == SYNC_STATE_ROW_ID, or_(abandoned, due))
            .values(running=True, sync_requested=False, heartbeat_at=now, next_sync_at=next_sync_at)
            .returning(McpCatalogSyncState.id)
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one_or_none() is not None

    async def request(self, *, finished_at: datetime | None, stale_before: datetime) -> bool:
        """Ask for a sync if none is running or asked for and none ended since the caller read ``finished_at``.

        Answers whether the row matched. One statement, so a claim or a finish between the caller's read
        and this write leaves the row alone rather than queueing a second crawl.
        """
        not_running = or_(
            McpCatalogSyncState.running.is_(False),
            McpCatalogSyncState.heartbeat_at.is_(None),
            McpCatalogSyncState.heartbeat_at < stale_before,
        )
        result = await self.db.execute(
            update(McpCatalogSyncState)
            .where(
                McpCatalogSyncState.id == SYNC_STATE_ROW_ID,
                not_running,
                McpCatalogSyncState.sync_requested.is_(False),
                McpCatalogSyncState.finished_at.is_not_distinct_from(finished_at),
            )
            .values(sync_requested=True)
            .returning(McpCatalogSyncState.id)
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one_or_none() is not None

    async def hold(self, beat: datetime, now: datetime) -> bool:
        """Move a running sync's heartbeat from ``beat`` to ``now``, answering whether the caller still owns it.

        False once another replica has claimed the sync, which replaces the heartbeat the caller last wrote.
        """
        result = await self.db.execute(
            update(McpCatalogSyncState)
            .where(
                McpCatalogSyncState.id == SYNC_STATE_ROW_ID,
                McpCatalogSyncState.running.is_(True),
                McpCatalogSyncState.heartbeat_at == beat,
            )
            .values(heartbeat_at=now)
            .returning(McpCatalogSyncState.id)
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one_or_none() is not None
