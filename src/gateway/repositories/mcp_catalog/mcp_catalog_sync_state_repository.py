"""Data access for the one MCP catalog sync state row. Flushes, never commits."""

from datetime import datetime
from typing import Never

from sqlalchemy import and_, false, or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

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
        runs in a SAVEPOINT, so the one that loses the race rolls back its row
        alone and reads the winner's.
        """
        row = await self.db.get(McpCatalogSyncState, SYNC_STATE_ROW_ID, populate_existing=True)
        if row is None:
            try:
                async with self.db.begin_nested():
                    self.db.add(McpCatalogSyncState(id=SYNC_STATE_ROW_ID))
            except IntegrityError:
                pass
            row = await self.db.get(McpCatalogSyncState, SYNC_STATE_ROW_ID, populate_existing=True)
            if row is None:
                raise LookupError("the MCP catalog sync state row is missing after its insert")
            await self.db.refresh(row)
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
