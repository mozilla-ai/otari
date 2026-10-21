"""Data access for the one update check state row. Flushes, never commits."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.update_check import STATE_ROW_ID, UpdateCheckState
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.update_check import UpdateCheckAnswer


class UpdateCheckStateRepository(BaseRepository[UpdateCheckState, UpdateCheckAnswer, UpdateCheckAnswer]):
    """Repository for the ``update_check_state`` row."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        super().__init__(db, UpdateCheckState)

    async def get_or_create(self) -> UpdateCheckState:
        """The state row, created with a new install id the first time it is read.

        Every replica's poll and a page load can find no row at once. The insert
        runs in a SAVEPOINT, so the one that loses the race rolls back its row
        alone and reads the winner's, and the deployment keeps one install id.
        """
        row = await self.db.get(UpdateCheckState, STATE_ROW_ID, populate_existing=True)
        if row is None:
            try:
                async with self.db.begin_nested():
                    self.db.add(UpdateCheckState(id=STATE_ROW_ID))
            except IntegrityError:
                pass
            row = await self.db.get(UpdateCheckState, STATE_ROW_ID, populate_existing=True)
            if row is None:
                raise LookupError("the update check state row is missing after its insert")
            await self.db.refresh(row)
        return row

    async def schedule_if_unset(self, next_check_at: datetime) -> None:
        """Set the first check's time, unless a replica already has."""
        await self.db.execute(
            update(UpdateCheckState)
            .where(UpdateCheckState.id == STATE_ROW_ID, UpdateCheckState.next_check_at.is_(None))
            .values(next_check_at=next_check_at)
            .execution_options(synchronize_session=False)
        )

    async def claim_due_check(self, now: datetime, held_until: datetime) -> uuid.UUID | None:
        """Hold the row until ``held_until`` if a check is due, returning the install id, or None.

        One statement, so of two replicas polling at once only one sees its update match the row.
        The hold outlives a replica that dies mid-check, and the check's outcome replaces it.
        """
        result = await self.db.execute(
            update(UpdateCheckState)
            .where(UpdateCheckState.id == STATE_ROW_ID, UpdateCheckState.next_check_at <= now)
            .values(next_check_at=held_until)
            .returning(UpdateCheckState.install_id)
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def record_answer(self, row: UpdateCheckState, *, enabled: bool, now: datetime) -> UpdateCheckState:
        """Store an operator's answer. A yes is checked within a poll; a no forgets what the last check heard."""
        row.enabled = enabled
        row.answered_at = now
        row.failures = 0
        row.next_check_at = now if enabled else None
        if not enabled:
            row.latest_version = row.release_url = row.checked_version = None
            row.advisories = []
        await self.db.flush()
        return row

    async def record_reply(
        self,
        row: UpdateCheckState,
        *,
        now: datetime,
        version: str,
        latest_version: str | None,
        release_url: str | None,
        advisories: list[dict[str, Any]],
        next_check_at: datetime,
    ) -> None:
        """Store what a check heard, and when the next one is sent."""
        row.last_checked_at = now
        row.checked_version = version
        row.latest_version = latest_version
        row.release_url = release_url
        row.advisories = advisories
        row.failures = 0
        row.next_check_at = next_check_at
        await self.db.flush()

    async def record_failure(self, row: UpdateCheckState, *, next_check_at: datetime) -> None:
        """Count a failed check and set when it is retried. What the last good check heard is kept."""
        row.failures += 1
        row.next_check_at = next_check_at
        await self.db.flush()
