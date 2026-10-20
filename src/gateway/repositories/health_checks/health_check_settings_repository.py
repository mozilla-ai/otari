"""Data access for the one health check settings row. Flushes, never commits."""

from datetime import datetime

from sqlalchemy import or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.health_checks import SETTINGS_ROW_ID, HealthCheckSettings
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.health_checks import HealthCheckSettingsUpdate


class HealthCheckSettingsRepository(
    BaseRepository[HealthCheckSettings, HealthCheckSettingsUpdate, HealthCheckSettingsUpdate]
):
    """Repository for the ``health_check_settings`` row."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        super().__init__(db, HealthCheckSettings)

    async def get_or_create(self) -> HealthCheckSettings:
        """The settings row, created with its defaults the first time it is read.

        Every replica's poll and a page load can find no row at once. The insert
        runs in a SAVEPOINT, so the one that loses the race rolls back its row
        alone and reads the winner's.
        """
        row = await self.db.get(HealthCheckSettings, SETTINGS_ROW_ID, populate_existing=True)
        if row is None:
            try:
                async with self.db.begin_nested():
                    self.db.add(HealthCheckSettings(id=SETTINGS_ROW_ID))
            except IntegrityError:
                pass
            row = await self.db.get(HealthCheckSettings, SETTINGS_ROW_ID, populate_existing=True)
            if row is None:
                raise LookupError("the health check settings row is missing after its insert")
            await self.db.refresh(row)
        return row

    async def claim_due_run(self, now: datetime, next_run_at: datetime) -> int | None:
        """Move ``next_run_at`` forward if a run is due, returning the run's number, or None.

        One statement, so of two replicas polling at once only one sees its update match the row.
        """
        result = await self.db.execute(
            update(HealthCheckSettings)
            .where(
                HealthCheckSettings.id == SETTINGS_ROW_ID,
                or_(
                    HealthCheckSettings.run_requested.is_(True),
                    HealthCheckSettings.schedule_enabled.is_(True)
                    & or_(HealthCheckSettings.next_run_at.is_(None), HealthCheckSettings.next_run_at <= now),
                ),
            )
            .values(
                next_run_at=next_run_at,
                run_requested=False,
                runs_started=HealthCheckSettings.runs_started + 1,
            )
            .returning(HealthCheckSettings.runs_started)
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one_or_none()
