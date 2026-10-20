"""Data access for health check runs. Flushes, never commits."""

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.health_checks import HealthCheckRun
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.health_checks import HealthCheckRunPublic


class HealthCheckRunRepository(BaseRepository[HealthCheckRun, HealthCheckRunPublic, HealthCheckRunPublic]):
    """Repository for ``health_check_runs`` rows."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        super().__init__(db, HealthCheckRun)

    async def start(self, started_at: datetime, *, included_reasoning: bool) -> HealthCheckRun:
        """Stage a new, unfinished run, flushed so its id answers."""
        row = HealthCheckRun(started_at=started_at, included_reasoning=included_reasoning)
        self.db.add(row)
        await self.db.flush()
        return row

    async def latest_unfinished(self, since: datetime) -> HealthCheckRun | None:
        """The newest run started after ``since`` that has not finished."""
        result = await self.db.execute(
            select(HealthCheckRun)
            .where(HealthCheckRun.finished_at.is_(None), HealthCheckRun.started_at >= since)
            .order_by(HealthCheckRun.started_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def latest_finished_with_reasoning(self, since: datetime) -> HealthCheckRun | None:
        """The newest finished run started after ``since`` that included the reasoning models."""
        result = await self.db.execute(
            select(HealthCheckRun)
            .where(
                HealthCheckRun.finished_at.is_not(None),
                HealthCheckRun.included_reasoning.is_(True),
                HealthCheckRun.started_at >= since,
            )
            .order_by(HealthCheckRun.started_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def finished_since(self, since: datetime) -> Sequence[HealthCheckRun]:
        """Every finished run started after ``since``, oldest first."""
        result = await self.db.execute(
            select(HealthCheckRun)
            .where(HealthCheckRun.finished_at.is_not(None), HealthCheckRun.started_at >= since)
            .order_by(HealthCheckRun.started_at)
        )
        return result.scalars().all()

    async def latest_finished(self) -> HealthCheckRun | None:
        """The newest finished run, however old."""
        result = await self.db.execute(
            select(HealthCheckRun)
            .where(HealthCheckRun.finished_at.is_not(None))
            .order_by(HealthCheckRun.started_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def ids_started_before(self, cutoff: datetime) -> list[uuid.UUID]:
        """The ids of the runs started before ``cutoff``."""
        result = await self.db.execute(select(HealthCheckRun.id).where(HealthCheckRun.started_at < cutoff))
        return list(result.scalars().all())

    async def delete_ids(self, run_ids: Sequence[uuid.UUID]) -> None:
        """Remove these runs."""
        if run_ids:
            await self.db.execute(delete(HealthCheckRun).where(HealthCheckRun.id.in_(run_ids)))
