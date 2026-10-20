"""Data access for health check recoveries. Flushes, never commits."""

from collections.abc import Iterable, Sequence
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.health_checks import HealthCheckRecovery
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.health_checks import HealthCheckRecoveryCreate


class HealthCheckRecoveryRepository(
    BaseRepository[HealthCheckRecovery, HealthCheckRecoveryCreate, HealthCheckRecoveryCreate]
):
    """Repository for ``health_check_recoveries`` rows, which only a run writes."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        super().__init__(db, HealthCheckRecovery)

    async def add_all(self, recoveries: Iterable[HealthCheckRecoveryCreate]) -> None:
        """Stage the outages one run ended."""
        self.db.add_all(HealthCheckRecovery(**recovery.model_dump()) for recovery in recoveries)
        await self.db.flush()

    async def for_selector(self, selector: str) -> Sequence[HealthCheckRecovery]:
        """One model's recorded outages, newest first."""
        result = await self.db.execute(
            select(HealthCheckRecovery)
            .where(HealthCheckRecovery.selector == selector)
            .order_by(HealthCheckRecovery.recovered_at.desc())
        )
        return result.scalars().all()

    async def delete_before(self, cutoff: datetime) -> None:
        """Remove the outages that ended before ``cutoff``."""
        await self.db.execute(delete(HealthCheckRecovery).where(HealthCheckRecovery.recovered_at < cutoff))
