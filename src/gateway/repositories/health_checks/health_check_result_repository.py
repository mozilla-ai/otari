"""Data access for health check results. Flushes, never commits."""

import uuid
from collections.abc import Collection, Iterable, Sequence

from sqlalchemy import case, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.health_checks import HealthCheckResult
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.health_checks import HealthCheckResultPublic


class HealthCheckResultRepository(BaseRepository[HealthCheckResult, HealthCheckResultPublic, HealthCheckResultPublic]):
    """Repository for ``health_check_results`` rows."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        super().__init__(db, HealthCheckResult)

    async def add_all(self, run_id: uuid.UUID, results: Iterable[HealthCheckResultPublic]) -> None:
        """Stage one run's results."""
        self.db.add_all(
            HealthCheckResult(run_id=run_id, **result.model_dump(exclude={"failing_since", "failed_runs"}))
            for result in results
        )
        await self.db.flush()

    async def for_runs(
        self, run_ids: Sequence[uuid.UUID], selectors: Collection[str] | None = None
    ) -> Sequence[HealthCheckResult]:
        """Every result of these runs, or only those of ``selectors``, in selector order."""
        if not run_ids or (selectors is not None and not selectors):
            return []
        query = select(HealthCheckResult).where(HealthCheckResult.run_id.in_(run_ids))
        if selectors is not None:
            query = query.where(HealthCheckResult.selector.in_(selectors))
        result = await self.db.execute(query.order_by(HealthCheckResult.selector))
        return result.scalars().all()

    async def tallies(self, run_ids: Sequence[uuid.UUID]) -> dict[tuple[str, uuid.UUID], tuple[int, int]]:
        """How many of each provider's checked offerings answered in each run, as ``(answered, total)``.

        Counted in the database, so the page's history reads one row per provider per run.
        """
        if not run_ids:
            return {}
        answered = func.sum(case((HealthCheckResult.status == "pass", 1), else_=0))
        result = await self.db.execute(
            select(HealthCheckResult.provider, HealthCheckResult.run_id, answered, func.count())
            .where(HealthCheckResult.run_id.in_(run_ids), HealthCheckResult.status != "skip")
            .group_by(HealthCheckResult.provider, HealthCheckResult.run_id)
        )
        return {(provider, run_id): (int(passed or 0), int(total)) for provider, run_id, passed, total in result.all()}

    async def delete_for_runs(self, run_ids: Sequence[uuid.UUID]) -> None:
        """Remove the results of these runs, for a database that does not cascade (SQLite)."""
        if run_ids:
            await self.db.execute(delete(HealthCheckResult).where(HealthCheckResult.run_id.in_(run_ids)))
