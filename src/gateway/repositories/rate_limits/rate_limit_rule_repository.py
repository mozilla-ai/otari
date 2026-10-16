"""Data access for the rate-limit rules stored through the dashboard. Flushes, never commits."""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.rate_limits import StoredRateLimitRule
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.rate_limits import RateLimitRuleCreate, RateLimitRuleUpdate


class RateLimitRuleNameTaken(Exception):
    """The primary key refused a name another writer stored first.

    Raised here rather than letting ``IntegrityError`` travel, because the
    service that maps it to its domain error may not import a database library.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class RateLimitRuleRepository(BaseRepository[StoredRateLimitRule, RateLimitRuleCreate, RateLimitRuleUpdate]):
    """Repository for ``rate_limit_rules`` rows."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        super().__init__(db, StoredRateLimitRule)

    async def list_all(self) -> Sequence[StoredRateLimitRule]:
        """Every stored rule, in name order. The table holds a handful, so it is read whole.

        ``populate_existing`` so a refresh on a long-lived session reads what is
        committed rather than what its identity map loaded earlier.
        """
        result = await self.db.execute(
            select(StoredRateLimitRule).order_by(StoredRateLimitRule.name).execution_options(populate_existing=True)
        )
        return result.scalars().all()

    async def add(self, row: StoredRateLimitRule) -> StoredRateLimitRule:
        """Stage a new rule, flushed so the primary key answers while the caller can name it.

        Raises:
            RateLimitRuleNameTaken: a rule with this name is already stored.
        """
        self.db.add(row)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            raise RateLimitRuleNameTaken(row.name) from exc
        await self.db.refresh(row)
        return row
