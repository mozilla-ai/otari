"""Data access for the decision models that recommend a subagent's model.

Built on the Unit of Work. Flushes, never commits. The service hands this
repository an encrypted key: the plaintext never lands in a column.
"""

import uuid
from collections.abc import Sequence
from typing import Any, Never

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.routing_exceptions import AgentRecommenderSwitchConflictError
from gateway.models.routing import AGENT_RECOMMENDER_ENABLED_INDEX, AgentRecommender
from gateway.repositories.base_repository import BaseRepository

# SQLite reports the violated index by table and column, PostgreSQL by name.
_SQLITE_ENABLED_VIOLATION = f"UNIQUE constraint failed: {AgentRecommender.__tablename__}.enabled"


class AgentRecommenderRepository(BaseRepository[AgentRecommender, Never, Never]):
    """Repository for `agent_recommenders` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, AgentRecommender)

    async def list_all(self) -> Sequence[AgentRecommender]:
        """Every recommender, oldest first, read fresh so a switch made by another step is seen."""
        statement = (
            select(AgentRecommender)
            .order_by(AgentRecommender.created_at, AgentRecommender.id)
            .execution_options(populate_existing=True)
        )
        return (await self.db.execute(statement)).scalars().all()

    async def get(self, recommender_id: uuid.UUID) -> AgentRecommender | None:
        """One recommender by id, or None."""
        return await self.db.get(AgentRecommender, recommender_id)

    async def insert(self, **fields: Any) -> AgentRecommender:
        """Add a row from the fields the service checked. Flushed, not committed.

        Raises:
            AgentRecommenderSwitchConflictError: the row was switched on beside another that is.

        """
        return await self.save(AgentRecommender(**fields))

    async def save(self, row: AgentRecommender) -> AgentRecommender:
        """Flush ``row`` and read its defaults back.

        Raises:
            AgentRecommenderSwitchConflictError: the row was switched on beside another that is.

        """
        self.db.add(row)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            message = str(exc.orig)
            if AGENT_RECOMMENDER_ENABLED_INDEX not in message and _SQLITE_ENABLED_VIOLATION not in message:
                raise
            raise AgentRecommenderSwitchConflictError from None
        await self.db.refresh(row)
        return row

    async def switch_off_all_but(self, recommender_id: uuid.UUID) -> None:
        """Switch off every recommender but this one."""
        await self.db.execute(
            update(AgentRecommender)
            .where(AgentRecommender.enabled, AgentRecommender.id != recommender_id)
            .values(enabled=False)
        )

    async def delete_row(self, row: AgentRecommender) -> None:
        """Remove ``row``. Flushed, not committed."""
        await self.db.delete(row)
        await self.db.flush()
