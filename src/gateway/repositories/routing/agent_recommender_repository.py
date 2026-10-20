"""Data access for the decision models an operator adds as subagent recommenders.

Built on the Unit of Work. Flushes, never commits. Rows are written from field
values rather than a request schema: the plaintext key never lands in a column,
so the service hands this repository the ciphertext.
"""

import uuid
from collections.abc import Sequence
from typing import Any, Never

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.routing import AGENT_RECOMMENDER_ENABLED_INDEX, AgentRecommender
from gateway.repositories.base_repository import BaseRepository


class AgentRecommenderSwitchConflict(Exception):
    """The index that lets one row be on refused a second, switched on at the same moment.

    Raised in place of ``IntegrityError`` because the service that maps it to a
    domain error may not import a database library.
    """


class AgentRecommenderRepository(BaseRepository[AgentRecommender, Never, Never]):
    """Repository for ``agent_recommenders`` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, AgentRecommender)

    async def list_all(self) -> Sequence[AgentRecommender]:
        """Every recommender, oldest first. An operator adds them by hand, so the table is read whole.

        ``populate_existing`` so a reload reads what is committed rather than
        what the session's identity map loaded earlier.
        """
        result = await self.db.execute(
            select(AgentRecommender)
            .order_by(AgentRecommender.created_at, AgentRecommender.id)
            .execution_options(populate_existing=True)
        )
        return result.scalars().all()

    async def add(self, fields: dict[str, Any]) -> AgentRecommender:
        """Stage a new recommender from its column values.

        Raises:
            AgentRecommenderSwitchConflict: it is switched on and another one already is.

        """
        row = AgentRecommender(**fields)
        self.db.add(row)
        await self._flush()
        await self.db.refresh(row)
        return row

    async def change(self, row: AgentRecommender, fields: dict[str, Any]) -> AgentRecommender:
        """Stage new column values on ``row``.

        Raises:
            AgentRecommenderSwitchConflict: it is switched on and another one already is.

        """
        for name, value in fields.items():
            setattr(row, name, value)
        await self._flush()
        await self.db.refresh(row)
        return row

    async def switch_off_all_but(self, recommender_id: uuid.UUID) -> None:
        """Switch off every recommender but this one."""
        await self.db.execute(
            update(AgentRecommender)
            .where(AgentRecommender.enabled, AgentRecommender.id != recommender_id)
            .values(enabled=False)
        )

    async def _flush(self) -> None:
        try:
            await self.db.flush()
        except IntegrityError as exc:
            # PostgreSQL names the index; SQLite names the column it covers.
            message = str(exc.orig)
            if AGENT_RECOMMENDER_ENABLED_INDEX not in message and "agent_recommenders.enabled" not in message:
                raise
            raise AgentRecommenderSwitchConflict from exc
