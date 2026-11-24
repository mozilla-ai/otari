"""Data access for the subagent model recommendations the gateway answered.

Built on the Unit of Work. Flushes, never commits.
"""

from typing import Any, Never

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.routing import SubagentDecision
from gateway.repositories.base_repository import BaseRepository


class SubagentDecisionRepository(BaseRepository[SubagentDecision, Never, Never]):
    """Repository for `subagent_decisions` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, SubagentDecision)

    async def insert(self, **fields: Any) -> SubagentDecision:
        """Add a row from the fields the service chose. Flushed, not committed."""
        row = SubagentDecision(**fields)
        self.db.add(row)
        await self.db.flush()
        return row
