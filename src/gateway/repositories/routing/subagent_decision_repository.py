"""Data access for the subagent model recommendations the gateway answered.

Built on the Unit of Work. Flushes, never commits.
"""

import uuid
from typing import Any, Never

from sqlalchemy import select

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

    async def get_for_caller(
        self,
        decision_id: uuid.UUID,
        *,
        user_id: str | None,
        workspace_id: uuid.UUID | None,
        api_key_id: str | None = None,
    ) -> SubagentDecision | None:
        """One decision by id, or None. A user, a workspace or a key given narrows it to that caller's own.

        The row is locked until the transaction ends, so a read-then-write on it does not interleave.
        """
        statement = select(SubagentDecision).where(SubagentDecision.id == decision_id).with_for_update()
        if user_id is not None:
            statement = statement.where(SubagentDecision.user_id == user_id)
        if api_key_id is not None:
            statement = statement.where(SubagentDecision.api_key_id == api_key_id)
        if workspace_id is not None:
            statement = statement.where(SubagentDecision.workspace_id == workspace_id)
        return (await self.db.execute(statement)).scalar_one_or_none()

    async def save(self, row: SubagentDecision) -> None:
        """Flush the changes made to ``row``."""
        await self.db.flush()
