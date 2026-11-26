"""Data access for the setup guide's per-member state (``workspace_member_activation_state``)."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import WorkspaceMemberActivationState


class WorkspaceMemberActivationRepository:
    """One member's setup-guide row in one workspace."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, workspace_id: uuid.UUID, user_id: uuid.UUID) -> WorkspaceMemberActivationState | None:
        """The member's row, or None before the guide stored anything for them."""
        return await self.db.get(WorkspaceMemberActivationState, (workspace_id, user_id))

    async def get_for_update(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> WorkspaceMemberActivationState | None:
        """The member's row, locked for the rest of the transaction."""
        statement = (
            select(WorkspaceMemberActivationState)
            .where(
                WorkspaceMemberActivationState.workspace_id == workspace_id,
                WorkspaceMemberActivationState.user_id == user_id,
            )
            .with_for_update()
        )
        return (await self.db.execute(statement)).scalar_one_or_none()

    def stage(self, workspace_id: uuid.UUID, user_id: uuid.UUID) -> WorkspaceMemberActivationState:
        """Add a new row to the session; the caller flushes it, inside a savepoint where a race is possible."""
        state = WorkspaceMemberActivationState(workspace_id=workspace_id, user_id=user_id)
        self.db.add(state)
        return state
