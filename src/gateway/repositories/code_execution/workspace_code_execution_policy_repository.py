"""Data access for ``workspace_code_execution_policies`` that creating a workspace needs.

Reads and edits of a policy go through the policy service; this is the one
write a workspace's creation makes, inside the creator's own transaction.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tools import WorkspaceCodeExecutionPolicy


class WorkspaceCodeExecutionPolicyRepository:
    """Stages policy rows on the caller's session; the caller owns the transaction."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_enabled(self, workspace_id: uuid.UUID) -> None:
        """Stage a policy that turns code execution on for a workspace."""
        self.db.add(WorkspaceCodeExecutionPolicy(workspace_id=workspace_id, enabled=True))
        await self.db.flush()
