"""Data access for ``workspace_code_execution_policies`` that creating a workspace needs.

Reads and edits of a policy go through the policy service; this is the one
write a workspace's creation makes, inside the creator's own transaction.
"""

import uuid
from typing import Never

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import WorkspaceCodeExecutionPolicy
from gateway.repositories.base_repository import BaseRepository


class WorkspaceCodeExecutionPolicyRepository(BaseRepository[WorkspaceCodeExecutionPolicy, Never, Never]):
    """Stage policy rows in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, WorkspaceCodeExecutionPolicy)

    async def create_enabled(self, workspace_id: uuid.UUID) -> None:
        """Stage a policy that turns code execution on for a workspace."""
        self.db.add(WorkspaceCodeExecutionPolicy(workspace_id=workspace_id, enabled=True))
        await self.db.flush()
