"""Data access for ``workspace_code_execution_policies``, one row per workspace.

The only module that constructs or queries the table: the policy service writes
through it, and so does a workspace's creation when it starts with a policy.
"""

import uuid
from typing import Never

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import WorkspaceCodeExecutionPolicy
from gateway.repositories.base_repository import BaseRepository


class WorkspaceCodeExecutionPolicyRepository(BaseRepository[WorkspaceCodeExecutionPolicy, Never, Never]):
    """Read and stage policy rows. Writes flush and leave the commit to the caller's Unit of Work."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, WorkspaceCodeExecutionPolicy)

    async def put(
        self,
        workspace_id: uuid.UUID,
        *,
        enabled: bool,
        default_purpose_hint: str | None,
        max_iterations: int | None,
        exec_timeout_s: int | None,
        image: str | None,
        tools: list[str] | None,
        executor: str | None,
    ) -> WorkspaceCodeExecutionPolicy:
        """Stage the workspace's whole policy, creating its row when it has none.

        Two writers may both find no row and both insert, and the primary key
        refuses the second. A ``PUT`` of the whole policy is idempotent, so the
        loser applies its values over the winner's row instead. The insert runs
        in a SAVEPOINT, so losing rolls back that row alone and leaves the
        block's session usable.
        """
        policy = await self.db.get(WorkspaceCodeExecutionPolicy, workspace_id)
        if policy is None:
            policy = WorkspaceCodeExecutionPolicy(workspace_id=workspace_id)
            try:
                async with self.db.begin_nested():
                    self.db.add(policy)
            except IntegrityError:
                policy = await self.db.get(WorkspaceCodeExecutionPolicy, workspace_id)
                if policy is None:
                    raise  # not the race: nothing is there to have collided with
        policy.enabled = enabled
        policy.default_purpose_hint = default_purpose_hint
        policy.max_iterations = max_iterations
        policy.exec_timeout_s = exec_timeout_s
        policy.image = image
        policy.tools = tools
        policy.executor = executor
        await self.db.flush()
        await self.db.refresh(policy)
        return policy

    async def create_enabled(self, workspace_id: uuid.UUID) -> None:
        """Stage a policy that turns code execution on for a workspace."""
        self.db.add(WorkspaceCodeExecutionPolicy(workspace_id=workspace_id, enabled=True))
        await self.db.flush()

    async def delete_for(self, workspace_id: uuid.UUID) -> None:
        """Stage the removal of the workspace's policy, if it has one."""
        policy = await self.db.get(WorkspaceCodeExecutionPolicy, workspace_id)
        if policy is not None:
            await self.db.delete(policy)
            await self.db.flush()
