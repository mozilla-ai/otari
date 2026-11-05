"""Data access for a workspace's narrowing of its providers: per hosted provider, and its policy over all of them.

Built on the Unit of Work. Flushes, never commits.
"""

import uuid
from collections.abc import Sequence
from typing import Never

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.providers import WorkspaceHostedProviderAccess, WorkspaceProviderPolicy
from gateway.repositories.base_repository import BaseRepository


class WorkspaceHostedAccessRepository(BaseRepository[WorkspaceHostedProviderAccess, Never, Never]):
    """Read and stage one workspace's hosted provider rows."""

    def __init__(self, db: UnitOfWork) -> None:
        super().__init__(db, WorkspaceHostedProviderAccess)

    async def list_for_workspace(self, workspace_id: uuid.UUID) -> Sequence[WorkspaceHostedProviderAccess]:
        """Every row one workspace stores, by provider name."""
        result = await self.db.execute(
            select(WorkspaceHostedProviderAccess)
            .where(WorkspaceHostedProviderAccess.workspace_id == workspace_id)
            .order_by(WorkspaceHostedProviderAccess.provider)
        )
        return result.scalars().all()

    async def list_all(self) -> Sequence[WorkspaceHostedProviderAccess]:
        """Every row on the deployment, for the dispatch-path cache."""
        return (await self.db.execute(select(WorkspaceHostedProviderAccess))).scalars().all()

    async def put(
        self, workspace_id: uuid.UUID, provider: str, *, enabled: bool, models: list[str] | None
    ) -> WorkspaceHostedProviderAccess:
        """Stage the workspace's row for ``provider``, creating it when there is none.

        Two writers may both find no row and both insert, and the unique
        constraint refuses the second. The write replaces the whole row, so the
        loser applies its values over the winner's instead. The insert runs in
        a SAVEPOINT, so losing rolls back that row alone.
        """
        row = await self._get(workspace_id, provider)
        if row is None:
            row = WorkspaceHostedProviderAccess(workspace_id=workspace_id, provider=provider)
            try:
                async with self.db.begin_nested():
                    self.db.add(row)
            except IntegrityError:
                row = await self._get(workspace_id, provider)
                if row is None:
                    raise  # not the race: nothing is there to have collided with
        row.enabled = enabled
        row.models = models
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def delete_for(self, workspace_id: uuid.UUID, provider: str) -> None:
        """Stage the removal of the workspace's row for ``provider``, if it has one."""
        await self.db.execute(
            delete(WorkspaceHostedProviderAccess).where(
                WorkspaceHostedProviderAccess.workspace_id == workspace_id,
                WorkspaceHostedProviderAccess.provider == provider,
            )
        )
        await self.db.flush()

    async def _get(self, workspace_id: uuid.UUID, provider: str) -> WorkspaceHostedProviderAccess | None:
        result = await self.db.execute(
            select(WorkspaceHostedProviderAccess).where(
                WorkspaceHostedProviderAccess.workspace_id == workspace_id,
                WorkspaceHostedProviderAccess.provider == provider,
            )
        )
        return result.scalar_one_or_none()


class WorkspaceProviderPolicyRepository(BaseRepository[WorkspaceProviderPolicy, Never, Never]):
    """Read and stage what each workspace requires of its providers, one row per workspace or none."""

    def __init__(self, db: UnitOfWork) -> None:
        super().__init__(db, WorkspaceProviderPolicy)

    async def get_for(self, workspace_id: uuid.UUID) -> WorkspaceProviderPolicy | None:
        return await self.db.get(WorkspaceProviderPolicy, workspace_id)

    async def zero_data_retention_workspaces(self) -> frozenset[uuid.UUID]:
        """Every workspace that requires zero data retention, for the dispatch-path cache."""
        result = await self.db.execute(
            select(WorkspaceProviderPolicy.workspace_id).where(
                WorkspaceProviderPolicy.require_zero_data_retention.is_(True)
            )
        )
        return frozenset(result.scalars().all())

    async def put(self, workspace_id: uuid.UUID, *, require_zero_data_retention: bool) -> WorkspaceProviderPolicy:
        """Stage the workspace's policy, creating its row when it has none.

        Two writers may both find no row; the primary key refuses the second,
        which applies its value over the winner's instead, in a SAVEPOINT.
        """
        policy = await self.get_for(workspace_id)
        if policy is None:
            policy = WorkspaceProviderPolicy(workspace_id=workspace_id)
            try:
                async with self.db.begin_nested():
                    self.db.add(policy)
            except IntegrityError:
                policy = await self.get_for(workspace_id)
                if policy is None:
                    raise  # not the race: nothing is there to have collided with
        policy.require_zero_data_retention = require_zero_data_retention
        await self.db.flush()
        await self.db.refresh(policy)
        return policy
