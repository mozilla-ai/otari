"""Data access for stored routing policies.

New reads of the table go here. The routes and the policy store that read and
write it directly predate this module and are on the architecture check's
baseline until they move.
"""

import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.routing import RoutingPolicy
from gateway.repositories.base_repository import BaseRepository


class RoutingPolicyRepository(BaseRepository[RoutingPolicy, Never, Never]):
    """Read stored routing policy rows."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, RoutingPolicy)

    async def count_in_workspaces(self, workspace_ids: Collection[uuid.UUID]) -> int:
        """How many policies these workspaces store, user-scoped ones included."""
        if not workspace_ids:
            return 0
        result = await self.db.execute(
            select(func.count()).select_from(RoutingPolicy).where(RoutingPolicy.workspace_id.in_(workspace_ids))
        )
        return int(result.scalar_one())
