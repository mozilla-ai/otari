"""Data access for ``trace_content_barriers`` and ``trace_user_barriers``. Flushes, never commits."""

import uuid
from collections.abc import Collection
from datetime import datetime
from typing import Never

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import TraceContentBarrier, TraceUserBarrier
from gateway.repositories.base_repository import BaseRepository


class ContentBarrierRepository(BaseRepository[TraceContentBarrier, Never, Never]):
    """Read, lock and move on each workspace's content generation."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceContentBarrier)

    async def generation(self, workspace_id: uuid.UUID) -> int:
        """The workspace's current generation, without locking it; 0 before content was ever turned off."""
        result = await self.db.execute(
            select(TraceContentBarrier.generation).where(TraceContentBarrier.workspace_id == workspace_id)
        )
        return int(result.scalar_one_or_none() or 0)

    async def lock(self, workspace_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, int]:
        """Lock each workspace's row until the transaction ends, creating any missing, and return their generations.

        Rows are created first so there is always a row to lock, and locked in id
        order so two transactions cannot take them in opposite orders.
        """
        if not workspace_ids:
            return {}
        ordered = sorted(workspace_ids, key=str)
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        await self.db.execute(
            insert(TraceContentBarrier)
            .values([{"workspace_id": workspace_id, "generation": 0} for workspace_id in ordered])
            .on_conflict_do_nothing(index_elements=[TraceContentBarrier.workspace_id])
        )
        result = await self.db.execute(
            select(TraceContentBarrier.workspace_id, TraceContentBarrier.generation)
            .where(TraceContentBarrier.workspace_id.in_(ordered))
            .order_by(TraceContentBarrier.workspace_id)
            .with_for_update()
        )
        return {workspace_id: int(generation) for workspace_id, generation in result.tuples().all()}

    async def advance(self, workspace_id: uuid.UUID) -> None:
        """Move the workspace on to a new generation, holding its row until the transaction ends."""
        await self.lock([workspace_id])
        await self.db.execute(
            update(TraceContentBarrier)
            .where(TraceContentBarrier.workspace_id == workspace_id)
            .values(generation=TraceContentBarrier.generation + 1)
        )


class UserBarrierRepository(BaseRepository[TraceUserBarrier, Never, Never]):
    """Lock and move on each user's erasure time."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceUserBarrier)

    async def lock(self, user_ids: Collection[str]) -> dict[str, datetime | None]:
        """Lock each user's row until the transaction ends, creating any missing, and return when each was erased."""
        if not user_ids:
            return {}
        ordered = sorted(user_ids)
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        await self.db.execute(
            insert(TraceUserBarrier)
            .values([{"user_id": user_id} for user_id in ordered])
            .on_conflict_do_nothing(index_elements=[TraceUserBarrier.user_id])
        )
        result = await self.db.execute(
            select(TraceUserBarrier.user_id, TraceUserBarrier.erased_at)
            .where(TraceUserBarrier.user_id.in_(ordered))
            .order_by(TraceUserBarrier.user_id)
            .with_for_update()
        )
        return dict(result.tuples().all())

    async def forget_erased_before(self, cutoff: datetime) -> int:
        """Delete the rows of users erased before ``cutoff``, so an erased user's id is not kept past its use."""
        result = await self.db.execute(
            delete(TraceUserBarrier).where(TraceUserBarrier.erased_at.is_not(None), TraceUserBarrier.erased_at < cutoff)
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def erase(self, user_id: str, at: datetime) -> None:
        """Record the user's erasure, holding their row until the transaction ends."""
        await self.lock([user_id])
        await self.db.execute(update(TraceUserBarrier).where(TraceUserBarrier.user_id == user_id).values(erased_at=at))
