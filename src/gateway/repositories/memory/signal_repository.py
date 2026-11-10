"""Data access for ``memory_signals``. Flushes, never commits."""

import uuid

from sqlalchemy import func, select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.memory import MemorySignal, MemorySignalKind
from gateway.repositories.memory.fenced_repository import FencedRepository


class SignalRepository(FencedRepository[MemorySignal]):
    """Keep each member's latest word on an entry."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, MemorySignal)

    async def of_member(self, workspace_id: uuid.UUID, entry_id: uuid.UUID, user_id: uuid.UUID) -> MemorySignal | None:
        """The member's word on the entry, or None while they have given none."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemorySignal).where(
                MemorySignal.workspace_id == workspace_id,
                MemorySignal.entry_id == entry_id,
                MemorySignal.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def put(
        self,
        workspace_id: uuid.UUID,
        entry_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        signal: MemorySignalKind,
        reason: str | None,
    ) -> MemorySignal:
        """Stage the member's word on the entry, replacing the one they gave before."""
        row = await self.of_member(workspace_id, entry_id, user_id)
        if row is None:
            row = MemorySignal(workspace_id=workspace_id, entry_id=entry_id, user_id=user_id, signal=signal)
            self.db.add(row)
        row.signal = signal
        row.reason = reason
        await self.db.flush()
        return row

    async def counts(self, workspace_id: uuid.UUID, entry_id: uuid.UUID) -> tuple[int, int]:
        """How many members confirm the entry, and how many flag it."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemorySignal.signal, func.count())
            .where(MemorySignal.workspace_id == workspace_id, MemorySignal.entry_id == entry_id)
            .group_by(MemorySignal.signal)
        )
        tally = {signal: count for signal, count in result.all()}
        return tally.get("confirm", 0), tally.get("flag", 0)
