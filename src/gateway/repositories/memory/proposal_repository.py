"""Data access for ``memory_proposals``. Flushes, never commits."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select, update

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.memory import MemoryKind, MemoryProposal, MemoryScope, MemoryTrust
from gateway.repositories.memory.fenced_repository import FencedRepository


@dataclass(frozen=True)
class NewProposal:
    """What a proposal is stored with. Its text arrives sealed."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    scope: MemoryScope
    owner_user_id: uuid.UUID | None
    author_user_id: uuid.UUID | None
    kind: MemoryKind
    trust: MemoryTrust
    key_id: uuid.UUID
    content_sealed: str
    source_api_key_id: str | None = None
    source_session_id: str | None = None
    source_model: str | None = None


class ProposalRepository(FencedRepository[MemoryProposal]):
    """Store proposals until they are decided, and delete them once they are."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, MemoryProposal)

    async def add(self, proposal: NewProposal) -> MemoryProposal:
        """Stage one pending proposal."""
        await self._fence(proposal.workspace_id)
        row = MemoryProposal(
            id=proposal.id,
            workspace_id=proposal.workspace_id,
            scope=proposal.scope,
            owner_user_id=proposal.owner_user_id,
            author_user_id=proposal.author_user_id,
            kind=proposal.kind,
            trust=proposal.trust,
            state="pending",
            key_id=proposal.key_id,
            content_sealed=proposal.content_sealed,
            source_api_key_id=proposal.source_api_key_id,
            source_session_id=proposal.source_session_id,
            source_model=proposal.source_model,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def in_workspace(self, workspace_id: uuid.UUID, proposal_id: uuid.UUID) -> MemoryProposal | None:
        """One proposal, only if it is in ``workspace_id``."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryProposal).where(MemoryProposal.workspace_id == workspace_id, MemoryProposal.id == proposal_id)
        )
        return result.scalar_one_or_none()

    async def remove(self, workspace_id: uuid.UUID, proposal_id: uuid.UUID) -> bool:
        """Delete one decided proposal; whether the workspace had it."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryProposal)
            .where(MemoryProposal.workspace_id == workspace_id, MemoryProposal.id == proposal_id)
            .returning(MemoryProposal.id)
        )
        return result.scalar_one_or_none() is not None

    async def pending(self, workspace_id: uuid.UUID, *, limit: int) -> Sequence[MemoryProposal]:
        """The workspace's oldest pending proposals no other replica holds, locked until the block ends."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryProposal)
            .where(MemoryProposal.workspace_id == workspace_id, MemoryProposal.state == "pending")
            .order_by(MemoryProposal.created_at, MemoryProposal.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return result.scalars().all()

    async def count_pending(self, workspace_id: uuid.UUID) -> int:
        """How many proposals of the workspace wait for the consolidator."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(func.count())
            .select_from(MemoryProposal)
            .where(MemoryProposal.workspace_id == workspace_id, MemoryProposal.state == "pending")
        )
        return result.scalar_one()

    async def hold_for_review(self, workspace_id: uuid.UUID, proposal_id: uuid.UUID, reason: str) -> None:
        """Leave one proposal for an admin, with the reason code it waits for."""
        await self._fence(workspace_id)
        await self.db.execute(
            update(MemoryProposal)
            .where(MemoryProposal.workspace_id == workspace_id, MemoryProposal.id == proposal_id)
            .values(state="review", review_reason=reason)
            .execution_options(synchronize_session=False)
        )

    async def count_attempt(self, workspace_id: uuid.UUID, proposal_id: uuid.UUID) -> int:
        """Count one failed pass over a proposal; how many it has had."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            update(MemoryProposal)
            .where(MemoryProposal.workspace_id == workspace_id, MemoryProposal.id == proposal_id)
            .values(attempts=MemoryProposal.attempts + 1)
            .returning(MemoryProposal.attempts)
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one_or_none() or 0

    async def delete_older(self, workspace_id: uuid.UUID, *, created_before: datetime) -> int:
        """Delete the workspace's proposals made before ``created_before``, decided or not."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryProposal)
            .where(MemoryProposal.workspace_id == workspace_id, MemoryProposal.created_at < created_before)
            .returning(MemoryProposal.id)
        )
        return len(result.scalars().all())

    async def key_ids_in_use(self, workspace_id: uuid.UUID) -> set[uuid.UUID]:
        """The data keys some proposal of the workspace is sealed under."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryProposal.key_id).where(MemoryProposal.workspace_id == workspace_id).distinct()
        )
        return set(result.scalars().all())

    async def delete_all(self, workspace_id: uuid.UUID) -> int:
        """Delete every proposal of the workspace, decided or not."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryProposal).where(MemoryProposal.workspace_id == workspace_id).returning(MemoryProposal.id)
        )
        return len(result.scalars().all())
