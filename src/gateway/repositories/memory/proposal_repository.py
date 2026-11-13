"""Data access for ``memory_proposals``. Flushes, never commits."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select, update

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.memory import (
    MEMORY_REVIEW_REASONS,
    MemoryKind,
    MemoryProposal,
    MemoryReviewReason,
    MemoryScope,
    MemoryTrust,
)
from gateway.repositories.memory.fenced_repository import FencedRepository

# Between a held proposal's reason code and the entry it would change, in ``review_reason``. An entry id fits beside
# the longest code within the column's 64 characters, so recording it took no migration.
_ENTRY_SEPARATOR = ":"


def review_reason(reason: MemoryReviewReason, entry_id: uuid.UUID | None) -> str:
    """What ``review_reason`` holds for ``reason``, naming the entry the proposal would change where there is one."""
    return reason if entry_id is None else f"{reason}{_ENTRY_SEPARATOR}{entry_id}"


def parse_review_reason(value: str | None) -> tuple[MemoryReviewReason | None, uuid.UUID | None]:
    """The reason code and entry id a ``review_reason`` holds; None for either it does not hold."""
    if value is None:
        return None, None
    code, _, entry = value.partition(_ENTRY_SEPARATOR)
    reason = next((known for known in MEMORY_REVIEW_REASONS if known == code), None)
    try:
        entry_id = uuid.UUID(entry) if entry else None
    except ValueError:
        entry_id = None
    return reason, entry_id


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

    async def hold_for_review(
        self,
        workspace_id: uuid.UUID,
        proposal_id: uuid.UUID,
        reason: MemoryReviewReason,
        *,
        entry_id: uuid.UUID | None = None,
    ) -> None:
        """Leave one proposal for an admin, with the reason code it waits for and the entry it would change, if any."""
        await self._fence(workspace_id)
        await self.db.execute(
            update(MemoryProposal)
            .where(MemoryProposal.workspace_id == workspace_id, MemoryProposal.id == proposal_id)
            .values(state="review", review_reason=review_reason(reason, entry_id))
            .execution_options(synchronize_session=False)
        )

    async def held(self, workspace_id: uuid.UUID, *, scope: MemoryScope) -> Sequence[MemoryProposal]:
        """The workspace's proposals of ``scope`` waiting for an admin, oldest first."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryProposal)
            .where(
                MemoryProposal.workspace_id == workspace_id,
                MemoryProposal.state == "review",
                MemoryProposal.scope == scope,
            )
            .order_by(MemoryProposal.created_at, MemoryProposal.id)
        )
        return result.scalars().all()

    async def count_held(self, workspace_id: uuid.UUID, *, scope: MemoryScope) -> int:
        """How many proposals of ``scope`` wait for an admin."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(func.count())
            .select_from(MemoryProposal)
            .where(
                MemoryProposal.workspace_id == workspace_id,
                MemoryProposal.state == "review",
                MemoryProposal.scope == scope,
            )
        )
        return result.scalar_one()

    async def approve(self, workspace_id: uuid.UUID, proposal_id: uuid.UUID) -> bool:
        """Send one held proposal back to the consolidator as an admin's word; whether it was still held.

        It goes back as ``reviewed``, with its failed passes forgotten and its reason kept, so the consolidator
        knows the entry the admin saw it beside.
        """
        await self._fence(workspace_id)
        result = await self.db.execute(
            update(MemoryProposal)
            .where(
                MemoryProposal.workspace_id == workspace_id,
                MemoryProposal.id == proposal_id,
                MemoryProposal.state == "review",
            )
            .values(state="pending", trust="reviewed", attempts=0)
            .returning(MemoryProposal.id)
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one_or_none() is not None

    async def remove_held(self, workspace_id: uuid.UUID, proposal_id: uuid.UUID, *, scope: MemoryScope) -> bool:
        """Delete one proposal of ``scope`` waiting for an admin; whether the workspace had it."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryProposal)
            .where(
                MemoryProposal.workspace_id == workspace_id,
                MemoryProposal.id == proposal_id,
                MemoryProposal.state == "review",
                MemoryProposal.scope == scope,
            )
            .returning(MemoryProposal.id)
        )
        return result.scalar_one_or_none() is not None

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
