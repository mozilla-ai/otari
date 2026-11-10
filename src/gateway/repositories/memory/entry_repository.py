"""Data access for ``memory_entries``. Flushes, never commits."""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, or_, select, update

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.memory import MemoryEntry, MemoryKind, MemoryScope, MemoryStatus, MemoryTrust
from gateway.repositories.memory.fenced_repository import FencedRepository


@dataclass(frozen=True)
class NewEntry:
    """What a new entry version is stored with. Its text and embedding arrive sealed."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    scope: MemoryScope
    owner_user_id: uuid.UUID | None
    author_user_id: uuid.UUID | None
    kind: MemoryKind
    trust: MemoryTrust
    key_id: uuid.UUID
    content_sealed: str
    embedding_sealed: str | None
    embedding_model: str | None
    confidence: float
    version: int = 1
    supersedes_id: uuid.UUID | None = None
    source_api_key_id: str | None = None
    source_session_id: str | None = None
    source_model: str | None = None


class EntryRepository(FencedRepository[MemoryEntry]):
    """Store entry versions, and find the ones a caller may recall."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, MemoryEntry)

    async def add(self, entry: NewEntry) -> MemoryEntry:
        """Stage one entry version, active."""
        await self._fence(entry.workspace_id)
        row = MemoryEntry(
            id=entry.id,
            workspace_id=entry.workspace_id,
            scope=entry.scope,
            owner_user_id=entry.owner_user_id,
            author_user_id=entry.author_user_id,
            kind=entry.kind,
            status="active",
            trust=entry.trust,
            version=entry.version,
            supersedes_id=entry.supersedes_id,
            key_id=entry.key_id,
            content_sealed=entry.content_sealed,
            embedding_sealed=entry.embedding_sealed,
            embedding_model=entry.embedding_model,
            confidence=entry.confidence,
            source_api_key_id=entry.source_api_key_id,
            source_session_id=entry.source_session_id,
            source_model=entry.source_model,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def recallable(
        self, workspace_id: uuid.UUID, *, member_id: uuid.UUID, include_workspace: bool, include_external: bool
    ) -> Sequence[MemoryEntry]:
        """The active entries ``member_id`` may recall: their own, and the workspace's where ``include_workspace``.

        Another member's entries are never among them, whoever asks.
        """
        await self._fence(workspace_id)
        visible = MemoryEntry.owner_user_id == member_id
        if include_workspace:
            visible = or_(visible, MemoryEntry.scope == "workspace")
        query = select(MemoryEntry).where(
            MemoryEntry.workspace_id == workspace_id, MemoryEntry.status == "active", visible
        )
        if not include_external:
            query = query.where(MemoryEntry.trust != "external")
        result = await self.db.execute(query.order_by(MemoryEntry.created_at, MemoryEntry.id))
        return result.scalars().all()

    async def in_workspace(self, workspace_id: uuid.UUID, entry_id: uuid.UUID) -> MemoryEntry | None:
        """One entry version, only if it is in ``workspace_id``."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryEntry).where(MemoryEntry.workspace_id == workspace_id, MemoryEntry.id == entry_id)
        )
        return result.scalar_one_or_none()

    async def count_active(self, workspace_id: uuid.UUID) -> int:
        """How many active entries the workspace holds, every scope counted."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(func.count())
            .select_from(MemoryEntry)
            .where(MemoryEntry.workspace_id == workspace_id, MemoryEntry.status == "active")
        )
        return result.scalar_one()

    async def set_status(self, workspace_id: uuid.UUID, entry_id: uuid.UUID, status: MemoryStatus) -> bool:
        """Move one entry to ``status``; whether the workspace had it."""
        row = await self.in_workspace(workspace_id, entry_id)
        if row is None:
            return False
        row.status = status
        await self.db.flush()
        return True

    async def comparable(
        self, workspace_id: uuid.UUID, *, scope: MemoryScope, owner_user_id: uuid.UUID | None, embedding_model: str
    ) -> Sequence[MemoryEntry]:
        """The active entries a proposal of ``scope`` is compared with: same scope, same owner, same model."""
        await self._fence(workspace_id)
        query = select(MemoryEntry).where(
            MemoryEntry.workspace_id == workspace_id,
            MemoryEntry.status == "active",
            MemoryEntry.scope == scope,
            MemoryEntry.embedding_model == embedding_model,
        )
        if owner_user_id is not None:
            query = query.where(MemoryEntry.owner_user_id == owner_user_id)
        result = await self.db.execute(query.order_by(MemoryEntry.created_at, MemoryEntry.id))
        return result.scalars().all()

    async def least_used(self, workspace_id: uuid.UUID) -> MemoryEntry | None:
        """The active entry nobody has used for longest, which the cap retires first."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryEntry)
            .where(MemoryEntry.workspace_id == workspace_id, MemoryEntry.status == "active")
            .order_by(MemoryEntry.last_used_at, MemoryEntry.created_at, MemoryEntry.id)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def touch(self, workspace_id: uuid.UUID, entry_ids: Collection[uuid.UUID], at: datetime) -> None:
        """Count one use of each entry, as of ``at``."""
        if not entry_ids:
            return
        await self._fence(workspace_id)
        await self.db.execute(
            update(MemoryEntry)
            .where(MemoryEntry.workspace_id == workspace_id, MemoryEntry.id.in_(list(entry_ids)))
            .values(use_count=MemoryEntry.use_count + 1, last_used_at=at)
            .execution_options(synchronize_session=False)
        )

    async def set_signals(
        self,
        workspace_id: uuid.UUID,
        entry_id: uuid.UUID,
        *,
        confirmations: int,
        flags: int,
        confidence: float,
        status: MemoryStatus,
    ) -> None:
        """Store what the members' signals add up to for one entry."""
        await self._fence(workspace_id)
        await self.db.execute(
            update(MemoryEntry)
            .where(MemoryEntry.workspace_id == workspace_id, MemoryEntry.id == entry_id)
            .values(confirmations=confirmations, flags=flags, confidence=confidence, status=status)
            .execution_options(synchronize_session=False)
        )

    async def delete_versions(self, workspace_id: uuid.UUID, entry_ids: Collection[uuid.UUID]) -> int:
        """Delete entry versions outright; how many the workspace had."""
        if not entry_ids:
            return 0
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryEntry)
            .where(MemoryEntry.workspace_id == workspace_id, MemoryEntry.id.in_(list(entry_ids)))
            .returning(MemoryEntry.id)
        )
        return len(result.scalars().all())
