"""Data access for ``memory_entries``. Flushes, never commits."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, or_, select

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
