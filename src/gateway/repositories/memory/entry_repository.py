"""Data access for ``memory_entries``. Flushes, never commits."""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import ColumnElement, and_, delete, func, or_, select, update

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


@dataclass(frozen=True)
class RecallableRef:
    """An entry a member may recall, as recall ranks it: no sealed column read."""

    id: uuid.UUID
    confidence: float


@dataclass(frozen=True)
class SealedEmbedding:
    """An entry's sealed embedding, with what opening it is bound to."""

    id: uuid.UUID
    scope: MemoryScope
    owner_user_id: uuid.UUID | None
    key_id: uuid.UUID
    embedding_sealed: str


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
        query = select(MemoryEntry).where(_recallable(workspace_id, member_id, include_workspace, include_external))
        result = await self.db.execute(query.order_by(MemoryEntry.created_at, MemoryEntry.id))
        return result.scalars().all()

    async def recallable_refs(
        self,
        workspace_id: uuid.UUID,
        *,
        member_id: uuid.UUID,
        include_workspace: bool,
        include_external: bool,
        embedding_model: str,
    ) -> list[RecallableRef]:
        """The entries :meth:`recallable` returns that ``embedding_model`` embedded, by id and confidence alone."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryEntry.id, MemoryEntry.confidence)
            .where(
                _recallable(workspace_id, member_id, include_workspace, include_external),
                MemoryEntry.embedding_model == embedding_model,
                MemoryEntry.embedding_sealed.is_not(None),
            )
            .order_by(MemoryEntry.created_at, MemoryEntry.id)
        )
        return [RecallableRef(entry_id, confidence) for entry_id, confidence in result.all()]

    async def sealed_embeddings(
        self, workspace_id: uuid.UUID, entry_ids: Collection[uuid.UUID]
    ) -> list[SealedEmbedding]:
        """The sealed embeddings of these entries of the workspace; an id it does not hold is left out."""
        await self._fence(workspace_id)
        if not entry_ids:
            return []
        result = await self.db.execute(
            select(
                MemoryEntry.id,
                MemoryEntry.scope,
                MemoryEntry.owner_user_id,
                MemoryEntry.key_id,
                MemoryEntry.embedding_sealed,
            ).where(
                MemoryEntry.workspace_id == workspace_id,
                MemoryEntry.id.in_(entry_ids),
                MemoryEntry.embedding_sealed.is_not(None),
            )
        )
        return [
            SealedEmbedding(entry_id, scope, owner_user_id, key_id, sealed)
            for entry_id, scope, owner_user_id, key_id, sealed in result.all()
            if sealed is not None
        ]

    async def many_in_workspace(
        self, workspace_id: uuid.UUID, entry_ids: Collection[uuid.UUID]
    ) -> Sequence[MemoryEntry]:
        """These entry versions, only those in ``workspace_id``."""
        await self._fence(workspace_id)
        if not entry_ids:
            return []
        result = await self.db.execute(
            select(MemoryEntry).where(MemoryEntry.workspace_id == workspace_id, MemoryEntry.id.in_(entry_ids))
        )
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

    async def delete_expired(self, workspace_id: uuid.UUID, *, used_before: datetime) -> int:
        """Delete the workspace's retired entries, and every version nobody has used since ``used_before``."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryEntry)
            .where(
                MemoryEntry.workspace_id == workspace_id,
                or_(MemoryEntry.status == "retired", MemoryEntry.last_used_at < used_before),
            )
            .returning(MemoryEntry.id)
        )
        return len(result.scalars().all())

    async def key_ids_in_use(self, workspace_id: uuid.UUID) -> set[uuid.UUID]:
        """The data keys some entry of the workspace is sealed under."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryEntry.key_id).where(MemoryEntry.workspace_id == workspace_id).distinct()
        )
        return set(result.scalars().all())

    async def delete_all(self, workspace_id: uuid.UUID) -> int:
        """Delete every entry version of the workspace, and with them every signal on them."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryEntry).where(MemoryEntry.workspace_id == workspace_id).returning(MemoryEntry.id)
        )
        return len(result.scalars().all())

    async def count_by_status(self, workspace_id: uuid.UUID) -> dict[str, int]:
        """How many entry versions the workspace holds, by status."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryEntry.status, func.count())
            .where(MemoryEntry.workspace_id == workspace_id)
            .group_by(MemoryEntry.status)
        )
        return {status: count for status, count in result.all()}


def _recallable(
    workspace_id: uuid.UUID, member_id: uuid.UUID, include_workspace: bool, include_external: bool
) -> ColumnElement[bool]:
    """The workspace's active entries ``member_id`` may recall: their own, and the workspace's where it shares.

    Another member's entries are never among them, whoever asks.
    """
    visible = MemoryEntry.owner_user_id == member_id
    if include_workspace:
        visible = or_(visible, MemoryEntry.scope == "workspace")
    condition = and_(MemoryEntry.workspace_id == workspace_id, MemoryEntry.status == "active", visible)
    return and_(condition, MemoryEntry.trust != "external") if not include_external else condition
