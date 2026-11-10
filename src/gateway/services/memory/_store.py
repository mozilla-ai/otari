"""Entries and proposals written sealed and read back opened. Nothing here stores or returns text unsealed at rest.

A row that does not open, because its key is gone from the key source or its
value was moved onto another row, is left out of what is read rather than
failing the read, and counted.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from gateway.metrics import REGISTRY, Counter
from gateway.models.memory import MemoryEntry, MemoryKind, MemoryScope, MemoryTrust
from gateway.repositories.memory import MemoryRepositories, NewEntry, NewProposal
from gateway.services.memory._keys import WorkspaceKeys
from gateway.services.memory._sealing import (
    SealedField,
    SealedValueError,
    associated_data,
    open_sealed,
    pack_embedding,
    seal,
    unpack_embedding,
)

UNOPENED_ROWS = Counter(
    "gateway_memory_rows_unopened",
    "Memory rows left out of a read because they did not open, by table",
    ["table"],
    registry=REGISTRY,
)


@dataclass(frozen=True)
class Provenance:
    """Where something an agent wrote came from: who, through which key and session, on which model."""

    author_user_id: uuid.UUID | None
    api_key_id: str | None = None
    session_id: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class EntryDraft:
    """A new entry version, in the clear, as the consolidator decided it."""

    workspace_id: uuid.UUID
    scope: MemoryScope
    owner_user_id: uuid.UUID | None
    kind: MemoryKind
    trust: MemoryTrust
    text: str
    embedding: Sequence[float] | None
    embedding_model: str | None
    confidence: float
    provenance: Provenance
    version: int = 1
    supersedes_id: uuid.UUID | None = None


@dataclass(frozen=True)
class ProposalDraft:
    """What an agent offered, in the clear, before it is sealed."""

    workspace_id: uuid.UUID
    scope: MemoryScope
    owner_user_id: uuid.UUID | None
    kind: MemoryKind
    trust: MemoryTrust
    text: str
    provenance: Provenance


@dataclass(frozen=True)
class OpenedEntry:
    """One active entry version, opened for recall."""

    id: uuid.UUID
    scope: MemoryScope
    owner_user_id: uuid.UUID | None
    author_user_id: uuid.UUID | None
    kind: MemoryKind
    trust: MemoryTrust
    version: int
    text: str
    embedding: tuple[float, ...] | None
    embedding_model: str | None
    confidence: float
    confirmations: int
    flags: int
    created_at: datetime


class EntryStore:
    """Seals what is written to memory, and opens what is read from it, for one Unit of Work's repositories."""

    def __init__(self, repos: MemoryRepositories, keys: WorkspaceKeys) -> None:
        self._repos = repos
        self._keys = keys

    async def write_entry(self, draft: EntryDraft) -> uuid.UUID:
        """Seal and stage one entry version; its id.

        Raises:
            MemorySealingUnavailableError: no data key could be had, so nothing was staged.
        """
        key = await self._keys.for_sealing(self._repos.keys, draft.workspace_id)
        entry_id = uuid.uuid4()

        def aad(field: SealedField) -> bytes:
            return associated_data(
                workspace_id=draft.workspace_id,
                scope=draft.scope,
                owner_user_id=draft.owner_user_id,
                record="entry",
                record_id=entry_id,
                field=field,
            )

        content_sealed = seal(key.plaintext, draft.text.encode(), aad("content"))
        embedding_sealed = (
            seal(key.plaintext, pack_embedding(draft.embedding), aad("embedding"))
            if draft.embedding is not None
            else None
        )
        await self._repos.entries.add(
            NewEntry(
                id=entry_id,
                workspace_id=draft.workspace_id,
                scope=draft.scope,
                owner_user_id=draft.owner_user_id,
                author_user_id=draft.provenance.author_user_id,
                kind=draft.kind,
                trust=draft.trust,
                key_id=key.id,
                content_sealed=content_sealed,
                embedding_sealed=embedding_sealed,
                embedding_model=draft.embedding_model if draft.embedding is not None else None,
                confidence=draft.confidence,
                version=draft.version,
                supersedes_id=draft.supersedes_id,
                source_api_key_id=draft.provenance.api_key_id,
                source_session_id=draft.provenance.session_id,
                source_model=draft.provenance.model,
            )
        )
        return entry_id

    async def write_proposal(self, draft: ProposalDraft) -> uuid.UUID:
        """Seal and stage one proposal; its id.

        Raises:
            MemorySealingUnavailableError: no data key could be had, so nothing was staged.
        """
        key = await self._keys.for_sealing(self._repos.keys, draft.workspace_id)
        proposal_id = uuid.uuid4()
        aad = associated_data(
            workspace_id=draft.workspace_id,
            scope=draft.scope,
            owner_user_id=draft.owner_user_id,
            record="proposal",
            record_id=proposal_id,
            field="content",
        )
        await self._repos.proposals.add(
            NewProposal(
                id=proposal_id,
                workspace_id=draft.workspace_id,
                scope=draft.scope,
                owner_user_id=draft.owner_user_id,
                author_user_id=draft.provenance.author_user_id,
                kind=draft.kind,
                trust=draft.trust,
                key_id=key.id,
                content_sealed=seal(key.plaintext, draft.text.encode(), aad),
                source_api_key_id=draft.provenance.api_key_id,
                source_session_id=draft.provenance.session_id,
                source_model=draft.provenance.model,
            )
        )
        return proposal_id

    async def recallable(
        self, workspace_id: uuid.UUID, *, member_id: uuid.UUID, include_workspace: bool, include_external: bool
    ) -> list[OpenedEntry]:
        """The active entries ``member_id`` may recall, opened; any that does not open is left out."""
        rows = await self._repos.entries.recallable(
            workspace_id, member_id=member_id, include_workspace=include_workspace, include_external=include_external
        )
        return await self._open_entries(workspace_id, rows)

    async def _open_entries(self, workspace_id: uuid.UUID, rows: Sequence[MemoryEntry]) -> list[OpenedEntry]:
        keys = await self._keys.for_opening(self._repos.keys, workspace_id, {row.key_id for row in rows})
        opened: list[OpenedEntry] = []
        for row in rows:
            key = keys.get(row.key_id)
            entry = _open_entry(row, key) if key is not None and row.workspace_id == workspace_id else None
            if entry is None:
                UNOPENED_ROWS.labels(table="memory_entries").inc()
                continue
            opened.append(entry)
        return opened


def _open_entry(row: MemoryEntry, key: bytes) -> OpenedEntry | None:
    def aad(field: SealedField) -> bytes:
        return associated_data(
            workspace_id=row.workspace_id,
            scope=row.scope,
            owner_user_id=row.owner_user_id,
            record="entry",
            record_id=row.id,
            field=field,
        )

    try:
        text = open_sealed(key, row.content_sealed, aad("content")).decode()
        embedding = (
            unpack_embedding(open_sealed(key, row.embedding_sealed, aad("embedding")))
            if row.embedding_sealed is not None
            else None
        )
    except (SealedValueError, UnicodeDecodeError):
        return None
    return OpenedEntry(
        id=row.id,
        scope=row.scope,
        owner_user_id=row.owner_user_id,
        author_user_id=row.author_user_id,
        kind=row.kind,
        trust=row.trust,
        version=row.version,
        text=text,
        embedding=embedding,
        embedding_model=row.embedding_model,
        confidence=row.confidence,
        confirmations=row.confirmations,
        flags=row.flags,
        created_at=row.created_at,
    )
