"""ORM tables for agent memory: a workspace's data keys, its entries, the proposals in front of them, and their use.

An entry is what agents recall; a proposal is what an agent offered, which the
consolidator turns into an entry or drops. Both keep their text, and an entry its
embedding, sealed under a data key of the workspace's own (``memory_keys``), so
nothing an agent wrote is readable from the database without the deployment's
content key.

Every row names its workspace, and on PostgreSQL each table but the settings is
under row-level security keyed on ``otari.memory_workspace``, so a query that
forgets its workspace filter reads nothing rather than another tenant's rows.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Final, Literal, get_args

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    false,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime

MemoryScope = Literal["workspace", "member"]
MemoryKind = Literal["fact", "decision", "procedure"]
MemoryStatus = Literal["active", "superseded", "retired", "quarantined"]
MemoryTrust = Literal["reviewed", "agent", "external"]
MemorySignalKind = Literal["confirm", "flag"]
MemoryWritePolicy = Literal["auto", "review"]
MemoryProposalState = Literal["pending", "review"]

MEMORY_SCOPES: Final = get_args(MemoryScope)
MEMORY_KINDS: Final = get_args(MemoryKind)
MEMORY_STATUSES: Final = get_args(MemoryStatus)
MEMORY_TRUST_LEVELS: Final = get_args(MemoryTrust)
MEMORY_SIGNAL_KINDS: Final = get_args(MemorySignalKind)
MEMORY_WRITE_POLICIES: Final = get_args(MemoryWritePolicy)
MEMORY_PROPOSAL_STATES: Final = get_args(MemoryProposalState)

# Who wrote a row and how: kept as it was, so provenance outlives the key and the session.
SOURCE_MAX_LENGTH: Final = 255
MODEL_NAME_MAX_LENGTH: Final = 255
FLAG_REASON_MAX_LENGTH: Final = 500


def _one_of(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(value) for value in values)})"


# A member entry belongs to one member, and only a member entry has an owner.
_OWNER_MATCHES_SCOPE = "(scope = 'member') = (owner_user_id IS NOT NULL)"


class MemoryKey(Base):
    """One data key of a workspace, wrapped by the deployment's content key.

    New entries are sealed under the workspace's newest key until it is older than
    ``memory_key_rotation_days``. A key is never changed or deleted on its own: it
    goes with its workspace, and until then it opens what it sealed.
    """

    __tablename__ = "memory_keys"
    __table_args__ = (Index("ix_memory_keys_workspace_created_at", "workspace_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("workspace.id", ondelete="CASCADE"))
    # Which ``ContentKeyPort`` wrapper made it, so the same one opens it.
    wrapper: Mapped[str] = mapped_column(String(16))
    wrapped: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class MemoryEntry(Base):
    """One version of something agents recall.

    A change writes a new version that names the one it replaces, which becomes
    ``superseded``, so every version says why the memory says what it says until
    retention removes it.
    """

    __tablename__ = "memory_entries"
    __table_args__ = (
        CheckConstraint(_one_of("scope", MEMORY_SCOPES), name="check_memory_entries_scope"),
        CheckConstraint(_one_of("kind", MEMORY_KINDS), name="check_memory_entries_kind"),
        CheckConstraint(_one_of("status", MEMORY_STATUSES), name="check_memory_entries_status"),
        CheckConstraint(_one_of("trust", MEMORY_TRUST_LEVELS), name="check_memory_entries_trust"),
        CheckConstraint(_OWNER_MATCHES_SCOPE, name="check_memory_entries_owner"),
        CheckConstraint("version > 0", name="check_memory_entries_version"),
        # What a search opens: the workspace's active entries.
        Index("ix_memory_entries_workspace_status", "workspace_id", "status"),
        # A member's own entries, for their page and their erasure.
        Index(
            "ix_memory_entries_workspace_owner",
            "workspace_id",
            "owner_user_id",
            postgresql_where=text("owner_user_id IS NOT NULL"),
            sqlite_where=text("owner_user_id IS NOT NULL"),
        ),
        # Retention's sweep: the entries nobody has used for longest.
        Index("ix_memory_entries_workspace_last_used_at", "workspace_id", "last_used_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("workspace.id", ondelete="CASCADE"))
    scope: Mapped[MemoryScope] = mapped_column(String(16))
    # Set on member entries only; deleting the member deletes them.
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("user.id", ondelete="CASCADE"))
    # Who proposed it. A workspace entry outlives its author, with the author cleared.
    author_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="SET NULL"), index=True
    )
    kind: Mapped[MemoryKind] = mapped_column(String(16))
    status: Mapped[MemoryStatus] = mapped_column(String(16), default="active", server_default="active")
    trust: Mapped[MemoryTrust] = mapped_column(String(16))
    version: Mapped[int] = mapped_column(default=1, server_default="1")
    # The version this one replaced. Cleared, not cascaded, when that version is deleted.
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("memory_entries.id", ondelete="SET NULL"), index=True
    )
    key_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("memory_keys.id", ondelete="RESTRICT"), index=True)
    content_sealed: Mapped[str] = mapped_column(Text)
    embedding_sealed: Mapped[str | None] = mapped_column(Text)
    # Search compares only vectors one model made, so a change of model leaves older ones out.
    embedding_model: Mapped[str | None] = mapped_column(String(MODEL_NAME_MAX_LENGTH))
    confidence: Mapped[float] = mapped_column(Float)
    confirmations: Mapped[int] = mapped_column(default=0, server_default="0")
    flags: Mapped[int] = mapped_column(default=0, server_default="0")
    # Provenance: the key, session and model the proposal came through. No foreign key, so it outlives them.
    source_api_key_id: Mapped[str | None] = mapped_column(String(SOURCE_MAX_LENGTH))
    source_session_id: Mapped[str | None] = mapped_column(String(SOURCE_MAX_LENGTH))
    source_model: Mapped[str | None] = mapped_column(String(MODEL_NAME_MAX_LENGTH))
    use_count: Mapped[int] = mapped_column(default=0, server_default="0")
    last_used_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now(), onupdate=lambda: datetime.now(UTC)
    )


class MemoryProposal(Base):
    """What an agent offered memory, waiting for the consolidator or an admin.

    Deleted as soon as it is decided, so it holds content only as long as the
    decision takes.
    """

    __tablename__ = "memory_proposals"
    __table_args__ = (
        CheckConstraint(_one_of("scope", MEMORY_SCOPES), name="check_memory_proposals_scope"),
        CheckConstraint(_one_of("kind", MEMORY_KINDS), name="check_memory_proposals_kind"),
        CheckConstraint(_one_of("trust", MEMORY_TRUST_LEVELS), name="check_memory_proposals_trust"),
        CheckConstraint(_one_of("state", MEMORY_PROPOSAL_STATES), name="check_memory_proposals_state"),
        CheckConstraint(_OWNER_MATCHES_SCOPE, name="check_memory_proposals_owner"),
        # The consolidator takes the oldest pending proposals first.
        Index("ix_memory_proposals_state_created_at", "state", "created_at"),
        Index("ix_memory_proposals_workspace_state", "workspace_id", "state"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("workspace.id", ondelete="CASCADE"))
    scope: Mapped[MemoryScope] = mapped_column(String(16))
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("user.id", ondelete="CASCADE"), index=True)
    author_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[MemoryKind] = mapped_column(String(16))
    trust: Mapped[MemoryTrust] = mapped_column(String(16))
    state: Mapped[MemoryProposalState] = mapped_column(String(16), default="pending", server_default="pending")
    key_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("memory_keys.id", ondelete="RESTRICT"), index=True)
    content_sealed: Mapped[str] = mapped_column(Text)
    # Why a proposal waits for an admin, in the consolidator's words: a reason code, never content.
    review_reason: Mapped[str | None] = mapped_column(String(64))
    source_api_key_id: Mapped[str | None] = mapped_column(String(SOURCE_MAX_LENGTH))
    source_session_id: Mapped[str | None] = mapped_column(String(SOURCE_MAX_LENGTH))
    source_model: Mapped[str | None] = mapped_column(String(MODEL_NAME_MAX_LENGTH))
    attempts: Mapped[int] = mapped_column(default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class MemorySignal(Base):
    """One member's word on an entry: it held up in use, or it is wrong or stale.

    One row per member and entry, so a member's latest word replaces their last
    and no one agent can move an entry's confidence alone.
    """

    __tablename__ = "memory_signals"
    __table_args__ = (
        CheckConstraint(_one_of("signal", MEMORY_SIGNAL_KINDS), name="check_memory_signals_signal"),
        UniqueConstraint("entry_id", "user_id", name="uq_memory_signals_entry_user"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), index=True)
    # No index of its own: the unique constraint starts with it.
    entry_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("memory_entries.id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("user.id", ondelete="CASCADE"), index=True)
    signal: Mapped[MemorySignalKind] = mapped_column(String(16))
    # A reason code a flag gives, from a closed list the tool offers; never free text.
    reason: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class MemoryRead(Base):
    """One recall: who read which entries, and when. It keeps no content.

    Names the entries with no foreign key, so the record outlives them, as an
    access record should. It goes with its workspace.
    """

    __tablename__ = "memory_reads"
    __table_args__ = (Index("ix_memory_reads_workspace_read_at", "workspace_id", "read_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("workspace.id", ondelete="CASCADE"))
    reader_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="SET NULL"), index=True
    )
    api_key_id: Mapped[str | None] = mapped_column(String(SOURCE_MAX_LENGTH))
    tool: Mapped[str] = mapped_column(String(64))
    entry_ids: Mapped[list[Any]] = mapped_column(JSON)
    read_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class MemoryOrganizationSettings(Base):
    """An organization's memory switch. No row is the default: memory off in every workspace of it.

    Memory runs in a workspace only while the deployment, its organization and the workspace all have it on,
    so an organization's owners can stop it in every workspace at once.
    """

    __tablename__ = "memory_organization_settings"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="CASCADE"), primary_key=True
    )
    enabled: Mapped[bool] = mapped_column(default=False, server_default=false())
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now(), onupdate=lambda: datetime.now(UTC)
    )


class MemoryWorkspaceSettings(Base):
    """A workspace's memory switches. No row is the default: memory off."""

    __tablename__ = "memory_workspace_settings"
    __table_args__ = (
        CheckConstraint(_one_of("write_policy", MEMORY_WRITE_POLICIES), name="check_memory_settings_write_policy"),
        CheckConstraint("retention_days IS NULL OR retention_days > 0", name="check_memory_settings_retention"),
        CheckConstraint("recall_min_similarity BETWEEN 0.3 AND 0.99", name="check_memory_settings_recall_similarity"),
        CheckConstraint("recall_max_entries BETWEEN 1 AND 5", name="check_memory_settings_recall_entries"),
        CheckConstraint("recall_min_relevance BETWEEN 0.05 AND 0.99", name="check_memory_settings_recall_relevance"),
        CheckConstraint("experiment_holdout_percent BETWEEN 0 AND 50", name="check_memory_settings_experiment_holdout"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    # The workspace's organization, kept beside its switches so memory reads the organization's switch
    # without reaching into tenancy's tables.
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="CASCADE"), index=True
    )
    enabled: Mapped[bool] = mapped_column(default=False, server_default=false())
    # Member entries come with ``enabled``; the shared workspace memory is a second switch.
    workspace_scope_enabled: Mapped[bool] = mapped_column(default=False, server_default=false())
    write_policy: Mapped[MemoryWritePolicy] = mapped_column(String(16), default="review", server_default="review")
    # Offer memory's tools to every request a member sends through the gateway, not only to one that asks.
    offer_to_every_request: Mapped[bool] = mapped_column(default=False, server_default=false())
    # Leave entries whose evidence came from outside (a web page, a fetched file, a third-party tool) out of recall.
    hide_external: Mapped[bool] = mapped_column(default=False, server_default=false())
    # Null keeps the deployment's ``memory_retention_days``.
    retention_days: Mapped[int | None] = mapped_column()
    # Search memory for each request's newest user turn and hand the model what is alike enough, with no tool call.
    auto_recall: Mapped[bool] = mapped_column(default=False, server_default=false())
    recall_min_similarity: Mapped[float] = mapped_column(default=0.75, server_default="0.75")
    recall_max_entries: Mapped[int] = mapped_column(default=3, server_default="3")
    # Have the deployment's rerank model score the entries most alike to the turn, and hand over only those that
    # reach ``recall_min_relevance``; the similarity floor then no longer applies.
    recall_rerank: Mapped[bool] = mapped_column(default=False, server_default=false())
    recall_min_relevance: Mapped[float] = mapped_column(default=0.5, server_default="0.5")
    # The share of sessions the last experiment served without memory, to measure what it saves; zero is none ever.
    experiment_holdout_percent: Mapped[int] = mapped_column(default=0, server_default="0")
    # When it began, which deals the sessions, and when it stopped; it runs while it has begun and not stopped.
    # Both are kept once it stops, so its results still read until another begins.
    experiment_started_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    experiment_ended_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now(), onupdate=lambda: datetime.now(UTC)
    )
