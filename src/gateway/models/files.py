"""ORM tables for the files the Files API stores, and for the copies a provider holds of them."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime


def new_file_id() -> str:
    """Mint the ID of a new file, in the ``file-<hex>`` shape the Files API serves."""
    return f"file-{uuid.uuid4().hex}"


class FileObject(Base):
    """Uploaded file metadata for the OpenAI-compatible files API.

    The raw bytes live in a pluggable blob store; this row holds metadata plus
    the ``storage_ref`` that store minted for them, which is allocated before
    the bytes are written so a partial write is always named by a row.
    ``pending_since`` is set while those bytes are still owed. Files are scoped to
    ``user_id`` for tenant isolation and soft-deleted via ``deleted_at``.
    ``workspace_id`` is a second, independent axis: it says which workspace the
    upload was made in, so a key confined to one workspace never reaches
    another's files even when the same user holds keys in both.
    """

    __tablename__ = "file_objects"
    __table_args__ = (
        # The listing's shape: the tenant predicates, then the keyset sort the
        # cursor pages on. Without them every page sorts the user's whole set;
        # the second serves a master-key listing that names no workspace.
        Index(
            "ix_file_objects_user_workspace_created",
            "user_id",
            "workspace_id",
            "created_at",
            "id",
        ),
        Index("ix_file_objects_user_created", "user_id", "created_at", "id"),
        # The sweep's pending arm. Partial, so it holds only the rows still
        # waiting for their bytes rather than one entry per file.
        Index(
            "ix_file_objects_pending_since",
            "pending_since",
            postgresql_where=text("pending_since IS NOT NULL"),
            sqlite_where=text("pending_since IS NOT NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_file_id)
    # Always set to the authenticated user; non-null enforces the user-scoping
    # contract at the schema level. CASCADE removes a user's files on delete.
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    # The workspace this row belongs to; see `APIKey.workspace_id` for why it is
    # NOT NULL and RESTRICT rather than nullable and cascading. Existing rows were
    # backfilled onto the deployment's default workspace, which is also where a
    # master-key upload lands.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column()
    mime_type: Mapped[str] = mapped_column()
    bytes: Mapped[int] = mapped_column()
    purpose: Mapped[str] = mapped_column(default="user_data")
    storage_ref: Mapped[str | None] = mapped_column(nullable=True)
    # Set when a provider's own sandbox produced the file. The three provider
    # columns record where it came from; nothing on the read path uses them.
    provider: Mapped[str | None] = mapped_column(nullable=True)
    provider_instance: Mapped[str | None] = mapped_column(nullable=True)
    provider_container_id: Mapped[str | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    # Set when the row is reserved and cleared once its bytes land. No read
    # serves a pending row, and the sweep reclaims one once it is stale.
    pending_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None, index=True)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)


class FileProviderCopy(Base):
    """A copy of a stored file that a provider holds, so a provider-native feature can name it.

    Otari's store stays the source of truth.
    The copy is a cache the provider expires on its own, and ``expires_at`` is
    when it stops being usable, so a later request can tell without asking.

    A row is recorded before its copy is uploaded, so every copy Otari makes is
    named by a row, and ``pending_since`` is set until the upload is confirmed.
    Each copy has its own row, so two requests copying one file at once each
    record the copy they made.

    ``account_identity`` says which provider account holds the copy, because a
    provider file ID exists only inside that account.
    ``provider_instance`` and ``credential_workspace_id`` say how that account's
    credential is found again, which the identity cannot say.
    """

    __tablename__ = "file_provider_copies"
    __table_args__ = (
        CheckConstraint(
            "(pending_since IS NULL AND provider_file_id IS NOT NULL AND expires_at IS NOT NULL)"
            " OR (pending_since IS NOT NULL AND provider_file_id IS NULL)",
            name="ck_file_provider_copies_pending_or_confirmed",
        ),
        Index("ix_file_provider_copies_file_account", "file_id", "account_identity"),
        # The workspace foreign key cascades, and no other index leads with it.
        Index("ix_file_provider_copies_credential_workspace_id", "credential_workspace_id"),
        Index(
            "ix_file_provider_copies_pending_since",
            "pending_since",
            postgresql_where=text("pending_since IS NOT NULL"),
            sqlite_where=text("pending_since IS NOT NULL"),
        ),
        Index(
            "ix_file_provider_copies_expires_at",
            "expires_at",
            postgresql_where=text("pending_since IS NULL"),
            sqlite_where=text("pending_since IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    file_id: Mapped[str] = mapped_column(ForeignKey("file_objects.id", ondelete="CASCADE"))
    account_identity: Mapped[str] = mapped_column()
    provider: Mapped[str] = mapped_column()
    provider_instance: Mapped[str] = mapped_column()
    # CASCADE rather than the RESTRICT a file uses: a copy is a cache, and
    # holding up a workspace deletion for one would be the only thing it ever did.
    credential_workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("workspace.id", ondelete="CASCADE"))
    provider_file_id: Mapped[str | None] = mapped_column(default=None)
    expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    pending_since: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=lambda: datetime.now(UTC))
