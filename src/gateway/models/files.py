"""ORM tables for the files the Files API stores, and for the copies a provider holds of them."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime


class FileObject(Base):
    """Uploaded file metadata for the OpenAI-compatible files API.

    The raw bytes live in a pluggable blob store; this row holds metadata plus
    the ``storage_ref`` that store minted for them. Files are scoped to
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
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: f"file-{uuid.uuid4().hex}")
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
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None, index=True)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)


class FileProviderCopy(Base):
    """A copy of a stored file that a provider holds, so a provider-native feature can name it.

    Otari's store stays the source of truth.
    The copy is a cache the provider expires on its own, and ``expires_at`` is
    when it stops being usable, so a later request can tell without asking.

    The key identifies the account the copy is in, because a provider file ID
    exists only inside the account of the credential that uploaded it, and a
    request resolving a different credential can neither name that copy nor
    delete it.
    Two things select that credential: the configured instance, and the
    workspace, whose organization may hold a provider key of its own that a bare
    ``provider:model`` selector resolves to.
    """

    __tablename__ = "file_provider_copies"
    # The workspace foreign key cascades, and the primary key indexes it only as
    # a trailing column, so a workspace deletion would scan the table.
    __table_args__ = (Index("ix_file_provider_copies_credential_workspace_id", "credential_workspace_id"),)

    file_id: Mapped[str] = mapped_column(ForeignKey("file_objects.id", ondelete="CASCADE"), primary_key=True)
    provider: Mapped[str] = mapped_column(primary_key=True)
    provider_instance: Mapped[str] = mapped_column(primary_key=True)
    # CASCADE rather than the RESTRICT a file uses: a copy is a cache, and
    # holding up a workspace deletion for one would be the only thing it ever did.
    credential_workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    provider_file_id: Mapped[str] = mapped_column()
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=lambda: datetime.now(UTC))
