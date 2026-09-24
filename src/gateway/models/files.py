"""ORM table for the files the Files API stores, and the metadata that outlives their bytes."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base


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
