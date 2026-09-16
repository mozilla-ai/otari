"""ORM tables for providers: provider instances configured at runtime, and model aliases."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, UniqueConstraint, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base
from gateway.models.secret_fields import redact_secret_like_values


class ModelAlias(Base):
    """A display name that resolves to a real model selector.

    The runtime counterpart of the ``aliases:`` block in config.yml: same
    meaning, but writable through the API. Pricing, budgets, and usage all key
    on the resolved target, so nothing here is billed against ``name``.

    There are two scopes, and they are independent. ``workspace_id`` says which
    tenant owns the alias: it resolves only for requests in that workspace, so
    two workspaces can each point ``fast`` somewhere different. Within a
    workspace, ``user_id`` narrows it further: ``NULL`` means every caller in
    that workspace sees it, which is what every row predating the column is, and
    a non-null ``user_id`` scopes it to that user, shadowing the workspace-wide
    row of the same name for them alone.

    Uniqueness needs two constraints rather than one because SQLite and
    PostgreSQL both treat NULLs as distinct in a unique index: the composite
    constraint keeps one row per (workspace, name, user), and the partial index
    keeps one workspace-wide row per (workspace, name), which the composite one
    cannot, its ``user_id`` being NULL. The surrogate ``id`` exists only because
    the natural key contains a nullable column, which a primary key cannot.
    """

    __tablename__ = "model_aliases"
    __table_args__ = (
        # Workspace-scoped, so two workspaces can each hold a "fast" entry
        # pointing somewhere different. Safe only because resolution is keyed by
        # workspace too (``services/alias_service``); while that cache was keyed
        # on name alone the second workspace's row silently shadowed the first at
        # request time, which is why this constraint waited for it.
        UniqueConstraint("workspace_id", "name", "user_id", name="uq_model_aliases_workspace_name_user"),
        Index(
            "uq_model_aliases_workspace_global_name",
            "workspace_id",
            "name",
            unique=True,
            sqlite_where=text("user_id IS NULL"),
            postgresql_where=text("user_id IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    # No index of its own: both constraints above lead with `workspace_id` and
    # carry `name` second, and a listing is always workspace-scoped. A third copy
    # would be paid for on every write to serve reads that mostly do not happen,
    # since resolution goes through the process-wide alias cache.
    name: Mapped[str] = mapped_column()
    target: Mapped[str] = mapped_column()
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    # The workspace this row belongs to; see `APIKey.workspace_id` for why.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ProviderCredential(Base):
    """A provider instance configured at runtime through the dashboard.

    The database counterpart of a ``providers:`` entry in config.yml: it is
    merged over the config-file providers at runtime (see
    ``provider_store_service``), with the stored row winning on an instance-name
    collision. The API key is held encrypted (``secret_box``); ``last4`` is kept
    in clear only so the UI can show which key is set without ever decrypting.
    Standalone mode only, never used in the hybrid platform path.
    """

    __tablename__ = "provider_credentials"

    instance: Mapped[str] = mapped_column(primary_key=True)
    provider_type: Mapped[str | None] = mapped_column()
    api_base: Mapped[str | None] = mapped_column()
    encrypted_api_key: Mapped[str | None] = mapped_column()
    last4: Mapped[str | None] = mapped_column()
    client_args: Mapped[dict[str, Any]] = mapped_column("client_args", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize for the API. Never includes the secret, only ``last4``.

        ``client_args`` is masked by key name the same way
        ``OrgProviderKey.to_public`` masks its own: a standalone Bedrock instance
        keeps its ``aws_secret_access_key`` there, so the field this table holds
        in clear is as much a credential as ``encrypted_api_key`` is, and it must
        not round-trip over the API either.
        """
        return {
            "instance": self.instance,
            "provider_type": self.provider_type,
            "api_base": self.api_base,
            "last4": self.last4,
            "client_args": redact_secret_like_values(self.client_args) or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
