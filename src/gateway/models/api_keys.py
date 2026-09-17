"""ORM table for API keys."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gateway.models.base import Base


class APIKey(Base):
    """API Key model for authentication and authorization."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(primary_key=True)
    key_hash: Mapped[str] = mapped_column(unique=True, index=True)
    # NOT NULL: every row belongs to a workspace. RESTRICT: deleting a workspace must
    # not silently delete its keys, usage, aliases, and policies.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    # Display-only leading characters of the plaintext key, kept so the dashboard can
    # recognize a key after its one-time reveal. Nullable: keys minted before this
    # column existed cannot be back-filled (the plaintext is unrecoverable).
    key_prefix: Mapped[str | None] = mapped_column()
    # Display-only trailing characters, stored so the dashboard can tell two keys
    # apart when they share a prefix. Nullable for the same reason as ``key_prefix``
    # and for one more: every key minted before this column will show prefix-only
    # forever, because the plaintext is unrecoverable. Named ``key_suffix`` rather
    # than the ``last4`` its provider-credential counterpart uses, so that the pair
    # on this table reads as a pair.
    key_suffix: Mapped[str | None] = mapped_column()
    key_name: Mapped[str | None] = mapped_column()
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(default=True)
    # When true, requests authenticated with this key are logged with their computed
    # cost but skip budget reservation/reconciliation: their spend is never written to
    # User.spend and never gates enforcement. Default false keeps every existing key
    # (and all keys minted before this column) on the normal enforced path.
    exclude_from_budget: Mapped[bool] = mapped_column(default=False)
    # Per-key override of the deployment-wide ``reject_user_mismatch`` setting.
    # NULL = inherit (the default, and where every key predating this column
    # stays), True = always reject a request naming a different ``user``, False =
    # always accept it. The override only decides the 403: spend binds to this
    # key's own user either way, so the client value stays a provider-side tag.
    # False is for clients whose ``user`` is telemetry rather than an identity
    # (Claude Code sends a per-session JSON blob); True lets a deployment that
    # relaxed the check globally keep an individual key strict.
    reject_user_mismatch: Mapped[bool | None] = mapped_column(default=None)
    # Per-key override of the deployment-wide ``capture_agent_telemetry`` setting.
    # NULL = inherit (the default), True = always store behavioral events from
    # this key, False = always discard them. Usage capture/billing is unaffected
    # either way; this only gates the content-free agent_telemetry row.
    capture_agent_telemetry: Mapped[bool | None] = mapped_column(default=None)
    # Per-key model allow-list. NULL = unrestricted (default; every key predating
    # this column stays unrestricted), [] = deny all, a list = canonical
    # instance:model entries (with instance:* / instance:prefix* wildcards).
    allowed_models: Mapped[list[str] | None] = mapped_column(JSON)

    # Set only on a key this deployment mints for itself and has to present
    # again later, which today is the one the hosted Playground forwards with
    # (``services/playground_dispatch``). Holds that key's plaintext encrypted
    # with ``secret_box``, because presenting a credential is the one thing the
    # hash above cannot do. NULL on every key a person created, which is what the
    # key listings filter on: a row with a value here is machinery rather than a
    # credential anybody manages, and no endpoint ever returns the value itself.
    internal_secret: Mapped[str | None] = mapped_column()

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    user = relationship("User", back_populates="api_keys")
    usage_logs = relationship("UsageLog", back_populates="api_key", passive_deletes=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "key_prefix": self.key_prefix,
            "key_suffix": self.key_suffix,
            "key_name": self.key_name,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "exclude_from_budget": self.exclude_from_budget,
            "reject_user_mismatch": self.reject_user_mismatch,
            "capture_agent_telemetry": self.capture_agent_telemetry,
            "allowed_models": self.allowed_models,
            "metadata": self.metadata_,
        }
