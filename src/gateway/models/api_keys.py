"""ORM tables for API keys and the federation rules that let a workload authenticate as one."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, String, Uuid, false, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gateway.models.base import Base, UtcDateTime

# How many budgets one key may list for its end users.
MAX_END_USER_BUDGETS = 100
# How many federation rules one key may hold.
MAX_FEDERATION_RULES_PER_KEY = 20


class APIKey(Base):
    """API Key model for authentication and authorization."""

    __tablename__ = "api_keys"

    __table_args__ = (
        # One row per config.yml entry, which is what lets replicas starting at
        # once insert it without a second copy: the loser's insert conflicts here.
        # NULLs are distinct on both engines, so every other key is unconstrained.
        Index("uq_api_keys_config_name", "config_name", unique=True),
    )

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
    # True for a key a member minted for themselves on the member key surface,
    # false for one an operator issued or config.yml declared, NULL where the
    # key predates the column and its origin is unrecorded. Demoting a member
    # to viewer switches off only the true ones.
    self_service: Mapped[bool | None] = mapped_column(default=None)
    # True while the key is off because an operator, or a removal from its
    # workspace or organization, switched it off; its owner may not switch it
    # back on. Cleared when an operator switches it on.
    operator_disabled: Mapped[bool] = mapped_column(default=False, server_default=false())
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
    # Per-key search tool allow-list: NULL = inherit the user's, then the tools
    # allowed_models names the deprecated way (see services/model_access.py),
    # [] = no tool, a list = tool names, '*' for every search tool.
    allowed_tools: Mapped[list[str] | None] = mapped_column(JSON)
    # A service key may name end users in a request's ``user`` field. Each one is
    # a ``users`` row owned by this key's user (``users.parent_user_id``), created
    # on first use, so a key can only ever bill end users in its owner's scope.
    # This key's own ceiling (an ``api_token`` scoped budget) is the pool over all
    # of them.
    is_service_key: Mapped[bool] = mapped_column(default=False, server_default=false())
    # The budget an end user this key creates is capped at. NULL leaves end users
    # uncapped, under the pool alone. Changing it does not move end users already
    # created; each one keeps the budget it was given, which the users API can
    # override one at a time.
    end_user_budget_id: Mapped[str | None] = mapped_column(
        ForeignKey("budgets.budget_id", ondelete="SET NULL"), index=True
    )
    # The budgets a request on this key may start a new end user on, named per
    # request with ``Otari-End-User-Budget``. NULL means the default alone. Kept
    # as a list rather than a join table because it is read only when a request
    # names a budget; deleting a budget removes it from every list.
    end_user_budget_ids: Mapped[list[str] | None] = mapped_column(JSON)
    # The most live end users this key may have on the budgets it assigns: a
    # request that would create or revive one more is refused. NULL is no cap.
    max_end_users: Mapped[int | None] = mapped_column(default=None)

    # The name config.yml declares this key under, set only while it does. Every
    # start finds the key by it, and writes the declared settings and secret back
    # when the entry's digest differs from ``declared_digest``.
    config_name: Mapped[str | None] = mapped_column(default=None)
    # A digest of the config.yml entry last written over this key, its secret's
    # hash included, so changing the entry or the secret writes it back.
    declared_digest: Mapped[str | None] = mapped_column(String(64), default=None)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    user = relationship("User", back_populates="api_keys")
    usage_logs = relationship("UsageLog", back_populates="api_key", passive_deletes=True)

    def assignable_end_user_budgets(self) -> list[str]:
        """The budgets this key may start an end user on: its list, or else its default alone."""
        if self.end_user_budget_ids is not None:
            return list(self.end_user_budget_ids)
        return [self.end_user_budget_id] if self.end_user_budget_id else []

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
            "allowed_tools": self.allowed_tools,
            "is_service_key": self.is_service_key,
            "end_user_budget_id": self.end_user_budget_id,
            "end_user_budget_ids": self.assignable_end_user_budgets(),
            "max_end_users": self.max_end_users,
            "config_name": self.config_name,
            "metadata": self.metadata_,
        }


class APIKeyFederationRule(Base):
    """An OpenID Connect identity that authenticates as an API key in place of its secret.

    A verified token whose issuer and audience equal the rule's, and whose
    subject and verified email equal each of the rule's that is set, is that key.
    Issuer and audience alone are never enough, since anyone can mint a token
    for any audience at a public issuer, so a rule names a subject or an email.

    An operator's rule names an issuer on ``federation_issuers`` and matches
    wherever a token is presented. A member's rule (``member_managed``) names
    any public issuer and matches only in the token exchange that names it by
    id, so nothing it says is consulted when a bare identity token is looked up
    across every key.
    """

    __tablename__ = "api_key_federation_rules"
    __table_args__ = (
        CheckConstraint("subject IS NOT NULL OR email IS NOT NULL", name="ck_api_key_federation_rules_principal"),
        # Authentication reads the rules by the token's issuer and audiences.
        Index("ix_api_key_federation_rules_issuer_audience", "issuer", "audience"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    api_key_id: Mapped[str] = mapped_column(
        String, ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False, index=True
    )
    issuer: Mapped[str] = mapped_column(String, nullable=False)
    audience: Mapped[str] = mapped_column(String, nullable=False)
    subject: Mapped[str | None] = mapped_column(String, default=None)
    # Stored lower case; compared with the token's email only when the token says it is verified.
    email: Mapped[str | None] = mapped_column(String, default=None)
    member_managed: Mapped[bool] = mapped_column(default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )
