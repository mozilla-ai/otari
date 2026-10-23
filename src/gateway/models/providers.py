"""ORM tables for providers: runtime instances, owned endpoints, hosted providers, and model aliases."""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, UniqueConstraint, Uuid, false, text
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
    """A provider instance the retired Deployment providers page stored.

    Nothing serves on it: at startup each row is moved onto a hosted provider of
    the same name and removed (``migrate_deployment_providers`` in
    ``services/providers``). A row the move cannot take stays, so the Hosted
    providers page can say why. The API key is held encrypted (``secret_box``).
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
        ``OrgProviderKey.to_public`` masks its own: a Bedrock instance
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


class ProviderEndpoint(Base):
    """A provider endpoint a workspace or one of its users brings, addressed by name.

    An app built on otari lets its users bring their own model server: a base
    URL, an API key, and sometimes fields the server wants in every request. A
    caller reaches one as ``<name>:<model>``, which resolves only for its owner,
    ahead of the deployment's own instances (``services/provider_kwargs``).

    Disjoint from ``org_provider_keys`` by construction: those resolve by
    provider for every workspace in an organization, these resolve by name for
    one workspace or one user in it, so no query over one table can pick up a
    row of the other.

    Owned the way ``ModelAlias`` is: ``workspace_id`` always, ``user_id`` (the
    billing identity) when the endpoint belongs to one user, who then shadows a
    workspace-wide endpoint of the same name. The same pair of constraints keeps
    one row per scope, for the reason given there.

    The owner pays the upstream, so requests to one never count toward a budget.
    """

    __tablename__ = "provider_endpoints"
    __table_args__ = (
        UniqueConstraint("workspace_id", "name", "user_id", name="uq_provider_endpoints_workspace_name_user"),
        Index(
            "uq_provider_endpoints_workspace_shared_name",
            "workspace_id",
            "name",
            unique=True,
            sqlite_where=text("user_id IS NULL"),
            postgresql_where=text("user_id IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # No index of its own: both constraints above lead with it. CASCADE, as on
    # ``org_provider_keys``: a credential means nothing once its owner is gone.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    # The any-llm implementation, which picks the wire; not a routing key.
    provider: Mapped[str] = mapped_column(String(64))
    api_base: Mapped[str] = mapped_column(String(1024))
    encrypted_api_key: Mapped[str | None] = mapped_column()
    last4: Mapped[str | None] = mapped_column(String(8))
    # Fields sent in every request body beneath the caller's own.
    default_params: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class HostedProvider(Base):
    """A provider this deployment serves hosted inference on, and the key it serves it with.

    The deployment's own credential for a request that brings none, asked for
    only after every rung upstream of it has missed. ``name`` is what a request,
    the roster, the price list and usage know it by; ``provider`` is the any-llm
    implementation that serves it. The name defaults to the implementation, and
    a different one lets two endpoints of one implementation coexist
    (``openai`` and ``openai-eu``).

    The key and the SDK client extras are both stored encrypted and read back
    in one place. ``api_key_last4`` is kept in clear so a listing can tell rows
    apart without decrypting every credential. A keyless backend stores no key.
    """

    __tablename__ = "hosted_providers"
    __table_args__ = (UniqueConstraint("name", name="uq_hosted_providers_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # The implementation's own name unless one is given.
    name: Mapped[str] = mapped_column(String(64), default=lambda context: context.get_current_parameters()["provider"])
    provider: Mapped[str] = mapped_column(String(64))
    encrypted_api_key: Mapped[str | None] = mapped_column()
    api_key_last4: Mapped[str | None] = mapped_column(String(4))
    api_base: Mapped[str | None] = mapped_column(String(1024))
    # Bedrock's region and IAM pair, as one encrypted JSON document: an IAM
    # secret in there is a credential, and masking it on the way out does not
    # protect a copy of the database.
    encrypted_client_args: Mapped[str | None] = mapped_column()
    enabled: Mapped[bool] = mapped_column(default=True)
    # Serve a model the roster does not list, as a ``config.yml`` instance does.
    # A model the roster switches off is still refused.
    serves_unlisted_models: Mapped[bool] = mapped_column(default=False, server_default=false())
    # The one organization served on this provider, or None for every one. Any
    # other organization is answered as if the provider were not configured.
    # RESTRICT: SET NULL would open the provider to every organization, and
    # CASCADE would destroy the operator's credential with it, so deleting the
    # organization fails until the restriction is lifted or the provider removed.
    restricted_to_organization_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="RESTRICT"), index=True, default=None
    )
    # The ISO country codes whose requests this provider serves, or None for
    # every country. Checked as well as ``provider_supported_countries``.
    supported_countries: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None, onupdate=lambda: datetime.now(UTC)
    )


HostedModelDisabledReason = Literal["unpriced", "operator"]
DISABLED_UNPRICED: HostedModelDisabledReason = "unpriced"
DISABLED_BY_OPERATOR: HostedModelDisabledReason = "operator"


class HostedProviderModel(Base):
    """One model the deployment offers on a hosted provider.

    Membership and a serving switch. The rate lives in ``model_pricing`` keyed
    ``name:model``, so this table cannot drift from what a request settles
    at. ``provider`` holds the hosted provider's name rather than a foreign
    key, because a hosted provider may also be declared in configuration with
    no row of its own, and its roster still lives here.

    Dispatch serves only a model whose row is switched on: a row switched
    off is refused, and so is a model with no row unless the provider
    ``serves_unlisted_models``.

    ``disabled_reason`` says why a row is off, and is null exactly when it is
    on: ``unpriced`` while it waits for a rate, ``operator`` while an operator
    has switched it off. ``serves_unpriced`` marks a row switched on without a
    rate on purpose, so a refresh does not hold it for one.
    """

    __tablename__ = "hosted_provider_models"
    __table_args__ = (UniqueConstraint("provider", "model", name="uq_hosted_provider_models_provider_model"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(255))
    enabled: Mapped[bool] = mapped_column(default=True)
    disabled_reason: Mapped[HostedModelDisabledReason | None] = mapped_column(String(16), default=None)
    serves_unpriced: Mapped[bool] = mapped_column(default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None, onupdate=lambda: datetime.now(UTC)
    )


class ConfiguredProviderImport(Base):
    """A ``providers:`` entry of config.yml already copied onto a hosted provider of its name.

    Recorded so the entry is read once: from then on the hosted provider is
    what serves the name, and removing it on the Hosted providers page is not
    undone by the next start.
    """

    __tablename__ = "configured_provider_imports"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
