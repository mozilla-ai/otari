"""Organization-scoped provider keys.

Decided at otari-ai#1748: the platform's ``ProviderKey`` shape
(organization scope, archival, one default per organization+provider,
per-workspace pin/disable overrides, per-workspace model allow-lists) ports
into otari as new, additive tables. ``provider_credentials`` and its
config.yml-merge overlay (``services/provider_store_service.py``) are
unchanged: they stay the mechanism for config.yml-defined and legacy
deployment-global stored credentials, addressed by instance name exactly as
before. These two mechanisms are disjoint by construction (see
``services/provider_kwargs.py``): an ``instance:model`` selector that matches
an existing ``config.providers`` entry never consults these tables, and a
bare ``provider:model`` selector never consults ``config.providers`` for a
workspace that has an org-scoped key. See mozilla-ai/otari#643.

Four tables, named to avoid a collision that already exists in this
codebase: ``ScopedBudget.provider_key_id`` (`models/budgets.py`) already
means "an instance-name string, no FK". These tables use ``org_provider_key``
throughout so no column here is ever ambiguously named ``provider_key_id``.

- ``OrgProviderKey`` (``org_provider_keys``): one BYO credential, scoped to an
  organization. otari-ai's ``ProviderKey`` also carries a "managed/phantom"
  bucket (a platform-hosted upstream credential with no stored ``api_key``);
  that concept has no otari-side equivalent (managed credentials are hosted
  depth) and is dropped entirely here, so ``is_org_default`` is one flag per
  ``(organization_id, provider)`` rather than per bucket.
- ``WorkspaceProviderKeyOverride`` (``workspace_provider_key_overrides``): a
  workspace's departure from its organization's default for one key. Absence
  of a row means full inheritance; a row pins the key as this workspace's
  default (``is_default``), opts the workspace out of it (``disabled``), or
  both fields are their default and the row is meaningless (the service layer
  deletes it rather than storing a no-op).
- ``WorkspaceProviderModelRestriction`` (``workspace_provider_model_restrictions``):
  a per-workspace, per-key model allow-list. No rows for a
  ``(workspace, key)`` pair means every model is allowed; one or more rows
  narrows it to exactly those.
- ``OrgProviderKeyModel`` (``org_provider_key_models``): the models the
  organization offers on one key, each with a serving switch. Absent rows mean
  the key is unnarrowed, the same convention as the table above; present rows
  narrow the organization to the enabled ones, in the catalog and at dispatch
  alike. The rate is not here: it lives in ``organization_model_pricing``.

Style follows ``models/tenancy.py``: SQLModel (not the declarative ``Base``
style) because these are tenancy-scoped tables sharing the same mixins and
``UtcDateTime`` timestamp handling, and no ``relationship()`` is declared
(lazy loading raises ``MissingGreenlet`` on an ``AsyncSession``); repositories
join explicitly.

CASCADE, not the ``RESTRICT`` default `AGENTS.md` states for a gateway table
gaining tenancy scope, is deliberate here: these four tables are org- and
workspace-*owned* resources, like ``organization_member``/``workspace_member``
(CASCADE), not durable request-plane history like ``usage_logs``/``api_keys``
(RESTRICT, so a workspace delete cannot silently take budgets or usage with
it). A credential or its overrides have no meaning once the organization or
workspace that owns them is gone.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Column, ForeignKeyConstraint, Index, UniqueConstraint, text, true
from sqlmodel import Field, SQLModel

from gateway.models.base import CreatedAtMixin, PrimaryKeyMixin, UpdatedAtMixin, _timestamp_field
from gateway.models.pricing import PriceSource
from gateway.models.secret_fields import redact_secret_like_values

# ``client_args`` is arbitrary JSON, and this gateway's own Bedrock support is
# the reason a credential-shaped entry in it cannot simply be rejected outright:
# standalone mode's classic AWS IAM shape genuinely requires
# ``aws_access_key_id`` and ``aws_secret_access_key`` inside ``client_args``
# (any-llm-sdk's ``BedrockProvider`` never forwards ``api_key`` into the boto3
# client it builds; see ``services/bedrock_gateway_auth.py``), so those are real
# credentials this field is *supposed* to carry, not smuggled duplicates of
# ``encrypted_api_key``. They still must never round-trip over the API, the same
# treatment ``encrypted_api_key`` already gets (only ``last4`` comes back);
# ``redact_secret_like_values`` is that treatment applied by key name rather
# than by field.

# ==============================================================================
# Org provider keys
# ==============================================================================


class OrgProviderKeyCreateRequest(SQLModel):
    """What a caller sends to create a key.

    The plaintext key is never stored as sent: the service encrypts it
    (`services/secret_box.py`) and keeps only the ciphertext and ``last4``,
    the same convention `providers.ProviderCredential` already uses.
    """

    provider: str = Field(max_length=255)
    name: str = Field(max_length=255)
    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None, max_length=1024)
    client_args: dict[str, Any] | None = None


class OrgProviderKeyUpdateRequest(SQLModel):
    """A partial update. Every field is optional; only what is set is applied."""

    name: str | None = Field(default=None, max_length=255)
    api_key: str | None = None
    api_base: str | None = Field(default=None, max_length=1024)
    client_args: dict[str, Any] | None = None


class OrgProviderKeyPublic(SQLModel):
    """The API-facing shape. Never carries the key, only whether one is set."""

    id: uuid.UUID
    organization_id: uuid.UUID
    provider: str
    name: str
    api_base: str | None = None
    client_args: dict[str, Any] | None = None
    last4: str | None = None
    is_org_default: bool
    usable: bool = Field(
        description=(
            "False when the stored credential cannot be decrypted on this deployment, so the key "
            "supplies nothing at dispatch and the catalog withholds its provider. A row this deployment "
            "cannot read is still listed, because deleting or replacing it is what fixes it."
        )
    )
    archived_at: datetime | None = None
    created_at: datetime
    updated_at: datetime | None = None


class OrgProviderKeysPublic(SQLModel):
    data: list[OrgProviderKeyPublic]
    count: int


class OrgProviderKey(SQLModel, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    """One organization-scoped, BYO provider credential."""

    __tablename__ = "org_provider_keys"
    __table_args__ = (
        UniqueConstraint("organization_id", "provider", "name", name="uq_org_provider_keys_org_provider_name"),
        # One default per (organization, provider). Enforced at the database
        # rather than only in the service layer so a race between two
        # concurrent "set default" calls has a real arbiter instead of a
        # last-write-wins column; `OrgProviderKeyRepository.set_org_default`
        # catches the resulting `IntegrityError`.
        Index(
            "uq_org_provider_keys_org_default",
            "organization_id",
            "provider",
            unique=True,
            postgresql_where=text("is_org_default AND archived_at IS NULL"),
            sqlite_where=text("is_org_default AND archived_at IS NULL"),
        ),
        # Covers (organization_id, id) so the two link tables below can carry a
        # composite FK to it, pinning each link row to *its own* organization
        # rather than trusting every write path to keep that invariant.
        UniqueConstraint("organization_id", "id", name="uq_org_provider_keys_org_id"),
    )

    organization_id: uuid.UUID = Field(foreign_key="organization.id", ondelete="CASCADE", index=True)
    provider: str = Field(max_length=255)
    name: str = Field(max_length=255)
    api_base: str | None = Field(default=None, max_length=1024)
    client_args: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    encrypted_api_key: str | None = Field(default=None)
    last4: str | None = Field(default=None, max_length=8)
    # _timestamp_field, not a bare Field(default=None): PostgreSQL renders a
    # plain ``datetime`` column as TIMESTAMP WITHOUT TIME ZONE, which asyncpg
    # then refuses a timezone-aware value against (the same trap
    # ``UtcDateTime``'s own docstring describes for the tenancy timestamps).
    archived_at: datetime | None = _timestamp_field(default=None, column_kwargs={})
    is_org_default: bool = Field(default=False, nullable=False)

    def to_public(self, *, usable: bool) -> OrgProviderKeyPublic:
        """Serialize for the API. Never includes the key, only ``last4``.

        ``usable`` is passed in rather than derived here: answering it means
        decrypting, which belongs to the service layer that owns the secret box.
        Required rather than defaulted so a caller cannot report a key as working
        without having asked.

        ``client_args`` is arbitrary JSON an admin can set (Bedrock's
        ``region_name``, other client kwargs), and a credential-shaped field
        placed there is never echoed back either: ``redact_secret_like_values``
        masks it the same way ``encrypted_api_key`` itself already stays off
        the wire (only ``last4`` comes back). That masking is the whole
        protection the field gets, and it is enough because it holds for every
        reader: this surface has one audience, the organization owners and
        admins each of its routes is gated on.
        """
        return OrgProviderKeyPublic(
            id=self.id,
            organization_id=self.organization_id,
            provider=self.provider,
            name=self.name,
            api_base=self.api_base,
            client_args=redact_secret_like_values(self.client_args),
            last4=self.last4,
            is_org_default=self.is_org_default,
            usable=usable,
            archived_at=self.archived_at,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


# ==============================================================================
# Workspace overrides
# ==============================================================================


class WorkspaceProviderKeyOverrideRequest(SQLModel):
    """Tri-state: an omitted field leaves that flag unchanged.

    Both fields false, whether that is the merged result or a value sent
    explicitly, is a no-op the service deletes rather than stores: absence of
    a row already means full inheritance from the organization default.
    Setting one true auto-resolves the other when they would otherwise
    conflict (pinning re-enables a disabled key; disabling un-pins a pinned
    one); sending both true explicitly is refused.
    """

    is_default: bool | None = None
    disabled: bool | None = None


class WorkspaceProviderKeyOverridePublic(SQLModel):
    """The effective view for one workspace+key: raw override flags plus the resolution."""

    workspace_id: uuid.UUID
    org_provider_key_id: uuid.UUID
    is_default: bool
    disabled: bool
    is_effective_default: bool
    # Whether this workspace disabled the key, and nothing more. A key it left
    # alone reads as enabled even when the deployment cannot decrypt it, which is
    # what `usable` is for: the two answer different questions and a caller
    # deciding whether the key will serve needs both.
    is_effective_enabled: bool
    usable: bool
    # Carried on the row rather than left to the per-key route: a caller
    # summarizing a workspace wants the narrowing alongside the flags, and
    # fetching it per key turns one read into one per key. Empty is the common
    # answer and means every model the key serves, never none of them.
    allowed_models: list[str]


class WorkspaceProviderKeyOverridesPublic(SQLModel):
    data: list[WorkspaceProviderKeyOverridePublic]


class WorkspaceProviderKeyOverride(SQLModel, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    """A workspace's departure from its organization's default for one key."""

    __tablename__ = "workspace_provider_key_overrides"
    __table_args__ = (
        UniqueConstraint("workspace_id", "org_provider_key_id", name="uq_workspace_provider_key_overrides_ws_key"),
        # Composite, not a plain FK on org_provider_key_id alone: pins the
        # referenced key to *this row's own* organization_id (see
        # `OrgProviderKey`'s matching unique constraint), so a cross-organization
        # override (a workspace in org A pointing at org B's key) is a
        # foreign-key violation rather than a silently-persisted row. The
        # service layer already only ever sets `organization_id` from the
        # workspace it resolved, so this never disagrees with a legitimate write.
        ForeignKeyConstraint(
            ["organization_id", "org_provider_key_id"],
            ["org_provider_keys.organization_id", "org_provider_keys.id"],
            ondelete="CASCADE",
        ),
    )

    workspace_id: uuid.UUID = Field(foreign_key="workspace.id", ondelete="CASCADE", index=True)
    # Denormalized from the workspace's own organization; see the composite FK
    # above for why it is stored rather than joined at read time.
    organization_id: uuid.UUID
    org_provider_key_id: uuid.UUID = Field(index=True)
    is_default: bool = Field(default=False, nullable=False)
    disabled: bool = Field(default=False, nullable=False)


# ==============================================================================
# Workspace model restrictions
# ==============================================================================


class WorkspaceProviderModelRestrictionRequest(SQLModel):
    model: str = Field(max_length=255)


class WorkspaceProviderModelRestrictionsPublic(SQLModel):
    models: list[str]


class WorkspaceProviderModelRestriction(SQLModel, PrimaryKeyMixin, CreatedAtMixin, table=True):
    """One allowed model for a workspace+key pair.

    No rows for a pair means every model is allowed; this is an allow-list,
    not a deny-list, so adding the first row narrows rather than widens.
    """

    __tablename__ = "workspace_provider_model_restrictions"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "org_provider_key_id",
            "model",
            name="uq_workspace_provider_model_restrictions_ws_key_model",
        ),
        # Same reasoning as `WorkspaceProviderKeyOverride`'s matching constraint.
        ForeignKeyConstraint(
            ["organization_id", "org_provider_key_id"],
            ["org_provider_keys.organization_id", "org_provider_keys.id"],
            ondelete="CASCADE",
        ),
    )

    workspace_id: uuid.UUID = Field(foreign_key="workspace.id", ondelete="CASCADE", index=True)
    # Denormalized from the workspace's own organization; see the composite FK
    # above for why it is stored rather than joined at read time.
    organization_id: uuid.UUID
    org_provider_key_id: uuid.UUID = Field(index=True)
    model: str = Field(max_length=255)


# ==============================================================================
# Offered models
# ==============================================================================


class OrgProviderKeyModelCreateRequest(SQLModel):
    """Offer one model on a key, for a backend whose models cannot be listed."""

    model: str = Field(max_length=255)


class OrgProviderKeyModelUpdateRequest(SQLModel):
    """Whether the runtime serves this model. The only field an update may change.

    A rate is not here: an organization's rates live in
    ``organization_model_pricing`` and are written through
    ``/organizations/me/pricing``, so a price set on this surface and a price set
    on that one could not disagree.
    """

    enabled: bool


class OrgProviderKeyModelPublic(SQLModel):
    """One offered model, with the rate the caller's organization is charged for it.

    ``price_source`` says which rung of ``services.pricing_service`` answered:
    ``organization`` for a rate an admin set, ``defaults`` for the
    community-maintained rate this surface seeded or the genai-prices fallback,
    ``deployment`` for the deployment's own price list, and None when nothing
    prices the model yet. ``pricing_id`` names the organization's own row where
    there is one, so a client can edit that rate without re-deriving the key.
    """

    id: uuid.UUID
    org_provider_key_id: uuid.UUID
    model: str
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cache_read_price_per_million: float | None = None
    cache_write_price_per_million: float | None = None
    cache_write_1h_price_per_million: float | None = None
    price_source: PriceSource | None = None
    pricing_id: uuid.UUID | None = None
    enabled: bool
    created_at: datetime
    updated_at: datetime | None = None


class OrgProviderKeyModelsPublic(SQLModel):
    """One page of a key's offered models, and how many there are in total."""

    data: list[OrgProviderKeyModelPublic]
    count: int


class OrgProviderModelsRefreshPublic(SQLModel):
    """What a refresh did: what it newly offered, what it repriced, and the list's new size.

    Failure is a field rather than a status, for the reason
    ``OrgProviderAvailableModelsPublic`` gives: the list is still standing, and
    the panel renders the reason beside it.
    """

    # Required rather than defaulted, both of them, so the wire contract says
    # these lists are always present. Defaulted, OpenAPI marks them optional and
    # every client has to guard a field the server always sends.
    added: list[str]
    repriced: list[str]
    count: int
    error: str | None = None
    discovery_unsupported: bool = False


class OrgProviderAvailableModelsPublic(SQLModel):
    """What the provider says it serves on this key's stored credential.

    Failure is a field rather than a status: an unreachable upstream, or a
    provider with no model listing, is an answer about the provider rather than
    about this request, and the form still has to render (with a plain text box)
    when the list cannot be fetched.
    """

    provider: str
    models: list[str] = Field(default_factory=list)
    error: str | None = None
    discovery_unsupported: bool = False


class OrgProviderKeyModel(SQLModel, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    """One model an organization offers on one of its provider keys.

    Membership and a serving switch, nothing more. The rate lives in
    ``organization_model_pricing`` keyed ``provider:model``, which is the store
    ``services.pricing_service.find_model_pricing`` already reads first for a
    request whose organization is known, so billing reads exactly what this
    surface writes and no copy can drift.

    **No rows for a key means the key is unnarrowed**, exactly as
    ``WorkspaceProviderModelRestriction`` means it: a key that has never been
    refreshed keeps reaching every model of its provider. One or more rows
    narrows the organization to the enabled ones, which is how the serving
    switch reaches both the catalog
    (``services/tenancy/organization_model_access``) and dispatch
    (``cached_org_model_restriction``).
    """

    __tablename__ = "org_provider_key_models"
    __table_args__ = (
        UniqueConstraint("org_provider_key_id", "model", name="uq_org_provider_key_models_key_model"),
        # Same reasoning as `WorkspaceProviderKeyOverride`'s matching constraint.
        ForeignKeyConstraint(
            ["organization_id", "org_provider_key_id"],
            ["org_provider_keys.organization_id", "org_provider_keys.id"],
            ondelete="CASCADE",
        ),
    )

    # Denormalized from the key's own organization; see the composite FK above
    # for why it is stored rather than joined at read time.
    organization_id: uuid.UUID
    org_provider_key_id: uuid.UUID = Field(index=True)
    model: str = Field(max_length=255)
    # A model nothing prices is offered but not served, so a model the pricing
    # data has not caught up with cannot be billed at nothing.
    enabled: bool = Field(default=True, nullable=False, sa_column_kwargs={"server_default": true()})



__all__ = [
    "OrgProviderKey",
    "OrgProviderKeyCreateRequest",
    "OrgProviderAvailableModelsPublic",
    "OrgProviderKeyModel",
    "OrgProviderKeyModelCreateRequest",
    "OrgProviderKeyModelPublic",
    "OrgProviderKeyModelUpdateRequest",
    "OrgProviderKeyModelsPublic",
    "OrgProviderModelsRefreshPublic",
    "OrgProviderKeyPublic",
    "OrgProviderKeyUpdateRequest",
    "OrgProviderKeysPublic",
    "WorkspaceProviderKeyOverride",
    "WorkspaceProviderKeyOverridePublic",
    "WorkspaceProviderKeyOverrideRequest",
    "WorkspaceProviderKeyOverridesPublic",
    "WorkspaceProviderModelRestriction",
    "WorkspaceProviderModelRestrictionRequest",
    "WorkspaceProviderModelRestrictionsPublic",
]
