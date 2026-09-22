"""Request and response models of the providers domain.

The wire shapes for a deployment's organization-scoped provider credentials:
the keys themselves, a workspace's departure from its organization's default,
and the models a key offers with the rate each is charged at.

They are SQLModel rather than plain Pydantic because they are read straight off
the rows in ``models/provider_keys.py``, whose tables are SQLModel too.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlmodel import Field, SQLModel

from gateway.models.pricing import PriceSource
from gateway.models.provider_keys import OrgProviderKey
from gateway.models.secret_fields import redact_secret_like_values


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

    @classmethod
    def from_row(cls, key: OrgProviderKey, *, usable: bool) -> OrgProviderKeyPublic:
        """Read one row for the API. Never includes the key, only ``last4``.

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
        return cls(
            id=key.id,
            organization_id=key.organization_id,
            provider=key.provider,
            name=key.name,
            api_base=key.api_base,
            client_args=redact_secret_like_values(key.client_args),
            last4=key.last4,
            is_org_default=key.is_org_default,
            usable=usable,
            archived_at=key.archived_at,
            created_at=key.created_at,
            updated_at=key.updated_at,
        )


class OrgProviderKeysPublic(SQLModel):
    data: list[OrgProviderKeyPublic]
    count: int


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


class WorkspaceProviderModelRestrictionRequest(SQLModel):
    model: str = Field(max_length=255)


class WorkspaceProviderModelRestrictionsPublic(SQLModel):
    models: list[str]


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


__all__ = [
    "OrgProviderAvailableModelsPublic",
    "OrgProviderKeyCreateRequest",
    "OrgProviderKeyModelCreateRequest",
    "OrgProviderKeyModelPublic",
    "OrgProviderKeyModelUpdateRequest",
    "OrgProviderKeyModelsPublic",
    "OrgProviderKeyPublic",
    "OrgProviderKeyUpdateRequest",
    "OrgProviderKeysPublic",
    "OrgProviderModelsRefreshPublic",
    "WorkspaceProviderKeyOverridePublic",
    "WorkspaceProviderKeyOverrideRequest",
    "WorkspaceProviderKeyOverridesPublic",
    "WorkspaceProviderModelRestrictionRequest",
    "WorkspaceProviderModelRestrictionsPublic",
]
