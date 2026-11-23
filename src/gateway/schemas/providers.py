"""Request and response models of the providers domain.

The wire shapes for a deployment's organization-scoped provider credentials:
the keys themselves, a workspace's departure from its organization's default,
and the models a key offers with the rate each is charged at. Also the
endpoints a workspace or one of its users owns, and the hosted providers the
deployment serves on its own credentials.

They are SQLModel rather than plain Pydantic because they are read straight off
the rows in ``models/provider_keys.py``, whose tables are SQLModel too.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import AwareDatetime, ValidationInfo, field_validator, model_validator
from sqlmodel import Field, SQLModel

from gateway.core.link_urls import normalize_link_url
from gateway.core.settings.inference import country_codes, data_location, data_locations
from gateway.models.money import as_float
from gateway.models.pricing import HostedPriceSource, ModelPricing, PriceSource
from gateway.models.provider_keys import OrgProviderKey
from gateway.models.providers import (
    HostedModelDisabledReason,
    HostedProvider,
    HostedProviderModel,
    PlatformConnectState,
    ProviderEndpoint,
)
from gateway.models.secret_fields import redact_secret_like_values
from gateway.types.endpoint_auth import EndpointAuthMethod, EndpointAuthState, UnreadableEndpointAuthError
from gateway.types.model_type import ModelType, OfferedModelType
from gateway.types.price_unit import AcceptedPriceUnit, PriceUnit


class EndpointAuthRequest(SQLModel):
    """The credential an edge proxy in front of a key's base URL requires.

    ``method`` is a closed set of ways to produce request headers. TLS client
    authentication is not one of them: it is presented when the connection is
    made, so it would be a field of its own rather than a method here.
    """

    method: EndpointAuthMethod = Field(
        description="How the credential is produced. `static_header` sends one stored value as one header."
    )
    header_name: str = Field(
        max_length=64,
        description=(
            "The header the value is sent in. Letters, digits and hyphens only. A header a provider sends "
            "its own credential in, one reserved for the connection, `Host`, and Otari's own namespace are "
            "all refused."
        ),
    )
    secret: str | None = Field(
        default=None,
        description=(
            "The value to send. Write-only: responses report only its last four characters. Required when "
            "the credential is first set, and on any change of base URL."
        ),
    )
    expires_at: AwareDatetime = Field(
        description=(
            "When the organization expects to rotate this credential. Must carry a UTC offset and be in "
            "the future. It is a rotation date, not a switch: a request is never refused because it "
            "passed, since the proxy answers that itself on every request."
        )
    )


class EndpointAuthPublic(SQLModel):
    """What an endpoint credential reports. Never carries the value, only its last four characters."""

    method: EndpointAuthMethod = Field(description="How the credential is produced.")
    header_name: str
    last4: str = Field(description="The last four characters of the stored value.")
    expires_at: datetime
    state: EndpointAuthState = Field(
        description=(
            "How the credential stands against its own expiry. Reported so every reader draws the line in "
            "the same place. No value of it refuses a request: the proxy answers that on every request."
        )
    )

    @classmethod
    def from_row(cls, key: OrgProviderKey, *, now: datetime) -> EndpointAuthPublic | None:
        """Read one row's endpoint credential for the API, or None where it carries none.

        A credential this build cannot read is reported as none, because a list
        of keys must render whatever a row holds.
        """
        try:
            auth = key.endpoint_auth
        except UnreadableEndpointAuthError:
            return None
        if auth is None:
            return None
        return cls(
            method=auth.method,
            header_name=auth.header_name,
            last4=auth.last4,
            expires_at=auth.expires_at,
            state=auth.state(now=now),
        )


class OrgProviderKeyCreateRequest(SQLModel):
    """What a caller sends to create a key.

    The plaintext key is never stored as sent: the service encrypts it
    (`services/secret_box.py`) and keeps only the ciphertext and ``last4``,
    the same convention `providers.ProviderCredential` already uses.
    """

    provider: str = Field(max_length=255)
    name: str = Field(max_length=255)
    slug: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "The name requests reach a key with a custom base URL by, as in `<slug>:<model>`. Only such a key "
            "has one, and once set it never changes. Left out, it is made from the key's name."
        ),
    )
    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None, max_length=1024)
    client_args: dict[str, Any] | None = None
    zero_data_retention: bool = Field(
        default=False,
        description=(
            "The organization's declaration that the provider account behind this key keeps no prompt or "
            "completion data. The catalog shows it and filters on it; nothing is enforced on it."
        ),
    )
    endpoint_auth: EndpointAuthRequest | None = Field(
        default=None,
        description="The credential an edge proxy in front of `api_base` requires. Only valid with an `api_base`.",
    )


class OrgProviderKeyUpdateRequest(SQLModel):
    """A partial update. Every field is optional; only what is set is applied."""

    name: str | None = Field(default=None, max_length=255)
    api_key: str | None = None
    api_base: str | None = Field(default=None, max_length=1024)
    client_args: dict[str, Any] | None = None
    zero_data_retention: bool | None = None
    endpoint_auth: EndpointAuthRequest | None = Field(
        default=None,
        description=(
            "The credential an edge proxy in front of `api_base` requires. Send it to set or replace one; "
            "send an explicit null to remove one. Leave it out to keep what is stored."
        ),
    )


class OrgProviderKeyPublic(SQLModel):
    """The API-facing shape. Never carries the key, only whether one is set."""

    id: uuid.UUID
    organization_id: uuid.UUID
    provider: str
    name: str
    slug: str | None = Field(
        default=None,
        description=(
            "What requests name this key by, as in `<slug>:<model>`, for a key with a custom base URL. "
            "Absent for a key reached as `<provider>:<model>`."
        ),
    )
    api_base: str | None = None
    client_args: dict[str, Any] | None = None
    last4: str | None = None
    endpoint_auth: EndpointAuthPublic | None = None
    is_org_default: bool
    zero_data_retention: bool = False
    usable: bool = Field(
        description=(
            "False when the stored credential cannot be decrypted on this deployment, so the key "
            "supplies nothing at dispatch and the catalog withholds its provider. A row this deployment "
            "cannot read is still listed, because deleting or replacing it is what fixes it."
        )
    )
    offered_count: int = Field(
        description=(
            "How many models the organization offers on this key. Zero means the key has never been "
            "refreshed, so it is unnarrowed and reaches whatever its provider serves."
        )
    )
    serving_count: int = Field(
        description=(
            "How many of the offered models have their serving switch on. A key offering models and "
            "serving none refuses every request through it until one is switched on."
        )
    )
    archived_at: datetime | None = None
    created_at: datetime
    updated_at: datetime | None = None

    @classmethod
    def from_row(
        cls,
        key: OrgProviderKey,
        *,
        client_args: dict[str, Any] | None,
        usable: bool,
        now: datetime,
        offered_count: int = 0,
        serving_count: int = 0,
    ) -> OrgProviderKeyPublic:
        """Read one row for the API. Never includes the key, only ``last4``.

        ``now`` is passed rather than read so a caller holds the clock, as the
        endpoint credential's own state already requires.

        ``client_args`` and ``usable`` are both passed in rather than derived
        here: answering either means decrypting, which belongs to the service
        layer that owns the secret box. Both are required rather than defaulted
        so a caller cannot report a key as working, or as carrying no settings,
        without having asked. The two counts come from the offered-models table,
        and default to a key that offers nothing, which is the one a row has
        before its first refresh.

        ``client_args`` is arbitrary JSON an admin can set (Bedrock's
        ``region_name``, other client kwargs), and a credential-shaped entry in
        it is never echoed back: ``redact_secret_like_values`` masks it the same
        way ``encrypted_api_key`` itself stays off the wire, with only ``last4``
        coming back.
        """
        return cls(
            id=key.id,
            organization_id=key.organization_id,
            provider=key.provider,
            name=key.name,
            slug=key.slug,
            api_base=key.api_base,
            client_args=redact_secret_like_values(client_args),
            endpoint_auth=EndpointAuthPublic.from_row(key, now=now),
            last4=key.last4,
            is_org_default=key.is_org_default,
            zero_data_retention=key.zero_data_retention,
            usable=usable,
            offered_count=offered_count,
            serving_count=serving_count,
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
    # The organization's declaration on the key, so a workspace that requires
    # zero data retention can be shown which of its keys that leaves.
    zero_data_retention: bool = False


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
    community-maintained rate this surface seeded or the models.dev fallback,
    ``deployment`` for the deployment's own price list, and None when nothing
    prices the model yet. ``pricing_id`` names the organization's own row where
    there is one, so a client can edit that rate without re-deriving the key.
    ``unit`` says what the rates are per, as on a pricing row: token, request
    or image, so a per-request rate is not read as a per-token one. ``type``
    is the kind of model, from models.dev, and None when it does not know.
    """

    id: uuid.UUID
    org_provider_key_id: uuid.UUID
    model: str
    type: ModelType | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cache_read_price_per_million: float | None = None
    cache_write_price_per_million: float | None = None
    price_source: PriceSource | None = None
    pricing_id: uuid.UUID | None = None
    unit: PriceUnit = "token"
    enabled: bool
    created_at: datetime
    updated_at: datetime | None = None


class OrgProviderKeyModelsPublic(SQLModel):
    """One page of a key's offered models, and how many there are in total.

    ``count`` is the total of the kind asked for, and ``type_counts`` how many
    of each kind the key offers, ``unknown`` for a model of no known kind.
    """

    data: list[OrgProviderKeyModelPublic]
    count: int
    type_counts: dict[OfferedModelType, int] = Field(default_factory=dict)


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


_ENDPOINT_NAME_DESC = (
    "What callers put before the colon to reach this endpoint, as '<name>:<model>'. Letters, digits, '.', '_' "
    "and '-', starting with a letter or digit. It may not be a provider's name or a configured instance's."
)
_ENDPOINT_PROVIDER_DESC = (
    "The implementation that speaks to the endpoint: 'openai' for an OpenAI-compatible server, 'anthropic' for an "
    "Anthropic-compatible one. Whichever it is, callers may use any of the chat, responses and messages routes."
)
_ENDPOINT_API_BASE_DESC = (
    "Base URL of the endpoint. Refused when it resolves to a private, loopback, link-local or reserved address."
)
_ENDPOINT_DEFAULT_PARAMS_DESC = (
    "Fields added to every request body sent to this endpoint, beneath the caller's own: a field the caller sets "
    "wins. For fields the gateway does not model, such as vLLM's 'chat_template_kwargs'. Credential and "
    "transport fields are refused."
)


class ProviderEndpointCreateRequest(SQLModel):
    """What a caller sends to create an owned endpoint. The key is stored encrypted."""

    name: str = Field(max_length=64, description=_ENDPOINT_NAME_DESC)
    provider: str = Field(max_length=64, description=_ENDPOINT_PROVIDER_DESC)
    api_base: str = Field(max_length=1024, description=_ENDPOINT_API_BASE_DESC)
    api_key: str | None = Field(default=None, description="Sent to the endpoint. Never returned, only its last four.")
    default_params: dict[str, Any] | None = Field(default=None, description=_ENDPOINT_DEFAULT_PARAMS_DESC)
    workspace_id: uuid.UUID | None = Field(
        default=None, description="Workspace that owns the endpoint. Omit for the deployment's default workspace."
    )
    user_id: str | None = Field(
        default=None,
        description=(
            "User that owns the endpoint. Omit for one every caller in the workspace reaches. A user's endpoint "
            "shadows a workspace-wide one of the same name for that user alone."
        ),
    )


class ProviderEndpointUpdateRequest(SQLModel):
    """A partial update. Only what is set is applied; an explicit null ``api_key`` clears it."""

    name: str | None = Field(default=None, max_length=64, description=_ENDPOINT_NAME_DESC)
    provider: str | None = Field(default=None, max_length=64, description=_ENDPOINT_PROVIDER_DESC)
    api_base: str | None = Field(default=None, max_length=1024, description=_ENDPOINT_API_BASE_DESC)
    api_key: str | None = None
    default_params: dict[str, Any] | None = Field(default=None, description=_ENDPOINT_DEFAULT_PARAMS_DESC)


class ProviderEndpointPublic(SQLModel):
    """The API-facing shape. Never carries the key, only whether one is set."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: str | None = None
    name: str
    provider: str
    api_base: str
    last4: str | None = None
    default_params: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: ProviderEndpoint, *, default_params: dict[str, Any] | None) -> ProviderEndpointPublic:
        """Read one row for the API, masking credential-shaped ``default_params`` entries.

        ``default_params`` is passed in rather than read off the row: it is held
        encrypted, and decrypting belongs to the service layer that owns the
        secret box.
        """
        return cls(
            id=row.id,
            workspace_id=row.workspace_id,
            user_id=row.user_id,
            name=row.name,
            provider=row.provider,
            api_base=row.api_base,
            last4=row.last4,
            default_params=redact_secret_like_values(default_params),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class ProviderEndpointsPublic(SQLModel):
    data: list[ProviderEndpointPublic]
    count: int


class HostedProviderCreateRequest(SQLModel):
    """Configure a provider this deployment will serve hosted inference on.

    ``provider`` is the any-llm implementation. ``name`` is what requests
    address it by (``name:model``), defaulting to ``provider``; a name of its
    own lets two endpoints of one implementation coexist. ``api_key`` may be
    left out for a backend that needs none, or for an endpoint of its own
    (``api_base``), which is called with a placeholder key.

    ``client_args`` is whatever the provider's SDK client needs beyond the key
    (Bedrock's region and IAM pair), the same shape an organization provider
    key takes. Values under credential-shaped names come back masked.
    ``restricted_to_organization_id`` keeps the provider to one organization;
    every other one is served as if it were not configured.
    ``serves_unlisted_models`` serves a model the roster does not list, as a
    ``config.yml`` instance does, and switches on every model it offers
    whether or not something prices it. ``supported_countries`` keeps the
    provider to requests from those ISO country codes, checked as well as
    ``provider_supported_countries``; null serves every country.
    ``data_location`` declares where the upstream keeps request data: an ISO
    3166-1 alpha-2 country code, ``EU`` or ``GLOBAL``; null leaves it undeclared.
    A workspace that lists the locations it accepts is served only from one of them.
    ``zero_data_retention`` declares that the upstream keeps no prompt or
    completion data; the catalog shows it and nothing enforces it.
    ``terms_url`` and ``privacy_url`` link the upstream's own terms of service
    and privacy policy from the public Providers pages: absolute http(s)
    addresses with no username or password.
    """

    model_config = {"json_schema_extra": {"example": {"provider": "openai", "api_key": "<your OpenAI API key>"}}}

    provider: str = Field(min_length=1, max_length=64)
    name: str | None = Field(default=None, min_length=1, max_length=64)
    api_key: str | None = Field(default=None, min_length=1, max_length=1024)
    api_base: str | None = Field(default=None, max_length=1024)
    client_args: dict[str, Any] | None = None
    enabled: bool = True
    restricted_to_organization_id: uuid.UUID | None = None
    serves_unlisted_models: bool = False
    supported_countries: list[str] | None = Field(default=None, max_length=300)
    data_location: str | None = Field(default=None, max_length=16)
    zero_data_retention: bool = False
    terms_url: str | None = Field(default=None, max_length=1024)
    privacy_url: str | None = Field(default=None, max_length=1024)

    @field_validator("supported_countries")
    @classmethod
    def _country_codes(cls, value: list[str] | None) -> list[str] | None:
        return country_codes(value) if value is not None else None

    @field_validator("data_location")
    @classmethod
    def _data_location(cls, value: str | None) -> str | None:
        return data_location(value) if value is not None else None

    @field_validator("terms_url", "privacy_url")
    @classmethod
    def _link_url(cls, value: str | None, info: ValidationInfo) -> str | None:
        return normalize_link_url(value, info.field_name or "link")


class HostedProviderUpdateRequest(SQLModel):
    """Rotate the key, repoint the base, or turn a provider off.

    ``provider`` and ``name`` are immutable: changing either is a delete plus a
    create, because the key belongs to the provider it was issued by and
    requests address the name. Omitting ``api_key`` leaves the stored
    credential untouched, so a toggle never re-sends a secret the caller does
    not have; an explicit null removes it, where the provider can be served
    without one. ``client_args`` omitted is left alone; an explicit
    null clears it, and an entry echoed back as the mask keeps the stored value.
    ``restricted_to_organization_id`` reads the same way: omitted keeps the
    restriction, null lifts it, and so does ``supported_countries``, where an
    empty list closes the provider to every country, and ``data_location``. ``terms_url`` and
    ``privacy_url`` omitted are left alone; null or an empty string removes the link.
    """

    # A toggle alone, so a reader who edits the example never re-sends a key.
    model_config = {"json_schema_extra": {"example": {"enabled": False}}}

    api_key: str | None = Field(default=None, min_length=1, max_length=1024)
    api_base: str | None = Field(default=None, max_length=1024)
    client_args: dict[str, Any] | None = None
    enabled: bool | None = None
    restricted_to_organization_id: uuid.UUID | None = None
    serves_unlisted_models: bool | None = None
    supported_countries: list[str] | None = Field(default=None, max_length=300)
    data_location: str | None = Field(default=None, max_length=16)
    zero_data_retention: bool | None = None
    terms_url: str | None = Field(default=None, max_length=1024)
    privacy_url: str | None = Field(default=None, max_length=1024)

    @field_validator("supported_countries")
    @classmethod
    def _country_codes(cls, value: list[str] | None) -> list[str] | None:
        return country_codes(value) if value is not None else None

    @field_validator("data_location")
    @classmethod
    def _data_location(cls, value: str | None) -> str | None:
        return data_location(value) if value is not None else None

    @field_validator("terms_url", "privacy_url")
    @classmethod
    def _link_url(cls, value: str | None, info: ValidationInfo) -> str | None:
        return normalize_link_url(value, info.field_name or "link")


class HostedProviderModelSummary(SQLModel):
    """How many models a hosted provider offers, how many it serves, and what holds the rest back.

    ``serving`` counts a model switched on whose vendor is switched on too.
    ``unpriced`` counts a model held off for want of a rate, and
    ``vendors_off`` the vendors the provider does not serve.
    """

    total: int
    serving: int
    unpriced: int
    vendors_off: int


class HostedProviderRegionCreateRequest(SQLModel):
    """Add a region a hosted provider serves from.

    Only what differs from the provider is given; every field left out is the
    provider's. ``slug`` pins the region in a selector
    (``<provider>@<slug>:<model>``) and names it in every route. ``api_key``
    left out uses the provider's key, which a region on another origin than
    the provider's endpoint may not: it needs a key of its own there.
    ``client_args`` are merged over the provider's on the provider's origin, so
    a Bedrock region gives its ``region_name`` alone; a region on another
    origin takes its own alone. An ``api_base`` that carries a key must be https.
    ``data_location`` declares where the region keeps request data. Left out, it
    copies the provider's declaration as it stands; null declares nothing. It is
    the region's own from then on: a later change to the provider's does not
    move it.

    A request is served by the first enabled region whose countries admit it
    and whose data location the workspace accepts; a pinned region is held to
    the same filters. A provider with no regions yet serves from its own connection. Adding its
    first region keeps that connection as a region of its own, ahead of the new
    one, under ``primary_slug`` and ``primary_label``: by default the
    provider's data location, or ``primary``.
    """

    model_config = {
        "json_schema_extra": {
            "example": {
                "slug": "sg",
                "label": "Southeast Asia",
                "api_base": "https://tokenhub-intl.example.com/v1",
                "api_key": "<the region's key>",
                "data_location": "SG",
            }
        }
    }

    slug: str = Field(min_length=1, max_length=32)
    label: str = Field(min_length=1, max_length=64)
    api_key: str | None = Field(default=None, min_length=1, max_length=1024)
    api_base: str | None = Field(default=None, max_length=1024)
    client_args: dict[str, Any] | None = None
    data_location: str | None = Field(default=None, max_length=16)
    supported_countries: list[str] | None = Field(default=None, max_length=300)
    enabled: bool = True
    primary_slug: str | None = Field(default=None, min_length=1, max_length=32)
    primary_label: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("supported_countries")
    @classmethod
    def _country_codes(cls, value: list[str] | None) -> list[str] | None:
        return country_codes(value) if value is not None else None

    @field_validator("data_location")
    @classmethod
    def _data_location(cls, value: str | None) -> str | None:
        return data_location(value) if value is not None else None


class HostedProviderRegionUpdateRequest(SQLModel):
    """Change one region. A field left out is left alone.

    Null puts a field the region inherits back on the provider's value:
    ``api_key`` null returns the region to the provider's key, an empty
    ``api_base`` or null to the provider's endpoint, and so on for
    ``client_args`` and ``supported_countries``. ``data_location`` is the
    region's own, so null declares nothing. Null on ``label``, ``enabled`` or
    ``position`` changes nothing. ``client_args`` echoed
    back with masked entries keeps the stored values. ``position`` moves the
    region to that place in the order regions are tried, counted from 0.
    """

    model_config = {"json_schema_extra": {"example": {"enabled": False}}}

    label: str | None = Field(default=None, min_length=1, max_length=64)
    api_key: str | None = Field(default=None, min_length=1, max_length=1024)
    api_base: str | None = Field(default=None, max_length=1024)
    client_args: dict[str, Any] | None = None
    data_location: str | None = Field(default=None, max_length=16)
    supported_countries: list[str] | None = Field(default=None, max_length=300)
    enabled: bool | None = None
    position: int | None = Field(default=None, ge=0, le=1000)

    @field_validator("supported_countries")
    @classmethod
    def _country_codes(cls, value: list[str] | None) -> list[str] | None:
        return country_codes(value) if value is not None else None

    @field_validator("data_location")
    @classmethod
    def _data_location(cls, value: str | None) -> str | None:
        return data_location(value) if value is not None else None


class HostedProviderRegionPublic(SQLModel):
    """One region of a hosted provider, as stored: null wherever it uses the provider's value.

    ``data_location`` is the exception: it is the region's own, and null there
    means nothing is declared.

    ``selector`` is what pins it in a request (``tokenhub@sg``). ``has_api_key``
    is true for a region with a key of its own, known by its last four
    characters; ``client_args`` holds only the region's own extras, masked as
    the provider's are. ``decryptable`` is false for a region whose own key or
    extras no configured ``OTARI_SECRET_KEY`` reads, which serves nothing until
    re-entered.
    """

    slug: str
    selector: str
    label: str
    position: int
    enabled: bool
    data_location: str | None = None
    has_api_key: bool = False
    api_key_last4: str | None = None
    api_base: str | None = None
    client_args: dict[str, Any] | None = None
    supported_countries: list[str] | None = None
    decryptable: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class HostedProviderPublic(SQLModel):
    """One hosted provider, known by its name, its implementation and the tail of its key.

    No shape in this section carries key material: a stored credential is
    known by its last four characters and nothing else, ``client_args`` goes
    out with credential-shaped values masked, and ``api_base`` with the
    password in its userinfo and its query values masked.

    ``name`` is what requests address and every route takes; ``provider`` is
    the any-llm implementation. ``source`` says where the provider is defined.
    A ``config`` provider comes from the ``hosted_providers:`` block, has no
    row and so no ``id`` or timestamps, and is changed by editing the
    configuration rather than through this API; its models are administered as
    a stored provider's are. ``shadows_config`` marks a stored provider that
    wins over a configuration entry of the same name, which serves again once
    the row is deleted. ``has_api_key`` is false for one stored without a key,
    and ``decryptable`` false for one whose key or extras no configured
    ``OTARI_SECRET_KEY`` reads, which serves nothing until re-entered.
    ``serves_unlisted_models`` is true for one that serves a model its roster
    does not list. ``supported_countries`` is null for one that serves every
    country. ``zero_data_retention`` is the operator's declaration that the
    upstream keeps no prompt or completion data. ``terms_url`` and
    ``privacy_url`` are the upstream's policies, null where none is linked.
    ``data_location`` is where the upstream keeps request data, null where
    nobody declared it. ``regions`` lists the regions it serves from, in the
    order they are tried, and is empty for a provider that serves from its own
    connection alone. ``models`` counts its offered models, on the list only.
    """

    id: uuid.UUID | None = None
    name: str
    provider: str
    source: Literal["database", "config"] = "database"
    has_api_key: bool = True
    decryptable: bool = True
    api_key_last4: str | None = None
    api_base: str | None = None
    client_args: dict[str, Any] | None = None
    enabled: bool
    restricted_to_organization_id: uuid.UUID | None = None
    serves_unlisted_models: bool = False
    supported_countries: list[str] | None = None
    data_location: str | None = None
    zero_data_retention: bool = False
    terms_url: str | None = None
    privacy_url: str | None = None
    shadows_config: bool = False
    regions: list[HostedProviderRegionPublic] = Field(default_factory=list)
    models: HostedProviderModelSummary | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(
        cls,
        row: HostedProvider,
        *,
        client_args: dict[str, Any] | None,
        api_base: str | None,
        decryptable: bool,
        shadows_config: bool = False,
    ) -> HostedProviderPublic:
        """Read one row for the API: no key, and the extras with credential-shaped values masked.

        ``client_args`` is passed in decrypted, because the row holds them
        encrypted and the service is what reads them back, and so is whether
        they could be. ``api_base`` is passed in already masked, because the URL
        masking lives in the service layer this one may not import.
        """
        return cls(
            id=row.id,
            name=row.name,
            provider=row.provider,
            source="database",
            has_api_key=row.encrypted_api_key is not None,
            decryptable=decryptable,
            api_key_last4=row.api_key_last4,
            api_base=api_base,
            client_args=redact_secret_like_values(client_args),
            enabled=row.enabled,
            restricted_to_organization_id=row.restricted_to_organization_id,
            serves_unlisted_models=row.serves_unlisted_models,
            supported_countries=row.supported_countries,
            data_location=row.data_location,
            zero_data_retention=row.zero_data_retention,
            terms_url=row.terms_url,
            privacy_url=row.privacy_url,
            shadows_config=shadows_config,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        provider: str,
        api_key_last4: str | None,
        has_api_key: bool,
        api_base: str | None,
        client_args: dict[str, Any] | None,
        enabled: bool,
        restricted_to_organization_id: uuid.UUID | None,
        supported_countries: list[str] | None = None,
        data_location: str | None = None,
        zero_data_retention: bool = False,
        terms_url: str | None = None,
        privacy_url: str | None = None,
    ) -> HostedProviderPublic:
        """Read one configuration entry for the API, masked as a row is.

        ``api_base`` is passed in already masked, as for a row.
        """
        return cls(
            name=name,
            provider=provider,
            source="config",
            has_api_key=has_api_key,
            api_key_last4=api_key_last4,
            api_base=api_base,
            client_args=redact_secret_like_values(client_args),
            enabled=enabled,
            restricted_to_organization_id=restricted_to_organization_id,
            supported_countries=supported_countries,
            data_location=data_location,
            zero_data_retention=zero_data_retention,
            terms_url=terms_url,
            privacy_url=privacy_url,
        )


class HostedProvidersPublic(SQLModel):
    """A page of hosted providers, and how many there are in total."""

    data: list[HostedProviderPublic]
    count: int


class HostedProviderTestRequest(SQLModel):
    """Credentials to try against a provider without storing them.

    The fields the create body takes. ``name`` naming an existing stored
    provider lets a field left out fall back to what it stores, so an edit form
    can test without re-entering the key; pointing the stored key at another
    origin still needs the key typed again, as an update does.
    """

    model_config = {"json_schema_extra": {"example": {"provider": "openai", "api_key": "<your OpenAI API key>"}}}

    provider: str = Field(min_length=1, max_length=64)
    name: str | None = Field(default=None, min_length=1, max_length=64)
    api_key: str | None = Field(default=None, min_length=1, max_length=1024)
    api_base: str | None = Field(default=None, max_length=1024)
    client_args: dict[str, Any] | None = None


class HostedProviderTestPublic(SQLModel):
    """What a connection test found: whether the provider answered, and how many models it lists.

    ``discovery_unsupported`` is set for a provider with no model listing,
    which a test cannot reach that way although it may still serve.
    """

    ok: bool
    model_count: int = 0
    error: str | None = None
    discovery_unsupported: bool = False


class HostedProvidersReencryptPublic(SQLModel):
    """What re-encrypting the stored hosted provider credentials did.

    ``unreadable`` counts providers no configured key reads, left as they were
    to be re-entered; ``skipped`` those edited while the rotation ran, already
    written under the primary key.
    """

    reencrypted: int
    unreadable: int
    skipped: int


DeploymentProviderMigrationOutcome = Literal["migrate", "migrated", "conflict", "unreadable", "invalid", "failed"]


class DeploymentProviderMigrationRowPublic(SQLModel):
    """What moving one deployment provider onto a hosted provider does, or did.

    ``instance`` is the deployment provider's name, which the hosted provider
    keeps, so ``instance:model`` selectors keep working. ``outcome`` is
    ``migrate`` in a preview and ``migrated`` once applied; ``conflict``,
    ``unreadable``, ``invalid`` and ``failed`` leave the row where it is, with
    ``detail`` saying why. ``source_removed`` says whether the deployment
    provider was removed: it is kept while the hosted provider offers no model,
    and when it changed while the move ran. ``notes`` are what an operator
    should check afterwards.
    """

    instance: str
    provider: str | None = None
    api_base: str | None = None
    outcome: DeploymentProviderMigrationOutcome
    detail: str | None = None
    models: int | None = None
    source_removed: bool = False
    notes: list[str] = Field(default_factory=list)
    updated_at: datetime | None = Field(
        default=None, description="The deployment provider's version, which an apply names to move this row as shown."
    )


class DeploymentProviderMigrationReviewedRow(SQLModel):
    """One row of the preview the caller reviewed: its instance and the version it was shown at."""

    instance: str
    updated_at: datetime | None = None


class DeploymentProviderMigrationRequest(SQLModel):
    """Which deployment providers to move: the ones the caller reviewed, as reviewed.

    ``rows`` names the rows of the preview the caller saw, each with the
    ``updated_at`` it was shown at. A row the preview did not show, or one
    changed since, is left where it is, so nothing moves that nobody reviewed.
    Omitted, every row that can move does.
    """

    model_config = {
        "json_schema_extra": {"example": {"rows": [{"instance": "openai-eu", "updated_at": "2026-10-19T09:00:00Z"}]}}
    }

    rows: list[DeploymentProviderMigrationReviewedRow] | None = None


class DeploymentProviderMigrationPublic(SQLModel):
    """The deployment providers the retired dashboard page stored, and how each moves onto a hosted provider."""

    rows: list[DeploymentProviderMigrationRowPublic]


def _rates_come_as_a_pair(request: HostedModelCreateRequest | HostedModelUpdateRequest) -> None:
    """Refuse half a price.

    ``model_pricing`` stores input and output in one row, so a single rate has
    no shape to land in. The cache rates and the unit are that row's optional
    half: each may come alone, but only beside the pair that gives the row its
    base rates.
    """
    if (request.input_price_per_million is None) != (request.output_price_per_million is None):
        msg = "input_price_per_million and output_price_per_million must be set together"
        raise ValueError(msg)
    rate_fields = {
        "cache_read_price_per_million",
        "cache_write_price_per_million",
        "unit",
    }
    if request.input_price_per_million is None and rate_fields & request.model_fields_set:
        msg = "cache rates and unit need input_price_per_million and output_price_per_million beside them"
        raise ValueError(msg)


class HostedModelCreateRequest(SQLModel):
    """Offer a model on a hosted provider, optionally at a custom price.

    Omitting the rates means the deployment serves the model at whatever the
    price ladder already answers: a stored deployment price, else the community
    default, which the offer stores as the deployment's own rate. ``unit`` is
    what the rates are per, as on ``POST /pricing``: a request or image rate is
    still stored per million.
    """

    model_config = {
        "json_schema_extra": {
            "example": {"model": "gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10}
        }
    }

    model: str = Field(min_length=1, max_length=255)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    cache_read_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_price_per_million: float | None = Field(default=None, ge=0)
    unit: AcceptedPriceUnit | None = None

    @model_validator(mode="after")
    def _prices_come_as_a_pair(self) -> Self:
        _rates_come_as_a_pair(self)
        return self


class HostedModelUpdateRequest(SQLModel):
    """Reprice an offered model, toggle whether it is served, or both.

    A price is the input/output pair or nothing: sending the pair writes a new
    pricing version, and the cache rates then follow ``POST /pricing``'s own
    semantics (a field the caller omits inherits the most recent stored value,
    an explicit null clears it). An omitted ``unit`` carries the stored one
    forward. ``enabled`` travels alone freely, so a toggle never re-sends rates
    it did not change.
    """

    # A toggle alone: an example carrying zero rates would, edited for the
    # switch, write a zero price as a new version.
    model_config = {"json_schema_extra": {"example": {"enabled": False}}}

    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    cache_read_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_price_per_million: float | None = Field(default=None, ge=0)
    unit: AcceptedPriceUnit | None = None
    enabled: bool | None = None

    @model_validator(mode="after")
    def _a_price_is_a_pair(self) -> Self:
        _rates_come_as_a_pair(self)
        return self


class HostedModelRates(SQLModel):
    """The rates a hosted model is currently served at, and which rung of the deployment's ladder said so."""

    source: HostedPriceSource
    input_price_per_million: float
    output_price_per_million: float
    cache_read_price_per_million: float | None = None
    cache_write_price_per_million: float | None = None
    unit: str = "token"

    @classmethod
    def from_version(cls, version: ModelPricing, source: HostedPriceSource) -> HostedModelRates:
        """Read one pricing version, stored or transient, into the display shape."""
        return cls(
            source=source,
            input_price_per_million=float(version.input_price_per_million),
            output_price_per_million=float(version.output_price_per_million),
            cache_read_price_per_million=as_float(version.cache_read_price_per_million),
            cache_write_price_per_million=as_float(version.cache_write_price_per_million),
            unit=version.unit or "token",
        )


class HostedModelPublic(SQLModel):
    """One offered model, with the price the deployment currently serves it at.

    ``price_source`` says which rung of the deployment's ladder answered:
    ``deployment`` for a rate an operator set, ``provider`` for the rate the
    provider's own listing named, ``defaults`` for the community-maintained
    rate, whether this surface stored it or the fallback supplies it, and None
    when nothing prices the model yet. ``unit`` says what
    the rates are per, so a per-request rate is not read as a per-token one.
    ``enabled`` is the serving switch; a model offered without a discoverable
    price starts off. ``disabled_reason`` says why a model is off: ``unpriced``
    while it waits for a rate, ``operator`` when an operator switched it off,
    ``unavailable`` when its provider refused to serve it. ``type`` is the
    kind of model, from models.dev, and None when it does not know.
    ``vendor`` is who makes the model, ``Other`` when nothing names one.
    ``default_rates`` is what the default rung (models.dev, or the provider's
    own listing) would price a model an operator priced, where it prices it
    and the two differ; None otherwise. ``region_rates`` holds, by region slug,
    the rates a region of the provider sets of its own (stored under
    ``<provider>@<slug>:<model>`` through the pricing endpoint); a region
    absent from it settles at the provider's rate.
    """

    id: uuid.UUID
    provider: str
    model: str
    type: ModelType | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cache_read_price_per_million: float | None = None
    cache_write_price_per_million: float | None = None
    unit: str | None = None
    price_source: HostedPriceSource | None = None
    default_rates: HostedModelRates | None = None
    region_rates: dict[str, HostedModelRates] = Field(default_factory=dict)
    vendor: str | None = None
    enabled: bool
    disabled_reason: HostedModelDisabledReason | None = None
    created_at: datetime
    updated_at: datetime | None = None

    @classmethod
    def from_row(
        cls,
        row: HostedProviderModel,
        *,
        rates: HostedModelRates | None,
        model_type: ModelType | None = None,
        vendor: str | None = None,
        default_rates: HostedModelRates | None = None,
        region_rates: dict[str, HostedModelRates] | None = None,
    ) -> HostedModelPublic:
        """Read one offered row for the API, with whatever currently prices it."""
        return cls(
            id=row.id,
            provider=row.provider,
            model=row.model,
            type=model_type,
            vendor=vendor,
            default_rates=default_rates,
            region_rates=region_rates or {},
            input_price_per_million=rates.input_price_per_million if rates else None,
            output_price_per_million=rates.output_price_per_million if rates else None,
            cache_read_price_per_million=rates.cache_read_price_per_million if rates else None,
            cache_write_price_per_million=rates.cache_write_price_per_million if rates else None,
            unit=rates.unit if rates else None,
            price_source=rates.source if rates else None,
            enabled=row.enabled,
            disabled_reason=row.disabled_reason,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class HostedVendorPublic(SQLModel):
    """One vendor whose models a hosted provider offers, and what it serves of them.

    ``enabled`` is the vendor's switch: off withholds every model of the
    vendor, including ones offered later, and keeps each model's own switch.
    The counts cover the whole provider, whatever the list was narrowed to:
    ``serving`` the models served, ``unpriced`` those held off for want of a
    rate, ``unavailable`` those the provider refused to serve, and
    ``custom_differs`` those an operator priced where the default rate now
    says something else.
    """

    vendor: str
    enabled: bool
    total: int
    serving: int
    unpriced: int
    unavailable: int
    custom_differs: int


class HostedModelsPublic(SQLModel):
    """A page of one hosted provider's offered models, and how many there are in total.

    ``count`` is the total of the kind asked for, and ``type_counts`` how many
    of each kind the provider offers, ``unknown`` for a model of no known kind.
    """

    data: list[HostedModelPublic]
    count: int
    type_counts: dict[OfferedModelType, int] = Field(default_factory=dict)
    vendors: list[HostedVendorPublic] = Field(default_factory=list)


class HostedVendorUpdateRequest(SQLModel):
    """Switch a vendor on or off on one hosted provider."""

    model_config = {"json_schema_extra": {"example": {"vendor": "Anthropic", "enabled": False}}}

    vendor: str = Field(min_length=1, max_length=128)
    enabled: bool


class HostedVendorModelsSwitchRequest(SQLModel):
    """Set the switch of every model of one vendor at once.

    ``enabled`` true serves each model that has a rate, and leaves one with
    none off. ``only_unpriced`` with ``enabled`` false withholds only the
    models served without a rate.
    """

    model_config = {"json_schema_extra": {"example": {"vendor": "Anthropic", "enabled": False}}}

    vendor: str = Field(min_length=1, max_length=128)
    enabled: bool
    only_unpriced: bool = False


class HostedVendorModelsSwitchPublic(SQLModel):
    """What a vendor-wide switch did: how many models it changed, and how many it left off.

    ``skipped_unpriced`` counts those left off for want of a rate, ``skipped_unavailable``
    those the provider refused to serve, which an operator switches on one at a time.
    """

    changed: int
    skipped_unpriced: int
    skipped_unavailable: int = 0


class HostedPriceResetRequest(SQLModel):
    """Put models an operator priced back on the default rate.

    ``models`` names them; null means every model, narrowed to ``vendor``
    when one is given. ``dry_run`` answers what a reset would change and
    writes nothing.
    """

    model_config = {"json_schema_extra": {"example": {"vendor": "Anthropic", "dry_run": True}}}

    models: list[str] | None = Field(default=None, max_length=1000)
    vendor: str | None = Field(default=None, max_length=128)
    dry_run: bool = False


class HostedPriceResetChangePublic(SQLModel):
    """One model a reset covers: the rate an operator set, and the default rate it moves to."""

    model: str
    vendor: str
    current: HostedModelRates
    default: HostedModelRates
    matches: bool


class HostedPriceResetPublic(SQLModel):
    """What a price reset changed, or would change on a dry run.

    ``changes`` lists every operator-priced model the default rung prices,
    ``matches`` marking one whose price is already the default; a price the
    configuration stored is never listed. ``skipped_no_default`` names the
    operator-priced models the default rung does not price, which keep their
    price. ``applied`` is true only when a reset wrote something.
    """

    changes: list[HostedPriceResetChangePublic]
    skipped_no_default: list[str]
    applied: bool


class HostedModelsRefreshPublic(SQLModel):
    """What a refresh did: the models newly offered, the seeded rates it moved, and the list's new size.

    Failure is a field rather than a status, for the reason
    ``HostedAvailableModelsPublic`` gives: the list is still standing, and the
    panel renders the reason beside it.
    """

    added: list[str]
    repriced: list[str]
    count: int
    error: str | None = None
    discovery_unsupported: bool = False


class HostedCatalogProviderRefreshPublic(SQLModel):
    """What one hosted provider newly lists, as a catalog sweep found it.

    Per provider rather than one total, because a dial that failed is a fact
    about that provider, and the operator has to be told which one went quiet
    rather than read a smaller number than they expected.
    """

    provider: str
    added: list[str]
    # Empty in a preview: moving a seeded price is a write.
    repriced: list[str] = Field(default_factory=list)
    error: str | None = None
    discovery_unsupported: bool = False
    # The stored key would not decrypt here, as against being refused upstream,
    # so the page does not offer to remove a provider whose credential merely
    # needs re-entering.
    credential_unreadable: bool = False
    # Where the provider is defined, so the page offers to remove only one it
    # can delete: a provider from configuration is changed there.
    source: Literal["database", "config"] = "database"


class HostedCatalogKeptGroupPublic(SQLModel):
    """Priced models a sweep left in place, grouped by the reason it did."""

    reason: str
    count: int
    # A sample; ``count`` is the whole group.
    models: list[str]


class HostedCatalogRefreshPublic(SQLModel):
    """What a catalog sweep did, or would do.

    ``removed`` is a sample and ``removed_count`` the total, so a deployment
    that has never been swept does not answer one button press with a few
    thousand strings. ``orphaned_rosters`` names the providers that still have
    a model list but nothing defining them, such as a removed ``config.yml``
    entry: nothing serves them, and an apply deletes their model lists.
    """

    providers: list[HostedCatalogProviderRefreshPublic]
    removed: list[str]
    removed_count: int
    removed_price_rows: int
    removed_override_rows: int
    kept: list[HostedCatalogKeptGroupPublic]
    orphaned_rosters: list[str] = Field(default_factory=list)


class HostedCatalogPriceChangePublic(SQLModel):
    """One model whose models.dev price a catalog update adds, changes or removes."""

    model_key: str
    change: Literal["added", "changed", "removed"]


class HostedCatalogPriceUpdatePublic(SQLModel):
    """The models.dev price changes a catalog update would apply.

    ``changes`` is a sample and the counts are the totals. ``digest`` names the
    fetched prices, so applying them applies what was reviewed. ``needs_review``
    is why the scheduled update would hold them for a person under ``auto``.
    """

    fetched_at: datetime
    added_count: int
    changed_count: int
    removed_count: int
    changes: list[HostedCatalogPriceChangePublic]
    changes_truncated: bool
    digest: str
    needs_review: bool
    review_reason: str | None = None


class HostedCatalogUpdatePreviewRequest(SQLModel):
    """What to preview: the update alone, or with the cleanup."""

    cleanup: bool = Field(
        default=False,
        description="Also preview removing the prices of models no hosted provider serves.",
    )


class HostedCatalogUpdateRequest(SQLModel):
    """What to apply, as the preview showed it."""

    digest: str | None = Field(
        default=None,
        description=(
            "The `digest` of the previewed prices, to apply them; 409 when the pending prices differ. "
            "Without it, no price changes."
        ),
    )
    cleanup: bool = Field(
        default=False,
        description="Also remove the prices of models no hosted provider serves, with their history.",
    )


class HostedCatalogUpdatePreviewPublic(SQLModel):
    """What a catalog update would do: its price changes, then what each provider newly lists.

    ``prices`` is ``None`` when models.dev changed nothing, or when it could not
    be fetched, which ``price_error`` then says. ``catalog`` carries the cleanup
    the preview was asked for, and no removals otherwise.
    """

    prices: HostedCatalogPriceUpdatePublic | None
    price_error: str | None = None
    catalog: HostedCatalogRefreshPublic


class HostedCatalogUpdatePublic(SQLModel):
    """What a catalog update did."""

    prices_applied: bool
    catalog: HostedCatalogRefreshPublic


class HostedAvailableModelsPublic(SQLModel):
    """What the provider says it serves on the stored credential.

    Failure is a field rather than a status: an unreachable upstream, or a
    provider with no model listing, is an answer about the provider rather than
    about this request, and the form still has to render a plain text box when
    the list cannot be fetched.
    """

    provider: str
    models: list[str] = Field(default_factory=list)
    error: str | None = None
    discovery_unsupported: bool = False


class WorkspaceHostedProviderAccessRequest(SQLModel):
    """What one workspace may do with one hosted provider. Replaces whatever it stored before."""

    enabled: bool = Field(description="False switches the provider off for this workspace.")
    models: list[str] | None = Field(
        default=None,
        max_length=1000,
        description=(
            "The models this workspace may call on the provider. Null for every model the deployment serves on it, "
            "including ones added later. A list must name at least one model."
        ),
    )


class WorkspaceHostedProviderPublic(SQLModel):
    """One hosted provider as one workspace sees it."""

    provider: str = Field(description="The hosted provider's name, as a request's selector prefix spells it.")
    enabled: bool
    models: list[str] | None = Field(description="This workspace's model list, or null for every served model.")
    customized: bool = Field(description="Whether this workspace stores a departure from full inheritance.")
    served_models: list[str] | None = Field(
        description=(
            "The models the deployment serves on the provider for this organization, or null for any priced one."
        ),
    )
    served_by_own_key: bool = Field(
        description="Whether the workspace calls this provider on the organization's own key, so this row is not used."
    )
    zero_data_retention: bool = Field(
        default=False, description="Whether the deployment declares this provider keeps no prompt or completion data."
    )
    data_locations: list[str | None] = Field(
        default_factory=list,
        description=(
            "Where each region the provider serves from declares it keeps request data (one entry for a provider "
            "with no regions), null where nothing is declared. A workspace listing allowed data locations is "
            "refused the provider when none of these is on its list."
        ),
    )


class WorkspaceHostedProvidersPublic(SQLModel):
    data: list[WorkspaceHostedProviderPublic]


class WorkspaceProviderPolicyRequest(SQLModel):
    """What one workspace requires of every provider serving it."""

    require_zero_data_retention: bool | None = Field(
        default=None,
        description=(
            "Refuse every provider not declared zero data retention: an organization key or a hosted provider "
            "nobody declared, and every provider configured in config.yml or endpoint the workspace owns. "
            "Omitted keeps the stored setting."
        ),
    )

    allowed_data_locations: list[str] | None = Field(
        default=None,
        description=(
            "The data locations a hosted provider region must declare to serve the workspace: ISO 3166-1 alpha-2 "
            "country codes, EU (which also admits each member state) or GLOBAL. Null accepts any location, and "
            "omitting the field keeps the stored list. A region whose location nobody declared never matches a list."
        ),
        max_length=300,
    )

    @field_validator("allowed_data_locations")
    @classmethod
    def _data_locations(cls, value: list[str] | None) -> list[str] | None:
        return data_locations(value) if value is not None else None


class WorkspaceProviderPolicyPublic(SQLModel):
    require_zero_data_retention: bool
    allowed_data_locations: list[str] | None = None


__all__ = [
    "DeploymentProviderMigrationOutcome",
    "DeploymentProviderMigrationPublic",
    "DeploymentProviderMigrationRequest",
    "DeploymentProviderMigrationReviewedRow",
    "DeploymentProviderMigrationRowPublic",
    "EndpointAuthPublic",
    "EndpointAuthRequest",
    "HostedAvailableModelsPublic",
    "HostedCatalogKeptGroupPublic",
    "HostedCatalogProviderRefreshPublic",
    "HostedCatalogRefreshPublic",
    "HostedModelCreateRequest",
    "HostedModelPublic",
    "HostedModelRates",
    "HostedModelUpdateRequest",
    "HostedModelsPublic",
    "HostedModelsRefreshPublic",
    "HostedProviderCreateRequest",
    "HostedProviderPublic",
    "HostedProviderRegionCreateRequest",
    "HostedProviderRegionPublic",
    "HostedProviderRegionUpdateRequest",
    "HostedProviderTestPublic",
    "HostedProviderTestRequest",
    "HostedProviderUpdateRequest",
    "HostedProvidersPublic",
    "HostedProvidersReencryptPublic",
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
    "ProviderEndpointCreateRequest",
    "ProviderEndpointPublic",
    "ProviderEndpointUpdateRequest",
    "ProviderEndpointsPublic",
    "WorkspaceProviderKeyOverridePublic",
    "WorkspaceHostedProviderAccessRequest",
    "WorkspaceHostedProviderPublic",
    "WorkspaceHostedProvidersPublic",
    "WorkspaceProviderPolicyPublic",
    "WorkspaceProviderPolicyRequest",
    "WorkspaceProviderKeyOverrideRequest",
    "WorkspaceProviderKeyOverridesPublic",
    "WorkspaceProviderModelRestrictionRequest",
    "WorkspaceProviderModelRestrictionsPublic",
]


class PlatformConnectStartPublic(SQLModel):
    """What an operator needs to approve a connection on otari.ai.

    ``user_code`` is what the approval page shows back, ``verification_uri_complete``
    the page to open, and ``expires_in`` the seconds left to approve.
    """

    user_code: str
    verification_uri_complete: str
    expires_in: int


class PlatformConnectStatusPublic(SQLModel):
    """Where the deployment's connection to otari.ai stands.

    ``idle`` means no attempt. The code, the link and the seconds left are set
    while ``pending`` and null otherwise. ``connected`` means the ``otari-ai``
    hosted provider holds the key otari.ai issued.
    """

    status: Literal["idle"] | PlatformConnectState
    user_code: str | None = None
    verification_uri_complete: str | None = None
    expires_in: int | None = None
