"""Request and response shapes for the web search tools.

An organization's web search keys and their workspace overrides, and the
catalog of providers a search or fetch tool may name.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from sqlmodel import Field, SQLModel

from gateway.core.settings.tools import ToolKind
from gateway.models.tools import OrgWebSearchKey


class OrgWebSearchKeyCreateRequest(SQLModel):
    """What a caller sends to add a key. The service keeps only its ciphertext and ``last4``."""

    model_config = {
        "json_schema_extra": {
            "example": {"provider": "tavily", "name": "production", "api_key": "<your Tavily API key>"}
        }
    }

    provider: str = Field(max_length=255, description="The search provider the key is for: tavily or brave.")
    name: str = Field(max_length=255, description="How the organization tells this key apart from its others.")
    api_key: str = Field(min_length=1)


class OrgWebSearchKeyUpdateRequest(SQLModel):
    """A partial update: only what is set is applied."""

    name: str | None = Field(default=None, max_length=255)
    api_key: str | None = Field(default=None, min_length=1)


class OrgWebSearchKeyPublic(SQLModel):
    """One key as the API shows it: never the key itself, only ``last4``."""

    id: uuid.UUID
    organization_id: uuid.UUID
    provider: str
    name: str
    last4: str | None = None
    is_org_default: bool
    usable: bool = Field(
        description=(
            "False when this deployment cannot decrypt the stored key, so no search uses it. "
            "It is still listed, because replacing or deleting it is what fixes it."
        )
    )
    archived_at: datetime | None = None
    created_at: datetime
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, key: OrgWebSearchKey, *, usable: bool) -> OrgWebSearchKeyPublic:
        """Read one row for the API; ``usable`` comes from the service, which owns decryption."""
        return cls(
            id=key.id,
            organization_id=key.organization_id,
            provider=key.provider,
            name=key.name,
            last4=key.last4,
            is_org_default=key.is_org_default,
            usable=usable,
            archived_at=key.archived_at,
            created_at=key.created_at,
            updated_at=key.updated_at,
        )


class OrgWebSearchKeysPublic(SQLModel):
    data: list[OrgWebSearchKeyPublic]
    count: int


class WorkspaceWebSearchKeyOverrideRequest(SQLModel):
    """Tri-state: an omitted flag keeps its value.

    Pinning a key re-enables it and unpins any other key of the workspace, and turning a
    key off unpins it. Sending both flags true is refused. Both false deletes the override.
    """

    model_config = {"json_schema_extra": {"example": {"is_default": True}}}

    is_default: bool | None = None
    disabled: bool | None = None


class WorkspaceWebSearchKeyPublic(SQLModel):
    """One of the organization's keys, as one workspace sees it."""

    org_web_search_key_id: uuid.UUID
    workspace_id: uuid.UUID
    provider: str
    name: str
    last4: str | None = None
    is_default: bool = Field(description="The workspace pinned this key as its own.")
    disabled: bool = Field(description="The workspace turned this key off.")
    is_effective: bool = Field(description="This is the key the workspace's searches use.")
    usable: bool


class WorkspaceWebSearchKeysPublic(SQLModel):
    data: list[WorkspaceWebSearchKeyPublic]
    count: int = Field(description="Every live key of the organization, not only this page.")


class SearchProviderOptionSchema(SQLModel):
    """One native option a provider accepts, under the provider's own name."""

    name: str
    type: Literal["string", "integer", "number", "boolean", "array", "object"]
    enum: list[str] | None = Field(default=None, description="The only values it takes, when it is limited to a list.")
    default: Any = Field(default=None, description="The value the provider uses when the option is not set.")
    description: str = ""
    operator_only: bool = Field(
        default=False,
        description="True when only an instance or a credential may set it, never a workspace or a request.",
    )


class SearchProviderSchema(SQLModel):
    """One provider a search or fetch instance may name, for the add-tool form.

    One schema for both capabilities: the fields they share, then each one's
    own, which are null on the other's entries.
    """

    id: str = Field(description="Value to send as 'provider'.")
    kind: ToolKind = Field(description="Whether this is a search provider or a fetch provider.")
    requires_api_key: bool = Field(description="True when a tool on this provider must carry an API key.")
    requires_api_base: bool = Field(
        description="True when this provider has no endpoint of its own, so the tool must say where the backend is."
    )
    default_api_base: str | None = Field(
        default=None,
        description=(
            "The endpoint a tool on this provider uses when it declares no api_base. "
            "Null means nothing supplies one, so an api_base is required. One that comes from the "
            "deployment's own settings, such as the web_search_url a searxng tool inherits, is shown only "
            "to a caller who operates the deployment: nothing else inherits it, an organization's key included."
        ),
    )
    doc_url: str | None = Field(default=None, description="The provider's API documentation.")
    tier: str | None = Field(default=None, description="The library's tier for the provider.")
    options: list[SearchProviderOptionSchema] | None = Field(
        default=None,
        description=(
            "The native options a tool may set, under the provider's own names. "
            "Null when there is no schema for the provider yet, so a tool's options are passed unchecked."
        ),
    )
    instances: list[str] = Field(
        default_factory=list,
        description=(
            "The names of this deployment's configured and stored tools on this provider, "
            "shown only to a caller who operates the deployment."
        ),
    )
    max_results: int | None = Field(default=None, description="Search: the most results one call can ask for.")
    query_in_url: bool | None = Field(
        default=None, description="Search: true when the query travels in the request URL."
    )
    key_in_url: bool | None = Field(
        default=None, description="Search: true when the API key travels in the request URL."
    )
    max_urls_per_call: int | None = Field(default=None, description="Fetch: the most pages one call can fetch.")
    renders_javascript: bool | None = Field(
        default=None, description="Fetch: true when the provider runs a page's JavaScript before reading it."
    )
    formats: list[str] | None = Field(default=None, description="Fetch: the formats the page text comes back in.")
