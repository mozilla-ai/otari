"""Request and response shapes for the web search tools.

An organization's web search keys and their workspace overrides, and the
catalog of providers a search or fetch tool may name.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

import pydantic
from sqlmodel import Field, SQLModel

from gateway.core.settings.tools import ToolKind
from gateway.models.tools import OrgWebSearchKey, SearchToolCredential


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


# /search-tools management: stored and configured instances, writes and connection tests.


class StoredSearchToolSchema(pydantic.BaseModel):
    """A runtime-stored search or fetch instance. The API key is never returned, only ``last4``."""

    name: str
    kind: ToolKind = pydantic.Field(default="search", description="Whether this is a search or a fetch instance.")
    provider: str
    fetch_tool: str | None = pydantic.Field(
        default=None,
        description="A search instance's enrichment fetch instance. Null means the fetch default enriches it.",
    )
    api_base: str | None = None
    last4: str | None = None
    timeout: float | None = None
    options: dict[str, Any] = pydantic.Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    # False when the stored key cannot be decrypted with the current
    # OTARI_SECRET_KEY. Such a tool is skipped at runtime, so the dashboard flags
    # it for the operator to fix.
    decryptable: bool = True
    shadows_config: bool = pydantic.Field(
        default=False,
        description="True when a config-file search tool of the same name exists; the stored one is in effect.",
    )

    @classmethod
    def from_model(
        cls,
        row: SearchToolCredential,
        *,
        decryptable: bool = True,
        shadows_config: bool = False,
    ) -> "StoredSearchToolSchema":
        return cls(**row.to_public_dict(), decryptable=decryptable, shadows_config=shadows_config)


class CreatedSearchToolSchema(StoredSearchToolSchema):
    """A stored instance as created, with the default setting the create also stored, if any."""

    pinned_web_search_default_tool: str | None = pydantic.Field(
        default=None,
        description=(
            "Set when this create also stored web_search_default_tool, naming the search instance that was "
            "the in-loop default because it was the only one, so that adding a second does not turn in-loop "
            "search off. This runtime value wins over the configuration file until it is cleared."
        ),
    )
    notice: str | None = pydantic.Field(
        default=None, description="What else the create changed, for the operator to read."
    )


class ConfigSearchToolSchema(pydantic.BaseModel):
    """A search or fetch instance declared in the config file, or ``builtin_fetch``. Read-only here."""

    name: str
    kind: ToolKind = pydantic.Field(default="search", description="Whether this is a search or a fetch instance.")
    provider: str
    fetch_tool: str | None = pydantic.Field(
        default=None,
        description="A search instance's enrichment fetch instance. Null means the fetch default enriches it.",
    )
    api_base: str | None = None
    has_api_key: bool = pydantic.Field(
        description="Whether the config entry carries an API key. The key itself is not shown."
    )
    shadowed: bool = pydantic.Field(
        default=False,
        description="True when a stored instance of the same name and kind overrides this entry.",
    )


class SearchToolsResponse(pydantic.BaseModel):
    """Every search instance ``POST /api/v1/search`` can name, or every fetch instance, by where it came from."""

    stored: list[StoredSearchToolSchema]
    config: list[ConfigSearchToolSchema]


class CreateSearchToolRequest(pydantic.BaseModel):
    """Create a stored search or fetch instance. ``api_key`` is write-only and requires OTARI_SECRET_KEY."""

    model_config = pydantic.ConfigDict(
        json_schema_extra={"example": {"name": "local", "provider": "searxng", "api_base": "http://searxng:8080"}}
    )

    name: str = pydantic.Field(
        min_length=1,
        pattern=r"^[^/:]+$",
        description=(
            "Name callers pass as 'search_tool_name' or in /api/v1/search/{tool}, or that names a fetch "
            "instance. It contains no '/' or ':', is not builtin_fetch or none in any case, and is unique "
            "across search and fetch instances."
        ),
    )
    kind: ToolKind = pydantic.Field(default="search", description="'search' or 'fetch'. It cannot change once created.")
    provider: str = pydantic.Field(
        description=(
            "Provider id. GET /api/v1/search-tools/providers lists the search providers, and with "
            "?kind=fetch the fetch providers."
        )
    )
    fetch_tool: str | None = pydantic.Field(
        default=None,
        description=(
            "For a search instance: the fetch instance that enriches its results, a configured or stored one "
            "or builtin_fetch. Omit it for the fetch default."
        ),
    )
    api_base: str | None = pydantic.Field(
        default=None,
        description="Backend endpoint. Omit to inherit the provider's default (searxng inherits web_search_url).",
    )
    api_key: str | None = pydantic.Field(
        default=None, description="Provider API key. Stored encrypted; never returned."
    )
    timeout: float | None = pydantic.Field(default=None, gt=0, description="Per-request timeout in seconds.")
    options: dict[str, Any] | None = pydantic.Field(
        default=None,
        description="Provider-native request fields used as defaults (e.g. exa's 'type', searxng's 'engines').",
    )


class UpdateSearchToolRequest(pydantic.BaseModel):
    """Update a stored search or fetch instance. Omitted fields are unchanged; ``api_key`` rotates in place."""

    kind: ToolKind | None = pydantic.Field(
        default=None, description="Accepted only when it matches the stored kind, which cannot change."
    )
    provider: str | None = None
    fetch_tool: str | None = pydantic.Field(
        default=None,
        description="For a search instance: the fetch instance that enriches its results. Null clears it.",
    )
    api_base: str | None = None
    api_key: str | None = pydantic.Field(
        default=None, description="New API key. Omit to keep the existing one. Never returned."
    )
    timeout: float | None = pydantic.Field(default=None, gt=0)
    options: dict[str, Any] | None = None
    expected_updated_at: str | None = pydantic.Field(
        default=None,
        description="Optimistic concurrency: if set, the update 412s unless it matches the stored updated_at.",
    )


class ReencryptSearchToolsResponse(pydantic.BaseModel):
    """Result of re-encrypting stored search-tool keys with the primary secret key."""

    reencrypted: int = pydantic.Field(description="Number of stored search-tool keys re-encrypted.")
    unreadable: int = pydantic.Field(
        description="Number of encrypted keys left untouched because they could not be decrypted."
    )
    skipped: int = pydantic.Field(
        default=0,
        description=(
            "Number of rows whose stored key changed between the read and the write, so the "
            "re-encryption was not applied. They already hold whoever wrote them last."
        ),
    )


class SearchToolTestRequest(CreateSearchToolRequest):
    """An unsaved search or fetch instance to test, and what to test it with."""

    query: str | None = pydantic.Field(
        default=None, min_length=1, description="For a search instance: the query to run."
    )
    url: str | None = pydantic.Field(default=None, min_length=1, description="For a fetch instance: the page to fetch.")


class StoredSearchToolTestRequest(pydantic.BaseModel):
    """What to test a configured or stored instance with."""

    query: str | None = pydantic.Field(
        default=None, min_length=1, description="For a search instance: the query to run."
    )
    url: str | None = pydantic.Field(default=None, min_length=1, description="For a fetch instance: the page to fetch.")


class SearchToolTestResponse(pydantic.BaseModel):
    """How one search or one fetch went. Never the results or the page."""

    ok: bool = pydantic.Field(description="Whether the provider answered the call without an error.")
    error: str | None = pydantic.Field(
        default=None,
        description=(
            "When not ok, the error's tag: timeout, network, http_error, invalid_response, or the provider's own."
        ),
    )
    hits: int | None = pydantic.Field(default=None, description="For a search that worked: how many hits came back.")
    characters: int | None = pydantic.Field(
        default=None, description="For a fetch that worked: how many characters of page text came back."
    )
