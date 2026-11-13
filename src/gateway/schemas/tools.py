"""Request and response shapes for the web search tools.

An organization's web search keys and their workspace overrides, and the
catalog of providers a search or fetch tool may name.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Self

import pydantic
from sqlmodel import Field, SQLModel

from gateway.core.settings.tools import ToolKind
from gateway.models.secret_fields import redact_secret_like_values
from gateway.models.tools import OrgWebSearchKey, SearchToolCredential


class OrgWebSearchKeyCreateRequest(SQLModel):
    """What a caller sends to add a key. The service keeps only its ciphertext and ``last4``."""

    model_config = {
        "json_schema_extra": {
            "example": {"kind": "search", "provider": "exa", "name": "production", "api_key": "<your Exa API key>"}
        }
    }

    kind: ToolKind = Field(
        default="search",
        description="Whether the key serves the search tool or the fetch tool, and enrichment. Fixed once created.",
    )
    provider: str = Field(
        max_length=255,
        description=(
            "The provider the key is for: one the provider catalog, GET /api/v1/search-tools/providers, "
            "lists for the key's kind, other than searxng, or tavily or brave for a search key."
        ),
    )
    name: str = Field(max_length=255, description="How the organization tells this key apart from its others.")
    api_key: str | None = Field(default=None, min_length=1, description="Required unless the provider takes no key.")
    api_base: str | None = Field(
        default=None,
        max_length=1024,
        description="The provider's endpoint, when not its own. Must be https when the key carries an API key.",
    )
    options: dict[str, Any] | None = Field(
        default=None, description="Provider options, under the names the provider catalog lists for the provider."
    )


class OrgWebSearchKeyUpdateRequest(SQLModel):
    """A partial update: only what is set is applied, and ``null`` clears ``api_base`` or ``options``.

    ``kind`` is fixed once a key is created, so sending it is refused.
    """

    name: str | None = Field(default=None, max_length=255)
    api_key: str | None = Field(default=None, min_length=1)
    api_base: str | None = Field(default=None, max_length=1024)
    options: dict[str, Any] | None = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def _refuse_kind(cls, data: Any) -> Any:
        if isinstance(data, dict) and "kind" in data:
            msg = "kind cannot be changed; add a key of the other kind instead"
            raise ValueError(msg)
        return data


class OrgWebSearchKeyPublic(SQLModel):
    """One key as the API shows it: never the key itself, only ``last4``."""

    id: uuid.UUID
    organization_id: uuid.UUID
    kind: ToolKind
    provider: str
    name: str
    last4: str | None = None
    api_base: str | None = None
    options: dict[str, Any] | None = Field(
        default=None, description="The key's provider options, with any value whose name looks like a secret masked."
    )
    is_org_default: bool
    usable: bool = Field(
        description=(
            "False when this deployment cannot decrypt the stored key, so no call uses it. "
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
            kind="fetch" if key.kind == "fetch" else "search",
            provider=key.provider,
            name=key.name,
            last4=key.last4,
            api_base=key.api_base,
            options=redact_secret_like_values(key.options),
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

    Pinning a key re-enables it and unpins the workspace's other key of its kind, and turning a
    key off unpins it. Sending both flags true is refused. Both false deletes the override.
    """

    model_config = {"json_schema_extra": {"example": {"is_default": True}}}

    is_default: bool | None = None
    disabled: bool | None = None


class WorkspaceWebSearchKeyPublic(SQLModel):
    """One of the organization's keys, as one workspace sees it."""

    org_web_search_key_id: uuid.UUID
    workspace_id: uuid.UUID
    kind: ToolKind
    provider: str
    name: str
    last4: str | None = None
    is_default: bool = Field(description="The workspace pinned this key as its own for its kind.")
    is_org_default: bool = Field(
        default=False,
        description=(
            "The organization's default of its kind, which a workspace that pinned no key uses unless it turned the "
            "key off or this deployment cannot decrypt it; the deployment's instance serves it then."
        ),
    )
    disabled: bool = Field(description="The workspace turned this key off.")
    is_effective: bool = Field(description="This is the key the workspace's calls of its kind use.")
    usable: bool


class WorkspaceWebSearchKeysPublic(SQLModel):
    data: list[WorkspaceWebSearchKeyPublic]
    count: int = Field(description="Every live key of the organization, not only this page.")
    search_provider: str | None = Field(
        default=None,
        description=(
            "The provider the workspace's searches run on: its effective search key's, else the deployment's "
            "default search's. Null when nothing searches for it."
        ),
    )
    fetch_provider: str | None = Field(
        default=None,
        description=(
            "The provider the workspace's fetch tool runs on, and the enrichment of its search hits unless the "
            "search's instance names a fetch instance of its own: its effective fetch key's, else the "
            "deployment's default fetch instance's, which is builtin for the built-in fetcher."
        ),
    )
    deployment_search_tool: str | None = Field(
        default=None,
        description=(
            "The name of the deployment's search tool, which the workspace's searches run on when no key serves "
            "them, whether or not one does now. Null when the deployment has none."
        ),
    )
    deployment_fetch_tool: str | None = Field(
        default=None,
        description=(
            "The name of the deployment's fetch tool, which the workspace's fetches run on when no key serves "
            "them, whether or not one does now: builtin_fetch for the built-in fetcher."
        ),
    )


class SearchProviderOptionSchema(SQLModel):
    """One native option a provider accepts, under the provider's own name."""

    name: str
    type: Literal["string", "integer", "number", "boolean", "array", "object"]
    enum: list[str] | None = Field(default=None, description="The only values it takes, when it is limited to a list.")
    default: Any = Field(default=None, description="The value the provider uses when the option is not set.")
    description: str = ""
    operator_only: bool = Field(
        default=False,
        description=(
            "True when only an instance or a credential may set it, never a workspace or a request: an option "
            "the provider's library marks so, or a cost tier of a provider that reports no cost."
        ),
    )
    cost_tier: bool = Field(
        default=False,
        description=(
            "True when the option's values are the provider's cost tiers: the provider bills a call at a price "
            "of its own for each value. enum lists the tiers, and default is the one used when the option is unset."
        ),
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
    reports_cost: bool = Field(
        default=False,
        description="True when the provider's answer says what each call cost, which is then what the call is billed.",
    )
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
    reports_final_url: bool | None = Field(
        default=None,
        description=(
            "Fetch: true when the provider reports where a page was found after redirects. When false it "
            "answers with the URL as asked, so a redirect it follows is not checked against the domain policy."
        ),
    )


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
        **fields: Any,
    ) -> Self:
        """The public view of a stored row: never the key, only its last four characters.

        ``fields`` sets those a subclass adds.
        """
        return cls(**row.to_public_dict(), decryptable=decryptable, shadows_config=shadows_config, **fields)


# Where a listed tool's price comes from: its own price row, that row while its
# config.yml entry sets ``price_per_call``, its old ``<provider>:<name>`` row,
# or no row.
ToolPriceSource = Literal["own", "config", "old_key", "none"]

_PRICE_PER_CALL_DESCRIPTION = (
    "USD for one call on the deployment's own account when the provider reports no cost of its own, from the "
    "deployment's price list. An override the operator set for one organization prices that organization's "
    "calls instead. Null when no row prices the tool: with require_pricing on, its calls are refused."
)
_PRICE_SOURCE_DESCRIPTION = (
    "Which row price_per_call comes from: 'own', the tool's own row (its kind and name as model_key); 'config', "
    "that row while the tool's config.yml entry sets price_per_call, which each start writes to it again when the "
    "rate in force differs, so a rate set through the API lasts until the next start; 'old_key', the tool's old "
    "<provider>:<name> row, which prices it while it has no row of its own; 'none', no row."
)


class ListedSearchToolSchema(StoredSearchToolSchema):
    """A stored instance as the listing shows it: with what one of its calls costs."""

    price_per_call: float | None = pydantic.Field(description=_PRICE_PER_CALL_DESCRIPTION)
    price_source: ToolPriceSource = pydantic.Field(description=_PRICE_SOURCE_DESCRIPTION)


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
    price_per_call: float | None = pydantic.Field(description=_PRICE_PER_CALL_DESCRIPTION)
    price_source: ToolPriceSource = pydantic.Field(description=_PRICE_SOURCE_DESCRIPTION)


class SearchToolsResponse(pydantic.BaseModel):
    """Every search instance ``POST /api/v1/search`` can name, or every fetch instance, by where it came from."""

    stored: list[ListedSearchToolSchema]
    config: list[ConfigSearchToolSchema]


class SearchToolCatalogEntry(pydantic.BaseModel):
    """One search tool a direct search may name, as any caller may see it: never its key, endpoint or options."""

    id: str = pydantic.Field(description="The tool's name, to send as 'search_tool_name' or in /api/v1/search/{tool}.")
    object: Literal["search_tool"] = "search_tool"
    provider: str = pydantic.Field(description="The search provider the tool runs on.")
    label: str = pydantic.Field(description="A display name for the tool: its provider's name, capitalized.")
    price_per_call: float | None = pydantic.Field(
        description=(
            "USD the caller's organization is charged for one search on the tool when the provider reports no "
            "cost of its own: the organization's override, else the deployment's price list. Null when no row "
            "prices it."
        )
    )
    default: bool = pydantic.Field(
        description=(
            "True for the deployment's default, the tool a direct search that names none runs on unless the "
            "caller's organization brings its own search key."
        )
    )


class SearchToolCatalogResponse(pydantic.BaseModel):
    """The search tools a direct search may name."""

    object: Literal["list"] = "list"
    data: list[SearchToolCatalogEntry]


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

    def to_entry(self) -> dict[str, Any]:
        """The request as a configuration file entry, the shape the checks and the overlay read."""
        fetch_tool = self.fetch_tool.strip() if self.fetch_tool is not None else None
        return {
            "provider": self.provider,
            "fetch_tool": fetch_tool,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "options": self.options,
        }


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
            "When not ok, the error's tag: timeout, network, http_error, invalid_response, unexpected_response "
            "(an answer whose shape the provider changed), or the provider's own."
        ),
    )
    hits: int | None = pydantic.Field(default=None, description="For a search that worked: how many hits came back.")
    characters: int | None = pydantic.Field(
        default=None, description="For a fetch that worked: how many characters of page text came back."
    )


# Perplexity's documented ceiling on search_domain_filter, mirrored so an
# oversized list is rejected here rather than by the upstream provider.
_MAX_DOMAIN_FILTERS = 20


class SearchRequest(pydantic.BaseModel):
    """A direct search request: LiteLLM's ``/v1/search`` body, plus ``include_raw`` and ``provider_options``."""

    model_config = pydantic.ConfigDict(
        json_schema_extra={"example": {"query": "otari llm gateway", "max_results": 5}},
    )

    query: str = pydantic.Field(min_length=1, description="The search query")
    search_tool_name: str | None = pydantic.Field(
        default=None,
        description=(
            "Configured search tool to run against. Optional when the deployment names a default "
            "(web_search_default_tool) or has exactly one tool, and ignored on "
            "POST /api/v1/search/{search_tool_name}."
        ),
    )
    max_results: int | None = pydantic.Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of results to return; 10 when unset. Lowered to the workspace's ceiling "
            "when it sets one, and to the provider's own maximum."
        ),
    )
    search_domain_filter: list[str] | None = pydantic.Field(
        default=None,
        max_length=_MAX_DOMAIN_FILTERS,
        description="Restrict results to these domains; prefix a domain with '-' to exclude it instead",
    )
    country: str | None = pydantic.Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Two-letter ISO country code to localize results to",
    )
    max_tokens_per_page: int | None = pydantic.Field(
        default=None, ge=1, description="Approximate cap on the page content returned per result"
    )
    include_raw: bool = pydantic.Field(default=False, description="Also return the provider's whole response, as raw.")
    provider_options: dict[str, Any] | None = pydantic.Field(
        default=None,
        description=(
            "Native options for the provider that serves the call, under the provider's own names, as "
            "GET /api/v1/search-tools/providers lists them. An option the provider does not take from a "
            "request is refused with a 400. On a call that names no tool they go to whichever provider "
            "serves it, so name the tool when they are meant for one provider."
        ),
    )
    user: str | None = pydantic.Field(default=None, description="User ID for usage attribution")


class SearchResultItem(pydantic.BaseModel):
    """One search result, shaped as LiteLLM's ``SearchResult``."""

    title: str = pydantic.Field(description="The page's title; empty when the provider gave none.")
    url: str
    snippet: str = pydantic.Field(
        description="The page text the provider returned, else its excerpt, cut to the page budget."
    )
    date: str | None = pydantic.Field(default=None, description="The page's date, as the provider gave it.")
    last_updated: str | None = pydantic.Field(default=None, description="Always null: no provider reports it.")


class SearchResponse(pydantic.BaseModel):
    """A completed direct search."""

    object: Literal["search"] = "search"
    search_tool: str = pydantic.Field(description="The configured search tool that served the request")
    results: list[SearchResultItem]
    raw: dict[str, Any] | None = pydantic.Field(
        default=None,
        exclude_if=lambda raw: raw is None,
        description="The provider's whole response, untouched; only when the request set include_raw.",
    )


class FetchRequest(pydantic.BaseModel):
    """A direct fetch request: one page, read through the deployment's fetch tool or the one named."""

    model_config = pydantic.ConfigDict(
        json_schema_extra={"example": {"url": "https://example.com/article", "max_chars": 20000}},
    )

    url: str = pydantic.Field(min_length=1, description="The page to fetch: one public http or https URL.")
    fetch_tool_name: str | None = pydantic.Field(
        default=None,
        description=(
            "Configured fetch tool to run on, builtin_fetch included. Optional: the deployment's "
            "web_fetch_default_tool serves, else builtin_fetch. Ignored on POST /api/v1/fetch/{fetch_tool_name}."
        ),
    )
    max_chars: int | None = pydantic.Field(
        default=None,
        ge=1,
        description=(
            "Cut the page text to this many characters. Unset, the tool's own max_chars option applies, "
            "else the provider's own limits."
        ),
    )
    include_raw: bool = pydantic.Field(
        default=False,
        description=(
            "Also return the provider's whole response, as raw. builtin_fetch calls no provider, so its raw "
            "holds only content_kind and redirect_count."
        ),
    )
    provider_options: dict[str, Any] | None = pydantic.Field(
        default=None,
        description=(
            "Native options for the provider that serves the call, under the provider's own names, as "
            "GET /api/v1/search-tools/providers?kind=fetch lists them. An option the provider does not take from a "
            "request is refused with a 400. On a call that names no tool they go to whichever provider "
            "serves it, so name the tool when they are meant for one provider."
        ),
    )
    user: str | None = pydantic.Field(default=None, description="User ID for usage attribution")


class FetchResponse(pydantic.BaseModel):
    """A completed direct fetch: one page's text and where it came from."""

    object: Literal["fetch"] = "fetch"
    fetch_tool: str = pydantic.Field(description="The configured fetch tool that served the request")
    url: str = pydantic.Field(description="The URL fetched, as asked, in canonical form.")
    final_url: str = pydantic.Field(
        description=(
            "Where the page was found after redirects. The URL asked for when the provider does not report "
            "where a redirect led (reports_final_url in the provider catalog)."
        )
    )
    title: str = pydantic.Field(default="", description="The page's title; empty when the provider gave none.")
    content_type: str = pydantic.Field(
        default="", description="The page's content type; empty when the provider does not report one."
    )
    text: str = pydantic.Field(description="The page's text, as the provider extracted it.")
    published: datetime | None = pydantic.Field(
        default=None, description="When the page was published, where the provider reports it."
    )
    truncated: bool = pydantic.Field(
        description=(
            "Whether text is less than the whole page: the body or the extracted text hit a limit of the "
            "provider's, or max_chars cut it."
        )
    )
    raw: dict[str, Any] | None = pydantic.Field(
        default=None,
        exclude_if=lambda raw: raw is None,
        description="The provider's whole response, untouched; only when the request set include_raw.",
    )


_MAX_CODE_EXECUTION_INPUTS = 32


class CodeExecuteRequest(pydantic.BaseModel):
    """A direct code execution request: the code to run, and the sandbox to run it in."""

    model_config = pydantic.ConfigDict(
        json_schema_extra={"example": {"code": "print(1 + 1)", "container": "auto"}},
    )

    code: str = pydantic.Field(min_length=1, description="Python source to run in the sandbox")
    container: str | None = pydantic.Field(
        default=None,
        description=(
            "'auto' to hold the sandbox after the run, or a container id an earlier response named in "
            "Otari-Container-Id to run in that sandbox again, with its files and interpreter state. "
            "Omitted, the sandbox is released when the run ends."
        ),
    )
    timeout_seconds: int | None = pydantic.Field(
        default=None,
        ge=1,
        description=(
            "How long the code may run. Never more than the workspace's code execution ceiling, "
            "or 60 seconds when it sets none."
        ),
    )
    environment: str | None = pydantic.Field(
        default=None,
        max_length=255,
        pattern=r"^[A-Za-z0-9._:/@-]+$",
        description=(
            "The sandbox environment to run in: a name GET /api/v1/code/environments lists. Omitted, the "
            "workspace's pinned environment, else the deployment's default. A held container keeps the "
            "environment it was created in."
        ),
    )
    file_ids: list[str] | None = pydantic.Field(
        default=None,
        max_length=_MAX_CODE_EXECUTION_INPUTS,
        description=(
            "Uploads from POST /api/v1/files to put in the working directory before the code runs, "
            "each under its own filename."
        ),
    )
    user: str | None = pydantic.Field(default=None, description="User ID for usage attribution")

    @pydantic.field_validator("code")
    @classmethod
    def _code_is_not_blank(cls, code: str) -> str:
        if not code.strip():
            raise ValueError("code must contain something to run")
        return code


class CodeExecutionOutputFile(pydantic.BaseModel):
    """A file the code wrote, stored in the Files API."""

    file_id: str = pydantic.Field(description="Download it from GET /api/v1/files/{file_id}/content")
    filename: str = pydantic.Field(description="The name the code gave it in the working directory")


class CodeExecuteResponse(pydantic.BaseModel):
    """A completed code execution. Code that failed is a completed execution with a nonzero ``return_code``."""

    object: Literal["code_execution.result"] = "code_execution.result"
    stdout: str
    stderr: str
    return_code: int | None = pydantic.Field(
        description="The program's exit status; null when the sandbox did not report one."
    )
    files: list[CodeExecutionOutputFile] = pydantic.Field(
        description="The files the code wrote. Empty when it wrote none, or where the Files API is off."
    )


class CodeExecutionEnvironment(pydantic.BaseModel):
    """A sandbox environment a direct code execution may name: what it contains, and what a run in it costs."""

    object: Literal["code_execution.environment"] = "code_execution.environment"
    name: str = pydantic.Field(description="What POST /api/v1/code/execute's environment field takes.")
    label: str = pydantic.Field(description="A display name; the name itself when the operator gave none.")
    description: str | None = pydantic.Field(description="What it contains and how big it is, when the operator said.")
    price_per_run: float | None = pydantic.Field(
        description="USD per run for the caller's organization: the environment's own price, else code execution's. "
        "Null when unpriced."
    )
    default: bool = pydantic.Field(description="Whether a run that names no environment runs in this one.")


class CodeExecutionEnvironmentList(pydantic.BaseModel):
    """The sandbox environments the caller's workspace may run code in."""

    object: Literal["list"] = "list"
    data: list[CodeExecutionEnvironment]


class OtariToolCallSource(pydantic.BaseModel):
    """One page a gateway-run search returned: where it is, never its text."""

    url: str
    title: str
    published_date: str | None = pydantic.Field(default=None, description="The page's date, as the provider gave it.")


class OtariToolCallError(pydantic.BaseModel):
    """What a failed gateway-run call may show: never a body, a URL or exception text."""

    provider: str | None = pydantic.Field(description="The provider that failed; null when none was reached.")
    status: int | None = pydantic.Field(description="The provider's HTTP error status, when it answered with one.")
    tag: str = pydantic.Field(description="Why the call failed, such as timeout or call_limit.")
    retryable: bool = pydantic.Field(description="Whether retrying the same call may help.")


class OtariToolCall(pydantic.BaseModel):
    """One tool call the gateway ran, as a Chat Completions response reports it in ``otari_tool_calls``.

    A stream sends the entry when the call starts and again when it ends, both with
    the same ``id``. Fields not set are left out of the wire entry.
    """

    id: str
    type: Literal["web_search", "web_fetch", "code_execution", "mcp"]
    status: Literal["in_progress", "completed", "failed"]
    query: str | None = pydantic.Field(default=None, description="A search's query.")
    url: str | None = pydantic.Field(
        default=None, description="The URL a fetch asked for, without its query string and fragment."
    )
    sources: list[OtariToolCallSource] | None = pydantic.Field(
        default=None, description="A completed search's results; empty when it found none."
    )
    code: str | None = pydantic.Field(
        default=None, description="The code the model sent a code execution; a failed one may not have run."
    )
    stdout: str | None = pydantic.Field(default=None, description="What an ended code execution printed.")
    stderr: str | None = pydantic.Field(default=None, description="What an ended code execution wrote to stderr.")
    exit_code: int | None = pydantic.Field(default=None, description="An ended code execution's exit status.")
    file_ids: list[str] | None = pydantic.Field(
        default=None, description="The /v1/files IDs of the files a code execution produced."
    )
    server: str | None = pydantic.Field(default=None, description="The MCP server that ran the call.")
    name: str | None = pydantic.Field(default=None, description="The MCP tool called.")
    output: str | None = pydantic.Field(default=None, description="The text an ended MCP call gave the model.")
    truncated: bool | None = pydantic.Field(
        default=None, description="Whether stdout, stderr or output was cut to the report's limit."
    )
    error: OtariToolCallError | None = pydantic.Field(default=None, description="Why a failed call failed, when known.")


class SandboxCredentialPublic(pydantic.BaseModel):
    """Where the key a hosted sandbox provider leases with comes from. Never the key itself."""

    source: Literal["stored", "environment"] | None = pydantic.Field(
        description=(
            "'stored' for a key set here, which wins; 'environment' for the provider's own environment "
            "variable; null when there is neither and code execution fails until one is set."
        )
    )
    last4: str | None = pydantic.Field(default=None, description="The key's last four characters, to tell keys apart.")
    env_var: str = pydantic.Field(description="The environment variable a deployment that stores no key is read from.")


class SandboxCredentialRequest(pydantic.BaseModel):
    """The key to lease hosted sandboxes with."""

    model_config = pydantic.ConfigDict(json_schema_extra={"example": {"api_key": "e2b_0123456789abcdef"}})

    api_key: str = pydantic.Field(min_length=1)
