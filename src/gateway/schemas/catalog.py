"""Request and response shapes for the caller's grouped model catalog."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from gateway.models.pricing import PriceSource


class CatalogCapabilities(BaseModel):
    """What a model can do, as models.dev reports it. Any offering's yes is the model's."""

    reasoning: bool = False
    tool_call: bool = False
    structured_output: bool = False
    attachment: bool = False
    temperature: bool = False


class CatalogModelSummary(BaseModel):
    """One model, as the list shows it."""

    id: str = Field(
        description="The catalog id, vendor-qualified where the vendor is known: `z-ai/glm-5.3`, else the bare slug."
    )
    selector: str | None = Field(
        default=None,
        description=(
            "The id as a selector: send it as `model` and the model's cheapest offering the caller can reach "
            "answers, the vendor's own provider first where it serves the model. "
            "Null until the gateway has indexed the catalog."
        ),
    )
    resolves_to: str | None = Field(default=None, description="The offering `selector` resolves to.")
    name: str
    vendor: str | None
    description: str | None = Field(default=None, description="models.dev's, from the offering that named the model.")
    family: str | None = None
    capabilities: CatalogCapabilities
    input_modalities: list[str]
    output_modalities: list[str]
    context_window: int | None = Field(default=None, description="The largest any offering serves.")
    max_output_tokens: int | None = Field(default=None, description="The largest any offering serves.")
    release_date: str | None = None
    knowledge_cutoff: str | None = None
    open_weights: bool = False
    deprecated: bool = Field(default=False, description="True only when every offering with metadata says so.")
    offering_count: int
    provider_count: int
    providers: list[str] = Field(description="The provider instances offering it, sorted.")
    selectors: list[str] = Field(description="Every offering's selector, so the list can be searched by one.")
    price_sources: list[PriceSource] = Field(
        description="Which price lists the priced offerings came from, distinct and sorted."
    )
    unpriced_count: int = Field(description="How many offerings carry no price for this caller.")
    discovered: bool = Field(description="Whether any offering was discovered from its provider.")
    min_input_price_per_million: float | None = Field(
        default=None,
        description="The cheapest offering's, at the comparison context where one was asked for.",
    )
    min_output_price_per_million: float | None = None


class CatalogCapability(StrEnum):
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    STRUCTURED_OUTPUT = "structured_output"
    ATTACHMENT = "attachment"
    OPEN_WEIGHTS = "open_weights"


class CatalogQuery(BaseModel):
    """Filters narrow the caller's catalog before sorting, counting, and paging."""

    at_context: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Compare prices for a request of this many input tokens: each model's minimum is taken "
            "from the pricing tier that request would settle at. Omitted, the base rates compare."
        ),
    )
    search: str | None = Field(
        default=None,
        max_length=200,
        description="Case-insensitive text in a model's name, vendor, id, selectors, or provider instances.",
    )
    skip: int = Field(default=0, ge=0, description="Number of matching models to skip.")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of models to return.")
    provider: list[str] = Field(default_factory=list, max_length=100, description="Match any named provider instance.")
    vendor: list[str] = Field(
        default_factory=list, max_length=100, description="Match any vendor; an empty value names unknown vendors."
    )
    input_modality: list[str] = Field(default_factory=list, max_length=100, description="Require every input modality.")
    output_modality: list[str] = Field(
        default_factory=list, max_length=100, description="Require every output modality."
    )
    capability: list[CatalogCapability] = Field(
        default_factory=list, max_length=100, description="Require every capability."
    )
    min_context: int = Field(default=0, ge=0, description="Minimum context window; unknown windows do not match.")
    max_input: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Maximum cheapest input price per million tokens; unpriced models do not match.",
    )
    pricing: Literal["all", "custom", "default", "priced", "unpriced"] = "all"
    source: Literal["all", "discovered", "custom"] = "all"
    released_within_days: int = Field(
        default=0,
        ge=0,
        le=36500,
        description="Release window ending today (UTC); zero disables it. Unknown and future releases do not match.",
    )
    sort: Literal["name", "released", "input", "output", "context", "providers"] = "name"
    direction: Literal["asc", "desc"] = "asc"
    include_facets: bool = Field(
        default=False,
        description="Include the filter choices drawn from the whole authorized catalog.",
    )


class CatalogVendorFacet(BaseModel):
    value: str = Field(description="The vendor's name; empty for models whose vendor is unknown.")
    vendor_slug: str | None = None


class CatalogFacets(BaseModel):
    """The filter rail's choices, from the caller's whole authorized catalog rather than one page."""

    total_count: int = Field(ge=0, description="Authorized models before any of the request's filters.")
    providers: list[str] = Field(description="Every provider instance offering an authorized model, sorted.")
    vendors: list[CatalogVendorFacet]


class CatalogPage(BaseModel):
    count: int
    models: list[CatalogModelSummary]
    facets: CatalogFacets | None = None
