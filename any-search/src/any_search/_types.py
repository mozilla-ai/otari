"""The envelope every provider answers in, and the metadata that describes each one."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TimeRange = Literal["day", "week", "month", "year"]
"""The shared recency filter every adapter maps onto its provider's own."""

TIME_RANGE_SPANS: dict[TimeRange, timedelta] = {
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
    "year": timedelta(days=365),
}
"""How far back each range reaches, for a provider that takes the date it starts on."""

OptionType = Literal["string", "integer", "number", "boolean", "array", "object"]


class OptionSpec(BaseModel):
    """One native option a provider accepts, under the provider's own name."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: OptionType
    enum: list[str] | None = None
    default: Any = None
    description: str = ""
    # Only the deployment's operator may set it, never a tenant or a request.
    # The library accepts it from anyone; enforcing that is the host's job.
    operator_only: bool = False
    # The option's values are the provider's cost tiers: a call made with one
    # value is billed at that tier's own price per call, as the provider's
    # pricing lists. It records no amount; the adapter's pricing note has those.
    # ``enum`` lists the tiers and ``default`` is the one the provider uses when
    # the option is not sent.
    cost_tier: bool = False

    @model_validator(mode="after")
    def _a_cost_tier_lists_its_tiers(self) -> "OptionSpec":
        if self.cost_tier and not (self.enum and self.default in self.enum):
            raise ValueError(f"cost tier option {self.name!r} needs an enum and a default from it")
        return self


class SearchError(BaseModel):
    """An error the provider signaled inside a successful response."""

    tag: str
    status: int | None = None


class SearchHit(BaseModel):
    """One result. A provider item without a URL never becomes a hit."""

    url: str
    title: str = ""
    snippet: str = ""
    # The page text, when the provider returned it.
    text: str | None = None
    published: datetime | None = None
    # The provider's own item, untouched. Never logged, and left out of repr.
    raw: dict[str, Any] = Field(repr=False)


class SearchResult(BaseModel):
    """What one search returned."""

    provider: str
    hits: list[SearchHit]
    # What the provider reported, in USD.
    cost: Decimal | None = None
    cost_source: Literal["reported", "none"]
    error: SearchError | None = None
    # The provider's own response, untouched. Never logged, and left out of repr.
    raw: dict[str, Any] = Field(repr=False)


class ProviderMetadata(BaseModel):
    """What a host needs to know about a provider before calling it."""

    model_config = ConfigDict(frozen=True)

    name: str
    doc_url: str
    env_key: str | None
    env_api_base: str | None
    requires_api_key: bool
    requires_api_base: bool
    default_api_base: str | None
    tier: Literal["production", "test"]
    max_results: int
    # The query, or the key, travels in the request URL.
    query_in_url: bool
    key_in_url: bool
    options: list[OptionSpec]
    # The provider's answer says what each call cost, which the adapter returns
    # as the result's cost.
    reports_cost: bool = False

    @model_validator(mode="after")
    def _one_cost_tier_at_most(self) -> "ProviderMetadata":
        tiers = [option.name for option in self.options if option.cost_tier]
        if len(tiers) > 1:
            raise ValueError(f"provider {self.name!r} has more than one cost tier option: {', '.join(tiers)}")
        return self
