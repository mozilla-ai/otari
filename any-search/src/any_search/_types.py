"""The envelope every provider answers in, and the metadata that describes each one."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TimeRange = Literal["day", "week", "month", "year"]
"""The shared recency filter every adapter maps onto its provider's own."""

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
