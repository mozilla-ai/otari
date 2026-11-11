"""The envelope every provider answers in, and the metadata that describes each one."""

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class FetchError(BaseModel):
    """An error the provider signaled inside a successful response."""

    tag: str
    status: int | None = None


class FetchedPage(BaseModel):
    """What one fetch returned."""

    url: str
    # Where the page was found, after redirects.
    final_url: str
    title: str = ""
    text: str
    content_type: str
    published: datetime | None = None
    # What the provider reported, in USD.
    cost: Decimal | None = None
    cost_source: Literal["reported", "none"]
    error: FetchError | None = None
    # The response body hit the fetcher's size limit.
    source_truncated: bool = False
    # The extracted text hit its own limit.
    text_truncated: bool = False
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
    max_urls_per_call: int
    renders_javascript: bool
    # Whether ``FetchedPage.final_url`` is where the page was found after redirects. When false it is
    # the URL as asked, since the provider follows redirects without saying where they led, so a
    # caller cannot check a redirect it followed.
    reports_final_url: bool
    formats: list[str]
    options: list[OptionSpec]
    # The provider's answer says what each call cost, which the adapter returns
    # as the page's cost.
    reports_cost: bool = False

    @model_validator(mode="after")
    def _one_cost_tier_at_most(self) -> "ProviderMetadata":
        tiers = [option.name for option in self.options if option.cost_tier]
        if len(tiers) > 1:
            raise ValueError(f"provider {self.name!r} has more than one cost tier option: {', '.join(tiers)}")
        return self


class BuiltinFetcher(Protocol):
    """What the host's ``builtin`` factory builds: a fetcher used for one or more fetches, then closed."""

    async def fetch(self, url: str, *, max_chars: int | None) -> FetchedPage: ...

    async def aclose(self) -> None: ...


BuiltinFactory = Callable[..., BuiltinFetcher]
"""Builds a ``BuiltinFetcher`` from the keyword arguments ``AnyFetch.create("builtin", ...)`` was given."""
