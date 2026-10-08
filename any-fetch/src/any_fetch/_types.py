"""The envelope every provider answers in, and the metadata that describes each one."""

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

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
    formats: list[str]
    options: list[OptionSpec]


class BuiltinFetcher(Protocol):
    """What the host's ``builtin`` factory builds: a fetcher used for one or more fetches, then closed."""

    async def fetch(self, url: str, *, max_chars: int | None) -> FetchedPage: ...

    async def aclose(self) -> None: ...


BuiltinFactory = Callable[..., BuiltinFetcher]
"""Builds a ``BuiltinFetcher`` from the keyword arguments ``AnyFetch.create("builtin", ...)`` was given."""
