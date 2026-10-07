"""The provider interface and the one-call helper."""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, ClassVar, Self, get_args

import httpx

from any_search import _registry
from any_search._credentials import resolve_api_base, resolve_api_key
from any_search._errors import UnsupportedParameterError
from any_search._http import ProviderHttp
from any_search._logging import provider_call
from any_search._types import ProviderMetadata, SearchResult, TimeRange

DEFAULT_TIMEOUT = 15.0
"""Seconds per request when the caller names no timeout."""


class AnySearch(ABC):
    """A web search provider. Build one with ``AnySearch.create``.

    Two parameters are shared, and every adapter maps them: ``max_results`` and
    ``time_range``. Every other option is the provider's own, under its own
    name, and an option the provider's metadata does not list is refused.
    """

    METADATA: ClassVar[ProviderMetadata]

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = resolve_api_key(self.METADATA, api_key)
        self._api_base = resolve_api_base(self.METADATA, api_base)
        self._http = ProviderHttp(self.METADATA.name, client, timeout)

    @classmethod
    def create(cls, provider: str, **kwargs: Any) -> "AnySearch":
        """Build a provider by name.

        Takes ``api_key`` and ``api_base`` (each from the environment variable
        the metadata names when not passed), ``timeout`` in seconds, and
        ``client``, an ``httpx.AsyncClient`` the caller owns and the provider
        sends through without closing.
        """
        return _registry.create(provider, **kwargs)

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Return the names of every provider this package serves."""
        return _registry.get_supported_providers()

    @classmethod
    def get_provider_class(cls, provider: str) -> type["AnySearch"]:
        """Return a provider's class without building it."""
        return _registry.get_provider_class(provider)

    @classmethod
    def get_provider_metadata(cls, provider: str) -> ProviderMetadata:
        """Return a provider's metadata."""
        return _registry.get_provider_metadata(provider)

    @property
    def metadata(self) -> ProviderMetadata:
        return self.METADATA

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        time_range: TimeRange | None = None,
        **options: Any,
    ) -> SearchResult:
        """Run one search.

        Raises:
            UnsupportedParameterError: An option the provider does not take.
            ProviderError: The provider failed the call or could not be reached.
                An error it signals inside a successful response is returned
                as ``SearchResult.error`` instead.

        """
        known = {option.name for option in self.METADATA.options}
        for name in options:
            if name not in known:
                raise UnsupportedParameterError(self.METADATA.name, name)
        if max_results is not None and max_results < 1:
            raise ValueError("max_results must be at least 1")
        if time_range is not None and time_range not in get_args(TimeRange):
            raise ValueError(f"time_range must be one of {', '.join(get_args(TimeRange))}")
        with provider_call(self.METADATA.name):
            return await self._search(query, max_results=max_results, time_range=time_range, options=options)

    @abstractmethod
    async def _search(
        self,
        query: str,
        *,
        max_results: int | None,
        time_range: TimeRange | None,
        options: dict[str, Any],
    ) -> SearchResult:
        """Run one search with validated arguments; the adapter's half of ``search``."""

    async def aclose(self) -> None:
        """Close the HTTP client the provider opened for itself, if any."""
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        # The key and the base URL stay out: either can be a secret.
        return f"<{type(self).__name__} provider={self.METADATA.name!r}>"


async def asearch(
    provider: str,
    query: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | None = None,
    client: httpx.AsyncClient | None = None,
    max_results: int | None = None,
    time_range: TimeRange | None = None,
    **options: Any,
) -> SearchResult:
    """Build a provider, run one search and close it. Only the settings passed reach ``create``."""
    settings = {"api_key": api_key, "api_base": api_base, "timeout": timeout, "client": client}
    passed = {name: value for name, value in settings.items() if value is not None}
    async with AnySearch.create(provider, **passed) as engine:
        return await engine.search(query, max_results=max_results, time_range=time_range, **options)
