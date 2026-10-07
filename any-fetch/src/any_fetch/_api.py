"""The provider interface and the one-call helper."""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, ClassVar, Self

import httpx

from any_fetch import _registry
from any_fetch._credentials import resolve_api_base, resolve_api_key
from any_fetch._errors import UnsupportedParameterError
from any_fetch._http import ProviderHttp
from any_fetch._logging import provider_call
from any_fetch._types import BuiltinFactory, FetchedPage, ProviderMetadata

DEFAULT_TIMEOUT = 15.0
"""Seconds per request when the caller names no timeout."""


class AnyFetch(ABC):
    """A web fetch provider. Build one with ``AnyFetch.create``.

    One parameter is shared, and every adapter maps it: ``max_chars``. Every
    other option is the provider's own, under its own name, and an option the
    provider's metadata does not list is refused.
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
    def create(cls, provider: str, **kwargs: Any) -> "AnyFetch":
        """Build a provider by name.

        Takes ``api_key`` and ``api_base`` (each from the environment variable
        the metadata names when not passed), ``timeout`` in seconds, and
        ``client``, an ``httpx.AsyncClient`` the caller owns and the provider
        sends through without closing. ``builtin`` passes every keyword
        argument to the factory the host registered instead.
        """
        return _registry.create(provider, **kwargs)

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Return the names of every provider this package serves, ``builtin`` included."""
        return _registry.get_supported_providers()

    @classmethod
    def get_provider_class(cls, provider: str) -> type["AnyFetch"]:
        """Return a provider's class without building it."""
        return _registry.get_provider_class(provider)

    @classmethod
    def get_provider_metadata(cls, provider: str) -> ProviderMetadata:
        """Return a provider's metadata."""
        return _registry.get_provider_metadata(provider)

    @classmethod
    def register_builtin(cls, factory: BuiltinFactory) -> None:
        """Register the host's implementation of ``builtin``; a second call replaces the first."""
        _registry.register_builtin(factory)

    @property
    def metadata(self) -> ProviderMetadata:
        return self.METADATA

    async def fetch(self, url: str, *, max_chars: int | None = None, **options: Any) -> FetchedPage:
        """Fetch one page.

        Raises:
            UnsupportedParameterError: An option the provider does not take.
            ProviderError: The provider failed the call or could not be reached.
                An error it signals inside a successful response is returned
                as ``FetchedPage.error`` instead.

        """
        known = {option.name for option in self.METADATA.options}
        for name in options:
            if name not in known:
                raise UnsupportedParameterError(self.METADATA.name, name)
        if max_chars is not None and max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        with provider_call(self.METADATA.name):
            return await self._fetch(url, max_chars=max_chars, options=options)

    @abstractmethod
    async def _fetch(self, url: str, *, max_chars: int | None, options: dict[str, Any]) -> FetchedPage:
        """Fetch one page with validated arguments; the adapter's half of ``fetch``."""

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


async def afetch(
    provider: str,
    url: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | None = None,
    client: httpx.AsyncClient | None = None,
    max_chars: int | None = None,
    **options: Any,
) -> FetchedPage:
    """Build a provider, fetch one page and close it.

    Only the settings passed reach ``create``, so ``builtin``'s factory sees
    nothing it was not given.
    """
    settings = {"api_key": api_key, "api_base": api_base, "timeout": timeout, "client": client}
    passed = {name: value for name, value in settings.items() if value is not None}
    async with AnyFetch.create(provider, **passed) as fetcher:
        return await fetcher.fetch(url, max_chars=max_chars, **options)
