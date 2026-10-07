"""``builtin``: the plain fetcher of ``http`` and ``https`` pages, supplied by the host.

This package declares the id and its metadata; the implementation is the
host's, registered with ``AnyFetch.register_builtin(factory)``.
``AnyFetch.create("builtin", **kwargs)`` passes its keyword arguments to that
factory and wraps what it builds, so a fetch through it runs inside the same
logging context as any other provider's.
"""

from typing import Any

from any_fetch._api import AnyFetch
from any_fetch._errors import BuiltinNotRegisteredError
from any_fetch._types import BuiltinFactory, BuiltinFetcher, FetchedPage, ProviderMetadata

_factory: BuiltinFactory | None = None


def register(factory: BuiltinFactory) -> None:
    """Make ``factory`` the implementation of ``builtin``, replacing any earlier one."""
    global _factory
    _factory = factory


def is_registered() -> bool:
    """Return whether a host has registered an implementation."""
    return _factory is not None


class BuiltinProvider(AnyFetch):
    """The library's wrapper around the fetcher the host's factory builds."""

    METADATA = ProviderMetadata(
        name="builtin",
        doc_url="https://github.com/mozilla-ai/otari/tree/main/any-fetch#the-builtin-provider",
        env_key=None,
        env_api_base=None,
        requires_api_key=False,
        requires_api_base=False,
        default_api_base=None,
        tier="production",
        max_urls_per_call=1,
        renders_javascript=False,
        # otari's fetcher extracts the readable part of a page as Markdown.
        formats=["markdown"],
        options=[],
    )

    def __init__(self, **kwargs: Any) -> None:
        if _factory is None:
            raise BuiltinNotRegisteredError()
        super().__init__()
        self._fetcher: BuiltinFetcher = _factory(**kwargs)

    async def _fetch(self, url: str, *, max_chars: int | None, options: dict[str, Any]) -> FetchedPage:
        return await self._fetcher.fetch(url, max_chars=max_chars)

    async def aclose(self) -> None:
        await self._fetcher.aclose()
        await super().aclose()
