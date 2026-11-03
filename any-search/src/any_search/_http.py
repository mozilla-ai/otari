"""The HTTP client a provider sends through."""

from typing import Any

import httpx

from any_search._errors import ProviderError

# Written at build time by setuptools-scm from the repository's Git tag (see
# pyproject.toml), as cli/'s is. Absent in a source tree nothing has built yet.
try:
    from any_search._version import __version__
except ImportError:
    __version__ = "0.0.0"

USER_AGENT = f"any-search/{__version__}"
"""Sent with every request, so a provider can tell the library's traffic apart and tie it to a release."""

# What a client sends when its host set no ``User-Agent``: the one value the library replaces.
_HTTPX_USER_AGENT = f"python-httpx/{httpx.__version__}"


class ProviderHttp:
    """Send a provider's requests through the host's client, or through one of its own.

    A host's client is used as it is: each request carries the provider's
    timeout, and the client is never closed or reconfigured, so the host keeps
    its pool and its proxy settings. Without one, a client with httpx's
    defaults and the provider's timeout is opened on the first request and
    closed by ``aclose``.

    Each request names the library in its ``User-Agent``, in place of httpx's
    default, so a provider can tell whose traffic it is. A host's client that
    sets a ``User-Agent`` of its own keeps it.
    """

    def __init__(self, provider: str, client: httpx.AsyncClient | None, timeout: float) -> None:
        self._provider = provider
        self._client = client
        self._own_client: httpx.AsyncClient | None = None
        self._timeout = timeout

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send one request.

        Transport failures become a ``ProviderError`` without the exception's
        text, which can carry the URL and with it a query or a key. It is
        raised after the handler, so the original is not even its context.
        """
        client = self._client
        if client is None:
            if self._own_client is None:
                self._own_client = httpx.AsyncClient(timeout=self._timeout)
            client = self._own_client
        try:
            # ``Headers`` encodes eagerly, so a header httpx cannot send fails inside the handler.
            headers = httpx.Headers(kwargs.pop("headers", None))
            if client.headers.get("User-Agent") == _HTTPX_USER_AGENT:
                headers["User-Agent"] = USER_AGENT
            return await client.request(method, url, headers=headers, timeout=self._timeout, **kwargs)
        except httpx.TimeoutException:
            tag = "timeout"
        except (httpx.HTTPError, httpx.InvalidURL):
            tag = "network"
        except UnicodeEncodeError:
            # A header httpx cannot send, in practice a key pasted with a character that is not ASCII.
            tag = "invalid_header"
        raise ProviderError(self._provider, None, tag)

    async def aclose(self) -> None:
        """Close the client this object opened, if it opened one."""
        if self._own_client is not None:
            await self._own_client.aclose()
            self._own_client = None
