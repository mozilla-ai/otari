"""The HTTP client a provider sends through."""

from typing import Any

import httpx

from any_fetch._errors import ProviderError


class ProviderHttp:
    """Send a provider's requests through the host's client, or through one of its own.

    A host's client is used as it is: each request carries the provider's
    timeout, and the client is never closed or reconfigured, so the host keeps
    its pool and its proxy settings. Without one, a client with httpx's
    defaults and the provider's timeout is opened on the first request and
    closed by ``aclose``.
    """

    def __init__(self, provider: str, client: httpx.AsyncClient | None, timeout: float) -> None:
        self._provider = provider
        self._client = client
        self._own_client: httpx.AsyncClient | None = None
        self._timeout = timeout

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send one request.

        Transport failures become a ``ProviderError`` without the exception's
        text, which can carry the URL and with it a query or a key.
        """
        client = self._client
        if client is None:
            if self._own_client is None:
                self._own_client = httpx.AsyncClient(timeout=self._timeout)
            client = self._own_client
        try:
            return await client.request(method, url, timeout=self._timeout, **kwargs)
        except httpx.TimeoutException:
            raise ProviderError(self._provider, None, "timeout") from None
        except (httpx.HTTPError, httpx.InvalidURL):
            raise ProviderError(self._provider, None, "network") from None

    async def aclose(self) -> None:
        """Close the client this object opened, if it opened one."""
        if self._own_client is not None:
            await self._own_client.aclose()
            self._own_client = None
