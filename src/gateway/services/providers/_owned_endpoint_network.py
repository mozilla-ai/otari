"""The only way the gateway dials an endpoint a workspace or a user owns.

The base URL is a tenant's, so it has to reach a public address and nothing
else: not the metadata service, not a sidecar, not the database. Checking the
URL once is not enough, because the name can be re-pointed after it was
checked, and the provider SDKs follow redirects by default.

So every request goes through :class:`OwnedEndpointTransport`, which resolves
the host itself, refuses the request unless every answer is publicly routable,
and dials one of the admitted addresses rather than letting the connection
resolve the name again. Redirects are not followed, and no cookie is kept, since
one client serves every tenant's endpoints. The same check runs when an
endpoint is saved, so a bad URL is refused where it is written.

Anthropic's SDK accepts only an httpx2 client, so its client runs the same
transport behind :class:`_Httpx2Bridge` rather than a second copy of it.
"""

import asyncio
import weakref
from collections.abc import AsyncIterator

import httpx
import httpx2
from any_llm import LLMProvider

from gateway.services.web_retrieval_network import (
    PINNED_TARGET_EXTENSION,
    NoCookieJar,
    PinnedAsyncHTTPTransport,
    PinnedTransportError,
    RetrievalAddressError,
    RetrievalTargetError,
    TransportFactory,
    validate_retrieval_target,
)

# The provider SDKs' own default: long generations need the read budget. A
# caller waiting on a full pool is refused within seconds rather than held for
# the whole read budget behind other callers' streams.
_TIMEOUT = httpx.Timeout(600.0, connect=5.0, pool=10.0)
_HTTPX2_TIMEOUT = httpx2.Timeout(600.0, connect=5.0, pool=10.0)
# Concurrent requests one worker holds open to one endpoint address. A stream
# holds its connection for the whole generation.
_MAX_CONNECTIONS_PER_ADDRESS = 100
# Pools are per endpoint origin and address, so the bound is how many distinct
# endpoints one worker can hold open at once.
_MAX_POOLS = 500


class OwnedEndpointAddressError(ValueError):
    """The endpoint's URL is malformed or reaches an address that is not public."""


async def check_owned_endpoint_api_base(api_base: str) -> None:
    """Refuse a base URL unless every address its host resolves to is publicly routable."""
    try:
        await validate_retrieval_target(api_base)
    except RetrievalAddressError as exc:
        raise OwnedEndpointAddressError(
            f"endpoint api_base refused: {exc}. A model server the deployment runs on a private "
            "address is added as a hosted provider instead."
        ) from None
    except RetrievalTargetError as exc:
        raise OwnedEndpointAddressError(f"endpoint api_base refused: {exc}") from None


class OwnedEndpointTransport(PinnedAsyncHTTPTransport):
    """Validate and pin each request's destination, whatever the method."""

    allowed_methods = frozenset({"GET", "POST", "DELETE"})

    def __init__(self, *, transport_factory: TransportFactory | None = None) -> None:
        super().__init__(
            transport_factory=transport_factory,
            max_connections=_MAX_CONNECTIONS_PER_ADDRESS,
            max_keepalive_connections=20,
            max_pools=_MAX_POOLS,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            target = await validate_retrieval_target(str(request.url))
        except RetrievalTargetError as exc:
            raise PinnedTransportError(f"endpoint refused: {exc}") from None
        request.extensions = {**request.extensions, PINNED_TARGET_EXTENSION: target}
        return await super().handle_async_request(request)


# One client per event loop: an httpx client's connections belong to the loop
# that opened them.
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = weakref.WeakKeyDictionary()


def owned_endpoint_http_client() -> httpx.AsyncClient:
    """The HTTP client an owned endpoint's provider SDK must use. Call from a running loop."""
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            transport=OwnedEndpointTransport(), follow_redirects=False, timeout=_TIMEOUT, cookies=NoCookieJar()
        )
        _clients[loop] = client
    return client


def _as_httpx2_error(exc: httpx.TransportError, request: httpx2.Request) -> httpx2.TransportError:
    """The httpx2 error of the same kind, so the SDK still tells a timeout from a refused connection."""
    for cls in type(exc).__mro__:
        counterpart = getattr(httpx2, cls.__name__, None)
        if isinstance(counterpart, type) and issubclass(counterpart, httpx2.TransportError):
            return counterpart(str(exc), request=request)
    return httpx2.TransportError(str(exc), request=request)


class _HttpxRequestStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx2.AsyncByteStream) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class _Httpx2ResponseStream(httpx2.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, request: httpx2.Request) -> None:
        self._stream = stream
        self._request = request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                yield chunk
        except httpx.TransportError as exc:
            raise _as_httpx2_error(exc, self._request) from exc

    async def aclose(self) -> None:
        await self._stream.aclose()


class _Httpx2Bridge(httpx2.AsyncBaseTransport):
    """Serve an httpx2 client's requests through an httpx transport."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if not isinstance(request.stream, httpx2.AsyncByteStream):
            raise httpx2.TransportError("request stream must be asynchronous", request=request)
        bridged = httpx.Request(
            method=request.method,
            url=str(request.url),
            headers=request.headers.raw,
            stream=_HttpxRequestStream(request.stream),
            extensions=request.extensions,
        )
        try:
            response = await self._transport.handle_async_request(bridged)
        except httpx.TransportError as exc:
            raise _as_httpx2_error(exc, request) from exc
        if not isinstance(response.stream, httpx.AsyncByteStream):
            raise httpx2.TransportError("response stream must be asynchronous", request=request)
        return httpx2.Response(
            status_code=response.status_code,
            headers=response.headers.raw,
            stream=_Httpx2ResponseStream(response.stream, request),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()


_httpx2_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx2.AsyncClient]" = (
    weakref.WeakKeyDictionary()
)


def _owned_endpoint_httpx2_client() -> httpx2.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _httpx2_clients.get(loop)
    if client is None or client.is_closed:
        client = httpx2.AsyncClient(
            transport=_Httpx2Bridge(OwnedEndpointTransport()),
            follow_redirects=False,
            timeout=_HTTPX2_TIMEOUT,
            cookies=NoCookieJar(),
        )
        _httpx2_clients[loop] = client
    return client


def owned_endpoint_sdk_client(provider: LLMProvider) -> httpx.AsyncClient | httpx2.AsyncClient:
    """The client ``provider``'s SDK dials an owned endpoint through. Call from a running loop."""
    if provider is LLMProvider.ANTHROPIC:
        return _owned_endpoint_httpx2_client()
    return owned_endpoint_http_client()
