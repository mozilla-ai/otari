"""The only way the gateway dials an endpoint a workspace or a user owns.

The base URL is a tenant's, so it has to reach a public address and nothing
else: not the metadata service, not a sidecar, not the database. Checking the
URL once is not enough, because the name can be re-pointed after it was
checked, and the provider SDKs follow redirects by default.

So every request goes through :class:`OwnedEndpointTransport`, which resolves
the host itself, refuses the request unless every answer is publicly routable,
and dials one of the admitted addresses rather than letting the connection
resolve the name again. Redirects are not followed. The same check runs when an
endpoint is saved, so a bad URL is refused where it is written.
"""

import asyncio
import weakref

import httpx

from gateway.services.web_retrieval_network import (
    PINNED_TARGET_EXTENSION,
    PinnedAsyncHTTPTransport,
    PinnedTransportError,
    RetrievalTargetError,
    TransportFactory,
    validate_retrieval_target,
)

# The provider SDKs' own default: long generations need the read budget. A
# caller waiting on a full pool is refused within seconds rather than held for
# the whole read budget behind other callers' streams.
_TIMEOUT = httpx.Timeout(600.0, connect=5.0, pool=10.0)
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
        client = httpx.AsyncClient(transport=OwnedEndpointTransport(), follow_redirects=False, timeout=_TIMEOUT)
        _clients[loop] = client
    return client
