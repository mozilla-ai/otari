"""The checked federation transport and the bounded transport layered on top of it, which
workload identity and OIDC issuer discovery share.

Nothing here reaches the network: a test of the checked transport resolves hostnames through a
monkeypatched resolver, and a test of the bounded transport alone stubs the transport it wraps.
"""

import gzip
import ipaddress
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from gateway.services.web_retrieval_network import (
    BoundedFederationTransport,
    CheckedFederationTransport,
    FederationResponseEncodedError,
    FederationResponseTooLargeError,
    PinnedAsyncHTTPTransport,
    SystemAddressResolver,
)

SAFE_HOST = "safe.issuer.example.com"
SAFE_URL = f"https://{SAFE_HOST}/discovery"
PRIVATE_REDIRECT_HOST = "internal.issuer.example.com"
PRIVATE_REDIRECT_URL = f"https://{PRIVATE_REDIRECT_HOST}/internal"
PLAIN_HTTP_HOST = "plain.issuer.example.com"
PLAIN_HTTP_URL = f"http://{PLAIN_HTTP_HOST}/discovery"
FETCH_URL = "https://federation.example.com/fetch"


@pytest.mark.asyncio
async def test_checked_federation_transport_refuses_a_redirect_to_a_private_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every hop is checked, not just the first: a redirect must not bypass the outbound check."""
    answers = {
        SAFE_HOST: [ipaddress.ip_address("93.184.216.34")],
        PRIVATE_REDIRECT_HOST: [ipaddress.ip_address("127.0.0.1")],
    }

    async def resolve(_self: object, host: str, _port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        return answers[host]

    monkeypatch.setattr(SystemAddressResolver, "resolve", resolve)

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": PRIVATE_REDIRECT_URL})

    transport = CheckedFederationTransport(allow_private_hosts=False)
    transport._transport = PinnedAsyncHTTPTransport(transport_factory=lambda: httpx.MockTransport(respond))

    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get(SAFE_URL)


@pytest.mark.asyncio
async def test_checked_federation_transport_refuses_plain_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host resolves and a connection would succeed, so only the scheme check can refuse this."""

    async def resolve(_self: object, _host: str, _port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        return [ipaddress.ip_address("93.184.216.34")]

    monkeypatch.setattr(SystemAddressResolver, "resolve", resolve)

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    transport = CheckedFederationTransport(allow_private_hosts=False)
    transport._transport = PinnedAsyncHTTPTransport(transport_factory=lambda: httpx.MockTransport(respond))

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get(PLAIN_HTTP_URL)


class _ChunkStream(httpx.AsyncByteStream):
    """Yield fixed chunks, so a byte ceiling test can cross it mid-stream."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class _StubTransport(httpx.AsyncBaseTransport):
    def __init__(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        self._respond = respond

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._respond(request)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_the_shared_transport_refuses_a_compressed_response() -> None:
    """A declared content encoding is refused outright, so the decoder it would drive never runs."""
    sent: list[httpx.Request] = []
    compressed = gzip.compress(b'{"keys": []}')

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, headers={"content-encoding": "gzip"}, content=compressed)

    transport = BoundedFederationTransport(_StubTransport(respond), max_bytes=1024)

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(FederationResponseEncodedError):
            await client.get(FETCH_URL)

    assert sent[0].headers["accept-encoding"] == "identity"


@pytest.mark.asyncio
async def test_the_shared_transport_refuses_a_response_past_the_ceiling() -> None:
    """A response that would otherwise succeed, split so the first chunk alone crosses the ceiling."""
    payload = b'{"keys": []}' + b" " * 64
    midpoint = len(payload) // 2 + 1

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_ChunkStream([payload[:midpoint], payload[midpoint:]]))

    transport = BoundedFederationTransport(_StubTransport(respond), max_bytes=midpoint - 1)

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(FederationResponseTooLargeError):
            await client.get(FETCH_URL)
