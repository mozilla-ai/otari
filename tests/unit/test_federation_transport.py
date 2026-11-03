"""The checked federation transport: the outbound-request seam workload identity and
OIDC issuer discovery share.

Hostnames are resolved through a monkeypatched resolver, so nothing here reaches the network.
"""

import ipaddress

import httpx
import pytest

from gateway.services.web_retrieval_network import (
    CheckedFederationTransport,
    PinnedAsyncHTTPTransport,
    SystemAddressResolver,
)

SAFE_HOST = "safe.issuer.example.com"
SAFE_URL = f"https://{SAFE_HOST}/discovery"
PRIVATE_REDIRECT_HOST = "internal.issuer.example.com"
PRIVATE_REDIRECT_URL = f"https://{PRIVATE_REDIRECT_HOST}/internal"
PLAIN_HTTP_HOST = "plain.issuer.example.com"
PLAIN_HTTP_URL = f"http://{PLAIN_HTTP_HOST}/discovery"


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
