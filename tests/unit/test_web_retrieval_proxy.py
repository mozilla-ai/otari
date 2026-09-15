"""Trusted environment proxies must never weaken the direct retrieval path."""

from __future__ import annotations

import asyncio
import ipaddress
import os
from typing import Any

import httpx
import pytest

from gateway.services.web_fetch_service import WebFetchService
from gateway.services.web_retrieval_backend import WebRetrievalBackend
from gateway.services.web_retrieval_network import (
    PinnedAsyncHTTPTransport,
    PinnedTransportError,
    RetrievalAddressError,
    RetrievalDomainPolicyError,
    TrustedProxyAsyncHTTPTransport,
)
from gateway.services.web_retrieval_policy import DomainPolicy, IPAddress, canonicalize_domain_rules

_PUBLIC_IP = "93.184.216.34"


class PublicResolver:
    async def resolve(self, host: str, port: int) -> tuple[IPAddress, ...]:
        return (ipaddress.ip_address(_PUBLIC_IP),)


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, proxy: httpx.Proxy | None) -> None:
        self.proxy = proxy
        self.requests: list[httpx.Request] = []
        self.closed = False
        self.redirect: str | None = None
        self.fail = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            raise httpx.ConnectError("proxy unavailable")
        if self.redirect:
            location, self.redirect = self.redirect, None
            return httpx.Response(302, headers={"Location": location}, stream=httpx.ByteStream(b""))
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, stream=httpx.ByteStream(b"page"))

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def transports(monkeypatch: pytest.MonkeyPatch) -> list[RecordingTransport]:
    for name in tuple(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    built: list[RecordingTransport] = []

    def factory(**kwargs: Any) -> RecordingTransport:
        transport = RecordingTransport(kwargs.get("proxy"))
        built.append(transport)
        return transport

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", factory)
    return built


@pytest.mark.asyncio
async def test_default_transport_ignores_environment_proxies(
    monkeypatch: pytest.MonkeyPatch, transports: list[RecordingTransport]
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    async with httpx.AsyncClient(transport=PinnedAsyncHTTPTransport(), trust_env=False) as client:
        result = await WebFetchService(client, resolver=PublicResolver()).fetch("https://example.com/")
    assert result.text == "page"
    assert len(transports) == 1
    assert transports[0].proxy is None
    assert transports[0].requests[0].url.host == _PUBLIC_IP
    assert transports[0].closed


@pytest.mark.parametrize(
    ("environment", "url", "expected_proxy"),
    [
        ({"HTTPS_PROXY": "http://proxy.example:8080"}, "https://example.com/", "proxy.example"),
        ({"HTTP_PROXY": "http://proxy.example:8080"}, "http://example.com/", "proxy.example"),
        ({"ALL_PROXY": "http://fallback.example:8080"}, "https://example.com/", "fallback.example"),
        ({"https_proxy": "http://lower.example:8080"}, "https://example.com/", "lower.example"),
        (
            {"HTTPS_PROXY": "http://upper.example:8080", "https_proxy": "http://lower.example:8080"},
            "https://example.com/",
            "lower.example",
        ),
        (
            {"HTTPS_PROXY": "http://specific.example:8080", "ALL_PROXY": "http://fallback.example:8080"},
            "https://example.com/",
            "specific.example",
        ),
        ({"HTTP_PROXY": "http://proxy.example:8080"}, "https://example.com/", None),
        ({}, "https://example.com/", None),
        (
            {"HTTPS_PROXY": "http://proxy.example:8080", "NO_PROXY": "example.com"},
            "https://notexample.com/",
            "proxy.example",
        ),
        (
            {"HTTPS_PROXY": "http://proxy.example:8080", "NO_PROXY": "example.com:443"},
            "https://example.com/",
            None,
        ),
        (
            {"HTTPS_PROXY": "http://proxy.example:8080", "NO_PROXY": "example.com:8443"},
            "https://example.com/",
            "proxy.example",
        ),
        (
            {"HTTPS_PROXY": "http://proxy.example:8080", "NO_PROXY": "example.com"},
            "https://example.com/",
            None,
        ),
        (
            {"HTTPS_PROXY": "http://proxy.example:8080", "no_proxy": ".example.com"},
            "https://sub.example.com/",
            None,
        ),
        (
            {"HTTPS_PROXY": "http://proxy.example:8080", "NO_PROXY": "*"},
            "https://example.com/",
            None,
        ),
        (
            {"HTTPS_PROXY": "http://proxy.example:8080", "NO_PROXY": "example.com:8443"},
            "https://example.com:8443/",
            None,
        ),
    ],
)
@pytest.mark.asyncio
async def test_environment_proxy_routing(
    monkeypatch: pytest.MonkeyPatch,
    transports: list[RecordingTransport],
    environment: dict[str, str],
    url: str,
    expected_proxy: str | None,
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    async with httpx.AsyncClient(transport=TrustedProxyAsyncHTTPTransport(), trust_env=False) as client:
        result = await WebFetchService(client, resolver=PublicResolver()).fetch(url)
    assert result.text == "page"
    used = [transport for transport in transports if transport.requests]
    assert len(used) == 1
    transport = used[0]
    request = transport.requests[0]
    if expected_proxy is None:
        assert transport.proxy is None
        assert request.url.host == _PUBLIC_IP
        assert request.extensions["sni_hostname"] == httpx.URL(url).host
    else:
        assert transport.proxy is not None
        assert transport.proxy.url.host == expected_proxy
        assert request.url == httpx.URL(url)
    assert request.headers["Host"] == httpx.Request("GET", url).headers["Host"]
    assert request.headers["Accept-Encoding"] == "gzip, deflate"
    assert all(transport.closed for transport in transports)


@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.asyncio
async def test_backend_selects_proxy_transport_only_with_opt_in(
    monkeypatch: pytest.MonkeyPatch, transports: list[RecordingTransport], trusted: bool
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    async with WebRetrievalBackend(base_url="http://searxng:8080", trust_env_proxy=trusted) as backend:
        assert backend._retrieval_service is not None
        backend._retrieval_service._resolver = PublicResolver()
        result = await backend._retrieval_service.fetch("https://example.com/")
    assert result.text == "page"
    assert (transports[0].proxy is not None) is trusted
    assert all(transport.closed for transport in transports)


@pytest.mark.asyncio
async def test_backend_closes_search_client_when_proxy_setup_fails(
    monkeypatch: pytest.MonkeyPatch, transports: list[RecordingTransport]
) -> None:
    def fail() -> TrustedProxyAsyncHTTPTransport:
        raise ValueError("invalid proxy configuration")

    monkeypatch.setattr("gateway.services.web_retrieval_backend.TrustedProxyAsyncHTTPTransport", fail)
    backend = WebRetrievalBackend(base_url="http://searxng:8080", trust_env_proxy=True)
    with pytest.raises(ValueError, match="invalid proxy"):
        await backend.__aenter__()
    assert backend._client is not None
    assert backend._client.is_closed


@pytest.mark.asyncio
async def test_proxy_auth_is_not_sent_to_destination(
    monkeypatch: pytest.MonkeyPatch, transports: list[RecordingTransport]
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://user:password@proxy.example:8080")
    async with httpx.AsyncClient(transport=TrustedProxyAsyncHTTPTransport(), trust_env=False) as client:
        await WebFetchService(client, resolver=PublicResolver()).fetch("https://example.com/")
    assert transports[0].proxy is not None
    assert transports[0].proxy.auth == ("user", "password")
    assert "Proxy-Authorization" not in transports[0].requests[0].headers


@pytest.mark.asyncio
async def test_proxy_failure_never_falls_back_to_direct(
    monkeypatch: pytest.MonkeyPatch, transports: list[RecordingTransport]
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    transport = TrustedProxyAsyncHTTPTransport()
    transports[0].fail = True
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        with pytest.raises(httpx.ConnectError):
            await WebFetchService(client, resolver=PublicResolver()).fetch("https://example.com/")
    assert len(transports) == 1
    assert len(transports[0].requests) == 1
    assert transports[0].closed


@pytest.mark.asyncio
async def test_redirect_to_no_proxy_uses_pinning(
    monkeypatch: pytest.MonkeyPatch, transports: list[RecordingTransport]
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", "direct.example")
    transport = TrustedProxyAsyncHTTPTransport()
    transports[0].redirect = "https://direct.example/final"
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        result = await WebFetchService(client, resolver=PublicResolver()).fetch("https://example.com/")
    assert result.redirect_count == 1
    assert len(transports) == 2
    assert transports[0].requests[0].url.host == "example.com"
    assert transports[1].proxy is None
    assert transports[1].requests[0].url.host == _PUBLIC_IP
    assert transports[1].requests[0].headers["Host"] == "direct.example"


@pytest.mark.asyncio
async def test_proxy_routing_snapshot_ignores_later_environment_changes(
    monkeypatch: pytest.MonkeyPatch, transports: list[RecordingTransport]
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    transport = TrustedProxyAsyncHTTPTransport()
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("HTTPS_PROXY", "http://changed.example:8080")
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        await WebFetchService(client, resolver=PublicResolver()).fetch("https://example.com/")
    assert len(transports) == 1
    assert transports[0].proxy is not None
    assert transports[0].proxy.url.host == "proxy.example"
    assert len(transports[0].requests) == 1


@pytest.mark.parametrize("location", ["https://127.0.0.1/", "https://blocked.example/"])
@pytest.mark.asyncio
async def test_proxy_redirects_still_enforce_address_and_domain_policy(
    monkeypatch: pytest.MonkeyPatch, transports: list[RecordingTransport], location: str
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    transport = TrustedProxyAsyncHTTPTransport()
    transports[0].redirect = location
    policy = DomainPolicy(blocked=canonicalize_domain_rules(("blocked.example",)))
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        with pytest.raises((RetrievalAddressError, RetrievalDomainPolicyError)):
            await WebFetchService(client, resolver=PublicResolver()).fetch("https://example.com/", policy=policy)
    assert len(transports[0].requests) == 1
    assert transports[0].closed


@pytest.mark.parametrize(
    "value", ["not a proxy", "socks5://proxy.example:1080", "http://user:secret@proxy.example:invalid/", "http://"]
)
def test_invalid_proxy_configuration_fails_without_echoing_credentials(
    monkeypatch: pytest.MonkeyPatch, transports: list[RecordingTransport], value: str
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", value)
    with pytest.raises(ValueError, match="web retrieval") as exc:
        TrustedProxyAsyncHTTPTransport()
    assert "secret" not in str(exc.value)
    assert not transports


@pytest.mark.asyncio
async def test_proxy_transport_requires_validated_target(
    monkeypatch: pytest.MonkeyPatch, transports: list[RecordingTransport]
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    async with httpx.AsyncClient(transport=TrustedProxyAsyncHTTPTransport(), trust_env=False) as client:
        with pytest.raises(PinnedTransportError, match="missing"):
            await client.get("https://example.com/")
    assert not transports[0].requests


@pytest.mark.asyncio
async def test_real_http_transport_sends_request_to_configured_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    received: list[bytes] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            received.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 4\r\n"
                b"Connection: close\r\n\r\npage"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(serve, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{port}")
        async with httpx.AsyncClient(transport=TrustedProxyAsyncHTTPTransport(), trust_env=False) as client:
            result = await WebFetchService(client, resolver=PublicResolver()).fetch("http://example.com/page")
    assert result.text == "page"
    assert len(received) == 1
    assert received[0].startswith(b"GET http://example.com/page HTTP/1.1\r\n")
    assert b"Host: example.com\r\n" in received[0]
