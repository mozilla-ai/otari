"""Discovering a customer's own OIDC issuer configuration.

The transport is stubbed at the module level, so nothing here reaches the network.
"""

import asyncio
import gzip
import json
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta
from typing import Any

import httpx
import pytest
from apron_auth.models import ServerMetadata

from gateway.exceptions.identity_exceptions import OIDCDiscoveryError
from gateway.services.identity._discovery import discover_connection_metadata

ISSUER = "https://sso.example.test"
_VALID_DOCUMENT: dict[str, Any] = {
    "issuer": ISSUER,
    "authorization_endpoint": f"{ISSUER}/authorize",
    "token_endpoint": f"{ISSUER}/token",
    "jwks_uri": f"{ISSUER}/jwks",
    "code_challenge_methods_supported": ["S256"],
}


class _StubTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler: Callable[[httpx.Request], Awaitable[httpx.Response]]) -> None:
        self._handler = handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._handler(request)

    async def aclose(self) -> None:
        return None


class _ChunkStream(httpx.AsyncByteStream):
    """Yield fixed chunks, so a byte ceiling test can cross it mid-stream."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class _ForeverStream(httpx.AsyncByteStream):
    """Yield a byte, then wait, forever: a document with no end and no silence."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            yield b"x"
            await asyncio.sleep(0.01)

    async def aclose(self) -> None:
        return None


def _stub_discovery_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], Awaitable[httpx.Response]]
) -> None:
    monkeypatch.setattr(
        "gateway.services.identity._discovery.CheckedFederationTransport",
        lambda *, allow_private_hosts: _StubTransport(handler),
    )


def _json_response(document: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "application/json"}, content=json.dumps(document).encode())


@pytest.mark.asyncio
async def test_discovery_refuses_a_document_naming_a_different_issuer(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response({**_VALID_DOCUMENT, "issuer": "https://a-different-issuer.example.test"})

    _stub_discovery_transport(monkeypatch, handler)

    with pytest.raises(OIDCDiscoveryError):
        await discover_connection_metadata(ISSUER, allow_private_hosts=False)


@pytest.mark.asyncio
async def test_discovery_refuses_a_plain_http_issuer() -> None:
    with pytest.raises(OIDCDiscoveryError):
        await discover_connection_metadata("http://sso.example.test", allow_private_hosts=False)


@pytest.mark.asyncio
async def test_discovery_refuses_a_provider_advertising_no_s256(monkeypatch: pytest.MonkeyPatch) -> None:
    document = {**_VALID_DOCUMENT, "code_challenge_methods_supported": ["plain"]}

    async def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(document)

    _stub_discovery_transport(monkeypatch, handler)

    with pytest.raises(OIDCDiscoveryError):
        await discover_connection_metadata(ISSUER, allow_private_hosts=False)


@pytest.mark.asyncio
async def test_discovery_refuses_a_provider_naming_no_code_challenge_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """RFC 8414 allows the field to be absent; this deployment refuses that anyway."""
    document = {key: value for key, value in _VALID_DOCUMENT.items() if key != "code_challenge_methods_supported"}

    async def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(document)

    _stub_discovery_transport(monkeypatch, handler)

    with pytest.raises(OIDCDiscoveryError):
        await discover_connection_metadata(ISSUER, allow_private_hosts=False)


@pytest.mark.asyncio
async def test_discovery_refuses_a_provider_advertising_an_empty_method_list(monkeypatch: pytest.MonkeyPatch) -> None:
    document = {**_VALID_DOCUMENT, "code_challenge_methods_supported": []}

    async def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(document)

    _stub_discovery_transport(monkeypatch, handler)

    with pytest.raises(OIDCDiscoveryError):
        await discover_connection_metadata(ISSUER, allow_private_hosts=False)


@pytest.mark.asyncio
async def test_discovery_refuses_a_compressed_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared content encoding is refused outright, so the decoder it would drive never runs."""
    sent: list[httpx.Request] = []
    compressed = gzip.compress(json.dumps(_VALID_DOCUMENT).encode())

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "content-encoding": "gzip"},
            content=compressed,
        )

    _stub_discovery_transport(monkeypatch, handler)

    with pytest.raises(OIDCDiscoveryError):
        await discover_connection_metadata(ISSUER, allow_private_hosts=False)

    assert sent[0].headers["accept-encoding"] == "identity"


@pytest.mark.asyncio
async def test_discovery_accepts_a_well_formed_document(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(_VALID_DOCUMENT)

    _stub_discovery_transport(monkeypatch, handler)

    result = await discover_connection_metadata(ISSUER, allow_private_hosts=False)

    assert result.resolved_issuer == ISSUER
    rebuilt = ServerMetadata(**result.document)
    assert rebuilt.issuer == ISSUER
    assert rebuilt.authorize_url == _VALID_DOCUMENT["authorization_endpoint"]
    assert rebuilt.jwks_url == _VALID_DOCUMENT["jwks_uri"]
    assert result.discovered_at.tzinfo is not None
    assert result.discovered_at.utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_discovery_refuses_a_document_over_the_byte_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A document that would otherwise discover successfully, split so the first chunk alone crosses the ceiling."""
    payload = json.dumps(_VALID_DOCUMENT).encode()
    midpoint = len(payload) // 2 + 1
    monkeypatch.setattr("gateway.services.identity._discovery.DISCOVERY_MAX_BYTES", midpoint - 1)

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_ChunkStream([payload[:midpoint], payload[midpoint:]]),
        )

    _stub_discovery_transport(monkeypatch, handler)

    with pytest.raises(OIDCDiscoveryError):
        await discover_connection_metadata(ISSUER, allow_private_hosts=False)


@pytest.mark.asyncio
async def test_discovery_refuses_a_document_that_never_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host that never finishes the document is refused on the total timeout."""
    monkeypatch.setattr("gateway.services.identity._discovery.DISCOVERY_TIMEOUT_SECONDS", 0.05)

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=_ForeverStream())

    _stub_discovery_transport(monkeypatch, handler)

    with pytest.raises(OIDCDiscoveryError):
        await discover_connection_metadata(ISSUER, allow_private_hosts=False)


@pytest.mark.asyncio
async def test_discovery_error_names_no_host_and_no_status(monkeypatch: pytest.MonkeyPatch) -> None:
    host = "leaky-issuer.example.test"
    issuer = f"https://{host}"

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    _stub_discovery_transport(monkeypatch, handler)

    with pytest.raises(OIDCDiscoveryError) as excinfo:
        await discover_connection_metadata(issuer, allow_private_hosts=False)

    message = str(excinfo.value)
    assert host not in message
    assert not any(character.isdigit() for character in message)

    rendered = "".join(traceback.format_exception(excinfo.value))
    assert host not in rendered
    assert excinfo.value.__cause__ is None
