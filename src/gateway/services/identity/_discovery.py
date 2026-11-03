"""Discover a customer's own OpenID Connect issuer configuration.

The issuer is a URL an organization administrator typed into a form, so it
names a host this deployment does not control and may not trust. The fetch
goes through the checked federation transport, with the response stream
capped in size and the whole exchange bounded by one wall-clock deadline, so
a hostile, slow, or oversized provider cannot exhaust memory or hang the
caller.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from apron_auth.errors import ConfigurationError, OidcDiscoveryError
from apron_auth.providers import oidc

from gateway.exceptions.identity_exceptions import OIDCDiscoveryError
from gateway.log_config import logger
from gateway.services.web_retrieval_network import (
    CheckedFederationTransport,
    NetworkDeadline,
    NetworkDeadlineExceeded,
)

DISCOVERY_MAX_BYTES = 256 * 1024
DISCOVERY_TIMEOUT_SECONDS = 10.0
DISCOVERY_TTL = timedelta(hours=12)


@dataclass(frozen=True)
class DiscoveredProvider:
    """A provider's configuration, read once and ready to store for ``DISCOVERY_TTL``."""

    resolved_issuer: str
    document: dict[str, Any]
    discovered_at: datetime


class _DocumentTooLargeError(httpx.TransportError):
    """Raised when a provider's configuration document exceeds ``DISCOVERY_MAX_BYTES``."""


class _DocumentEncodedError(httpx.TransportError):
    """Raised when a provider's response declares a content encoding other than identity."""


class _CappedDocumentStream(httpx.AsyncByteStream):
    """Stop iterating a response body once it passes the configured byte ceiling.

    The ceiling bounds what is read, not what was read: a chunk that would
    cross it is never retained, so a document that never stops streaming
    cannot be buffered into memory first and rejected after.
    """

    def __init__(self, stream: httpx.AsyncByteStream, *, max_bytes: int) -> None:
        self._stream = stream
        self._max_bytes = max_bytes

    async def __aiter__(self) -> AsyncIterator[bytes]:
        read = 0
        async for chunk in self._stream:
            read += len(chunk)
            if read > self._max_bytes:
                raise _DocumentTooLargeError("document exceeds the configured byte ceiling")
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class _BoundedDocumentTransport(httpx.AsyncBaseTransport):
    """Wrap a transport so the document it returns is capped in size as it streams.

    The request asks for an uncompressed body and the response is refused if it declares an
    encoding anyway, so the byte ceiling below bounds the document itself and not a decoder's
    unbounded output: a compressed reply would otherwise grow past the ceiling after decoding
    while the capped bytes on the wire stayed well inside it.
    """

    def __init__(self, transport: httpx.AsyncBaseTransport, *, max_bytes: int) -> None:
        self._transport = transport
        self._max_bytes = max_bytes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        headers = request.headers.copy()
        headers["accept-encoding"] = "identity"
        uncompressed_request = httpx.Request(
            request.method,
            request.url,
            headers=headers,
            stream=request.stream,
            extensions=request.extensions,
        )
        response = await self._transport.handle_async_request(uncompressed_request)
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in ("", "identity"):
            raise _DocumentEncodedError("document response declares a content encoding other than identity")
        if not isinstance(response.stream, httpx.AsyncByteStream):
            raise httpx.TransportError("document transport returned a synchronous stream")
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_CappedDocumentStream(response.stream, max_bytes=self._max_bytes),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()


async def discover_connection_metadata(issuer: str, *, allow_private_hosts: bool) -> DiscoveredProvider:
    """Read and validate the configuration an organization's own OIDC issuer publishes.

    The document is fetched once, through the checked federation transport, and refused if it
    names a different issuer, advertises a code-challenge method set that excludes ``S256``, grows
    past ``DISCOVERY_MAX_BYTES``, or the exchange runs past ``DISCOVERY_TIMEOUT_SECONDS``.

    Raises:
        OIDCDiscoveryError: If the issuer could not be read into a usable configuration.
    """

    def transport_factory(_url: str) -> httpx.AsyncBaseTransport:
        return _BoundedDocumentTransport(
            CheckedFederationTransport(allow_private_hosts=allow_private_hosts),
            max_bytes=DISCOVERY_MAX_BYTES,
        )

    deadline = NetworkDeadline(seconds=DISCOVERY_TIMEOUT_SECONDS)
    try:
        metadata = await deadline.run(lambda: oidc.discover(issuer, transport_factory=transport_factory))
    except (OidcDiscoveryError, ConfigurationError, NetworkDeadlineExceeded):
        logger.warning("OIDC issuer discovery failed for issuer %s", issuer, exc_info=True)
        raise OIDCDiscoveryError from None

    # NOTE: This refuses a provider naming no S256 method, including one that omits the field,
    # which RFC 8414 allows.
    if "S256" not in metadata.code_challenge_methods:
        logger.warning("OIDC issuer discovery refused issuer %s: no S256 code-challenge method advertised", issuer)
        raise OIDCDiscoveryError from None

    if metadata.issuer is None:  # pragma: no cover
        raise OIDCDiscoveryError from None

    return DiscoveredProvider(
        resolved_issuer=metadata.issuer,
        document=metadata.model_dump(mode="json"),
        discovered_at=datetime.now(UTC),
    )
