"""Pinned networking primitives for gateway-managed web retrieval.

The transport in this module never resolves a request hostname itself. Callers
first create a :class:`ValidatedTarget`, then attach it to an HTTPX request. The
connection pool retains the canonical URL origin for TLS SNI, certificate
verification, and HTTP Host while its network backend dials only an admitted IP.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import ssl
import zlib
from collections import OrderedDict
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Protocol, TypeAlias, TypeVar

import httpcore
import httpx
from httpcore._backends.auto import AutoBackend
from httpcore._backends.base import AsyncNetworkBackend, AsyncNetworkStream
from httpx._transports.default import AsyncResponseStream, map_httpcore_exceptions

from gateway.services.web_retrieval_policy import (
    CanonicalOrigin,
    CanonicalWebURL,
    DomainPolicy,
    IPAddress,
    WebURLValidationError,
    canonicalize_web_url,
    resolve_redirect_url,
)

PINNED_TARGET_EXTENSION = "otari.validated_target"
MAX_WEB_REDIRECTS = 5
MAX_DECODED_BODY_BYTES = 5 * 1024 * 1024
NETWORK_DEADLINE_SECONDS = 5.0
MAX_PINNED_POOLS = 100
_MAX_DECODE_CHUNK_BYTES = 64 * 1024
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

T = TypeVar("T")
PoolKey: TypeAlias = tuple[str, str, int, str]


@dataclass(slots=True)
class _PoolEntry:
    pool: httpcore.AsyncConnectionPool
    active_responses: int = 0


class RetrievalTargetError(ValueError):
    """Raised when a URL cannot become an admitted pinned target."""


class RetrievalDomainPolicyError(RetrievalTargetError):
    """Raised when workspace policy denies a retrieval destination."""


class RetrievalAddressError(RetrievalTargetError):
    """Raised when DNS or address safety checks deny a target."""


class PinnedTransportError(httpx.TransportError):
    """Raised when a pinned request violates its validated-target contract."""


class RedirectValidationError(ValueError):
    """Raised when a manual redirect walk cannot safely continue."""


class NetworkDeadlineExceeded(TimeoutError):
    """Raised when the one wall-clock retrieval deadline expires."""


class ContentDecodingError(ValueError):
    """Raised when a response encoding is unsupported, malformed, or incomplete."""


class AddressResolver(Protocol):
    """Resolve a canonical authority without blocking the event loop."""

    async def resolve(self, host: str, port: int) -> Sequence[IPAddress]: ...


class SystemAddressResolver:
    """The default async system resolver used before any connection is opened."""

    async def resolve(self, host: str, port: int) -> Sequence[IPAddress]:
        loop = asyncio.get_running_loop()
        try:
            records = await loop.getaddrinfo(
                host,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
        except socket.gaierror:
            return ()

        result: list[IPAddress] = []
        seen: set[IPAddress] = set()
        for _family, _type, _proto, _canonname, sockaddr in records:
            try:
                address = ipaddress.ip_address(sockaddr[0])
            except ValueError:
                continue
            if address not in seen:
                result.append(address)
                seen.add(address)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ValidatedTarget:
    """Canonical request URL plus every address admitted for this DNS result."""

    canonical_url: CanonicalWebURL
    addresses: tuple[IPAddress, ...]

    @property
    def url(self) -> httpx.URL:
        return self.canonical_url.url

    @property
    def origin(self) -> CanonicalOrigin:
        return self.canonical_url.origin

    @property
    def display_url(self) -> str:
        return self.canonical_url.display_url


def _blocked_address_reason(address: IPAddress) -> str | None:
    if address.is_unspecified:
        return "unspecified"
    if address.is_loopback:
        return "loopback"
    if address.is_link_local:
        return "link-local"
    if address.is_multicast:
        return "multicast"
    if address.is_private:
        return "private"
    if address.is_reserved:
        return "reserved"
    if not address.is_global:
        return "non-global"
    return None


class NetworkDeadline:
    """One absolute wall-clock deadline shared by every network operation."""

    def __init__(self, seconds: float = NETWORK_DEADLINE_SECONDS, *, now: Callable[[], float] = monotonic) -> None:
        if seconds <= 0:
            raise ValueError("network deadline must be positive")
        self._now = now
        self._expires_at = now() + seconds

    @property
    def remaining(self) -> float:
        remaining = self._expires_at - self._now()
        if remaining <= 0:
            raise NetworkDeadlineExceeded("web retrieval network deadline exceeded")
        return remaining

    async def run(self, operation: Awaitable[T]) -> T:
        """Run one operation within the remaining shared deadline."""
        try:
            async with asyncio.timeout(self.remaining):
                return await operation
        except TimeoutError as exc:
            raise NetworkDeadlineExceeded("web retrieval network deadline exceeded") from exc


async def validate_retrieval_target(
    value: str | CanonicalWebURL,
    *,
    policy: DomainPolicy | None = None,
    resolver: AddressResolver | None = None,
    deadline: NetworkDeadline | None = None,
) -> ValidatedTarget:
    """Canonicalize, resolve, and admit a destination before connection setup.

    If any DNS answer is unsafe, the whole target is rejected. This prevents a
    hostname with mixed public and internal answers from selecting the internal
    address through retry or address ordering.
    """
    try:
        canonical_url = value if isinstance(value, CanonicalWebURL) else canonicalize_web_url(value)
    except WebURLValidationError as exc:
        raise RetrievalTargetError(str(exc)) from exc

    if policy is not None and not policy.permits(canonical_url.origin.host):
        raise RetrievalDomainPolicyError("destination is disallowed by domain policy")

    literal = canonical_url.origin.host.ip
    if literal is not None:
        resolved: Sequence[IPAddress] = (literal,)
    else:
        active_resolver = resolver or SystemAddressResolver()
        operation = active_resolver.resolve(canonical_url.origin.host.value, canonical_url.origin.port)
        resolved = await deadline.run(operation) if deadline is not None else await operation
    if not resolved:
        raise RetrievalAddressError("destination hostname could not be resolved")

    admitted: list[IPAddress] = []
    seen: set[IPAddress] = set()
    for address in resolved:
        canonical_address = ipaddress.ip_address(str(address))
        reason = _blocked_address_reason(canonical_address)
        if reason is not None:
            raise RetrievalAddressError(f"destination resolves to a disallowed {reason} address")
        if canonical_address not in seen:
            admitted.append(canonical_address)
            seen.add(canonical_address)
    if not admitted:
        raise RetrievalAddressError("destination hostname had no usable addresses")
    return ValidatedTarget(canonical_url=canonical_url, addresses=tuple(admitted))


class RedirectTracker:
    """State for a bounded, equivalence-aware manual redirect walk."""

    def __init__(self, initial: CanonicalWebURL, *, max_redirects: int = MAX_WEB_REDIRECTS) -> None:
        if max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")
        self._current = initial
        self._max_redirects = max_redirects
        self._redirects_followed = 0
        self._visited = {initial.redirect_loop_key}

    @property
    def current(self) -> CanonicalWebURL:
        return self._current

    @property
    def redirects_followed(self) -> int:
        return self._redirects_followed

    def advance(self, location: str | None) -> CanonicalWebURL:
        """Validate and record one redirect target without resolving or dialing it."""
        if location is None:
            raise RedirectValidationError("redirect response is missing Location")
        if self._redirects_followed >= self._max_redirects:
            raise RedirectValidationError("redirect hop limit exceeded")
        try:
            next_url = resolve_redirect_url(self._current, location)
        except WebURLValidationError as exc:
            raise RedirectValidationError("redirect target is malformed") from exc
        if self._current.origin.scheme == "https" and next_url.origin.scheme == "http":
            raise RedirectValidationError("HTTPS-to-HTTP redirect is not allowed")
        if next_url.redirect_loop_key in self._visited:
            raise RedirectValidationError("redirect loop detected")

        self._redirects_followed += 1
        self._visited.add(next_url.redirect_loop_key)
        self._current = next_url
        return next_url


def is_redirect_status(status_code: int) -> bool:
    return status_code in _REDIRECT_STATUSES


class _PinnedNetworkBackend(AsyncNetworkBackend):
    """Ignore origin DNS and dial one already-admitted address."""

    def __init__(
        self,
        *,
        origin: CanonicalOrigin,
        address: IPAddress,
        backend: AsyncNetworkBackend,
    ) -> None:
        self._origin = origin
        self._address = address
        self._backend = backend

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        if host.lower().rstrip(".") != self._origin.host.value or port != self._origin.port:
            raise PinnedTransportError("connection origin does not match validated target")
        return await self._backend.connect_tcp(
            str(self._address),
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        raise PinnedTransportError("Unix sockets are not supported by the web retrieval transport")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _LeasedAsyncResponseStream(httpx.AsyncByteStream):
    """Release one transport pool lease when its response stream closes."""

    def __init__(
        self,
        stream: AsyncIterable[bytes],
        release: Callable[[], Awaitable[None]],
    ) -> None:
        self._stream = AsyncResponseStream(stream)
        self._release = release
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._stream.aclose()
        finally:
            await self._release()


class PinnedAsyncHTTPTransport(httpx.AsyncBaseTransport):
    """HTTPX transport with pools isolated by canonical origin and pinned IP.

    Each address has its own HTTP Core pool. A later validation may reuse a
    connection only when that exact address remains in the target's admitted
    address set. Idle pools are evicted in least-recently-used order at the
    configured bound; if every retained pool is active, a new origin fails fast
    rather than closing an in-use connection or growing without limit. Connect
    failures retry only the remaining validated addresses; no retry performs
    fresh DNS resolution.
    """

    def __init__(
        self,
        *,
        ssl_context: ssl.SSLContext | None = None,
        backend_factory: Callable[[], AsyncNetworkBackend] = AutoBackend,
        max_connections: int = 10,
        max_keepalive_connections: int = 10,
        keepalive_expiry: float = 5.0,
        max_pools: int = MAX_PINNED_POOLS,
    ) -> None:
        if max_pools <= 0:
            raise ValueError("max_pools must be positive")
        self._ssl_context = ssl_context or httpx.create_ssl_context(verify=True, trust_env=False)
        self._backend_factory = backend_factory
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        self._keepalive_expiry = keepalive_expiry
        self._max_pools = max_pools
        self._pools: OrderedDict[PoolKey, _PoolEntry] = OrderedDict()
        self._pool_lock = asyncio.Lock()
        self._closed = False

    @staticmethod
    def _pool_key(target: ValidatedTarget, address: IPAddress) -> PoolKey:
        return (
            target.origin.scheme,
            target.origin.host.value,
            target.origin.port,
            str(address),
        )

    async def _acquire_pool(self, target: ValidatedTarget, address: IPAddress) -> tuple[PoolKey, _PoolEntry]:
        key = self._pool_key(target, address)
        evicted: _PoolEntry | None = None
        async with self._pool_lock:
            if self._closed:
                raise PinnedTransportError("web retrieval transport is closed")
            entry = self._pools.get(key)
            if entry is None:
                if len(self._pools) >= self._max_pools:
                    for candidate_key, candidate in self._pools.items():
                        if candidate.active_responses == 0:
                            evicted = candidate
                            del self._pools[candidate_key]
                            break
                    else:
                        raise PinnedTransportError("web retrieval connection pool capacity is in use")
                network_backend = _PinnedNetworkBackend(
                    origin=target.origin,
                    address=address,
                    backend=self._backend_factory(),
                )
                entry = _PoolEntry(
                    pool=httpcore.AsyncConnectionPool(
                        ssl_context=self._ssl_context,
                        max_connections=self._max_connections,
                        max_keepalive_connections=self._max_keepalive_connections,
                        keepalive_expiry=self._keepalive_expiry,
                        http1=True,
                        http2=False,
                        retries=0,
                        network_backend=network_backend,
                    )
                )
                self._pools[key] = entry
            else:
                self._pools.move_to_end(key)
            entry.active_responses += 1

        if evicted is not None:
            try:
                await evicted.pool.aclose()
            except BaseException:
                await self._release_pool(key, entry)
                raise
        return key, entry

    async def _release_pool(self, key: PoolKey, entry: _PoolEntry) -> None:
        async with self._pool_lock:
            entry.active_responses -= 1
            if self._pools.get(key) is entry:
                self._pools.move_to_end(key)

    @staticmethod
    def _target_from_request(request: httpx.Request) -> ValidatedTarget:
        target = request.extensions.get(PINNED_TARGET_EXTENSION)
        if not isinstance(target, ValidatedTarget):
            raise PinnedTransportError("request is missing its validated retrieval target")
        if request.url != target.url:
            raise PinnedTransportError("request URL does not match its validated retrieval target")
        if request.method != "GET":
            raise PinnedTransportError("web retrieval transport permits GET requests only")
        if request.headers.get("host") != target.origin.authority:
            raise PinnedTransportError("request Host does not match its validated retrieval target")
        return target

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not isinstance(request.stream, httpx.AsyncByteStream):
            raise PinnedTransportError("web retrieval request stream must be asynchronous")
        target = self._target_from_request(request)

        last_connect_error: httpx.ConnectError | httpx.ConnectTimeout | None = None
        for address in target.addresses:
            key, entry = await self._acquire_pool(target, address)
            core_request = httpcore.Request(
                method=request.method,
                url=httpcore.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=request.stream,
                extensions=request.extensions,
            )
            try:
                with map_httpcore_exceptions():
                    response = await entry.pool.handle_async_request(core_request)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_connect_error = exc
                await self._release_pool(key, entry)
                continue
            except BaseException:
                await self._release_pool(key, entry)
                raise

            assert isinstance(response.stream, AsyncIterable)
            stream = _LeasedAsyncResponseStream(
                response.stream,
                lambda: self._release_pool(key, entry),
            )
            return httpx.Response(
                status_code=response.status,
                headers=response.headers,
                stream=stream,
                extensions=response.extensions,
            )

        if last_connect_error is not None:
            raise last_connect_error
        raise PinnedTransportError("validated retrieval target has no addresses")

    async def aclose(self) -> None:
        async with self._pool_lock:
            if self._closed:
                return
            self._closed = True
            pools = tuple(entry.pool for entry in self._pools.values())
            self._pools.clear()
        await asyncio.gather(*(pool.aclose() for pool in pools))


@dataclass(frozen=True, slots=True)
class CappedBody:
    content: bytes
    truncated: bool


async def _iter_bounded_decoded_chunks(
    response: httpx.Response,
    *,
    chunk_size: int,
) -> AsyncIterator[bytes]:
    """Decode a streamed body without allowing one decoder call to inflate freely."""
    try:
        loaded_content = response.content
    except httpx.ResponseNotRead:
        loaded_content = None
    if loaded_content is not None:
        for offset in range(0, len(loaded_content), chunk_size):
            yield loaded_content[offset : offset + chunk_size]
        return

    encodings = [
        encoding.strip().lower()
        for value in response.headers.get_list("content-encoding")
        for encoding in value.split(",")
        if encoding.strip()
    ]
    if not encodings or encodings == ["identity"]:
        async for raw_chunk in response.aiter_raw(chunk_size=chunk_size):
            yield raw_chunk
        return
    if len(encodings) != 1 or encodings[0] not in {"gzip", "x-gzip", "deflate"}:
        raise ContentDecodingError("response uses an unsupported content encoding")

    encoding = encodings[0]
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16 if encoding in {"gzip", "x-gzip"} else zlib.MAX_WBITS)
    can_retry_raw_deflate = encoding == "deflate"
    async for raw_chunk in response.aiter_raw(chunk_size=chunk_size):
        pending = raw_chunk
        while pending:
            try:
                decoded = decompressor.decompress(pending, max_length=chunk_size)
            except zlib.error as exc:
                if not can_retry_raw_deflate:
                    raise ContentDecodingError("response content encoding is malformed") from exc
                decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
                can_retry_raw_deflate = False
                pending = raw_chunk
                continue
            can_retry_raw_deflate = False
            pending = decompressor.unconsumed_tail
            if decoded:
                yield decoded
    if not decompressor.eof or decompressor.unused_data:
        raise ContentDecodingError("response content encoding is incomplete or has trailing data")


async def read_capped_decoded_body(
    response: httpx.Response,
    *,
    deadline: NetworkDeadline,
    max_bytes: int = MAX_DECODED_BODY_BYTES,
    chunk_size: int = 65536,
) -> CappedBody:
    """Read at most ``max_bytes`` decoded response bytes under the deadline.

    One byte beyond the cap is observed, but never retained, so a body exactly at
    the ceiling is distinguishable from a truncated body. Compressed streams are
    decoded incrementally with a hard output bound on every decoder call, before
    any decoded chunk reaches the retained-body buffer.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes cannot be negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    decode_chunk_size = min(chunk_size, _MAX_DECODE_CHUNK_BYTES)

    async def read() -> CappedBody:
        buffer = bytearray()
        async for chunk in _iter_bounded_decoded_chunks(response, chunk_size=decode_chunk_size):
            remaining = max_bytes - len(buffer)
            if len(chunk) > remaining:
                if remaining > 0:
                    buffer.extend(chunk[:remaining])
                return CappedBody(content=bytes(buffer), truncated=True)
            buffer.extend(chunk)
        return CappedBody(content=bytes(buffer), truncated=False)

    try:
        return await deadline.run(read())
    finally:
        await response.aclose()


@dataclass(frozen=True, slots=True)
class UTF8Truncation:
    text: str
    truncated: bool
    byte_length: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "byte_length", len(self.text.encode("utf-8")))


def truncate_utf8(value: str, max_bytes: int, *, suffix: str = "") -> UTF8Truncation:
    """Return a valid UTF-8 head within ``max_bytes``, optionally ending in a suffix."""
    if max_bytes < 0:
        raise ValueError("max_bytes cannot be negative")
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return UTF8Truncation(text=value, truncated=False)

    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) > max_bytes:
        suffix_bytes = suffix_bytes[:max_bytes]
        suffix = suffix_bytes.decode("utf-8", errors="ignore")
        suffix_bytes = suffix.encode("utf-8")
    head_limit = max_bytes - len(suffix_bytes)
    head = encoded[:head_limit].decode("utf-8", errors="ignore")
    result = f"{head}{suffix}"
    return UTF8Truncation(text=result, truncated=True)
