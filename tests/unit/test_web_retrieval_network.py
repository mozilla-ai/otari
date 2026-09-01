"""Focused tests for DNS admission, pinning, redirects, and network bounds."""

import asyncio
import gzip
import ipaddress
import ssl
import tracemalloc
import zlib
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpcore
import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from httpcore._backends.base import AsyncNetworkBackend, AsyncNetworkStream

from gateway.services.web_retrieval_network import (
    PINNED_TARGET_EXTENSION,
    AddressResolver,
    CappedBody,
    ContentDecodingError,
    NetworkDeadline,
    NetworkDeadlineExceeded,
    PinnedAsyncHTTPTransport,
    PinnedTransportError,
    RedirectTracker,
    RedirectValidationError,
    RetrievalAddressError,
    RetrievalDomainPolicyError,
    SystemAddressResolver,
    UTF8Truncation,
    ValidatedTarget,
    is_redirect_status,
    read_capped_decoded_body,
    truncate_utf8,
    validate_retrieval_target,
)
from gateway.services.web_retrieval_policy import (
    DomainPolicy,
    IPAddress,
    canonicalize_domain_rules,
    canonicalize_web_url,
)

_PUBLIC_V4 = ipaddress.ip_address("93.184.216.34")
_OTHER_PUBLIC_V4 = ipaddress.ip_address("8.8.8.8")
_PUBLIC_V6 = ipaddress.ip_address("2606:4700:4700::1111")
_PRIVATE_V4 = ipaddress.ip_address("10.0.0.5")


class OneChunkByteStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self._content = content
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._content

    async def aclose(self) -> None:
        self.closed = True


class StaticResolver:
    def __init__(self, addresses: Sequence[IPAddress]) -> None:
        self.addresses = addresses
        self.calls: list[tuple[str, int]] = []

    async def resolve(self, host: str, port: int) -> Sequence[IPAddress]:
        self.calls.append((host, port))
        return self.addresses


@pytest.mark.asyncio
async def test_target_contains_canonical_origin_and_complete_resolved_set() -> None:
    resolver = StaticResolver((_PUBLIC_V6, _PUBLIC_V4, _PUBLIC_V4))

    target = await validate_retrieval_target(
        "HTTPS://Example.COM.:443/a%2Fb?q=z%2Fz#fragment",
        resolver=resolver,
    )

    assert str(target.url) == "https://example.com/a%2Fb?q=z%2Fz"
    assert target.origin.host.value == "example.com"
    assert target.origin.port == 443
    assert target.addresses == (_PUBLIC_V6, _PUBLIC_V4)
    assert resolver.calls == [("example.com", 443)]


@pytest.mark.asyncio
async def test_ip_literal_skips_dns_and_is_canonicalized() -> None:
    resolver = StaticResolver((_OTHER_PUBLIC_V4,))

    target = await validate_retrieval_target(
        "https://[2606:4700:4700:0:0:0:0:1111]/",
        resolver=resolver,
    )

    assert target.origin.host.value == "2606:4700:4700::1111"
    assert target.addresses == (_PUBLIC_V6,)
    assert resolver.calls == []


@pytest.mark.parametrize(
    "address",
    [
        "0.0.0.0",
        "127.0.0.1",
        "10.0.0.1",
        "100.64.0.1",
        "169.254.169.254",
        "224.0.0.1",
        "240.0.0.1",
        "::",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
    ],
)
@pytest.mark.asyncio
async def test_direct_target_rejects_nonpublic_addresses(address: str) -> None:
    resolver = StaticResolver((ipaddress.ip_address(address),))

    with pytest.raises(RetrievalAddressError, match="disallowed"):
        await validate_retrieval_target("https://example.com/", resolver=resolver)


@pytest.mark.asyncio
async def test_mixed_public_and_private_dns_answers_fail_closed() -> None:
    resolver = StaticResolver((_PUBLIC_V4, _PRIVATE_V4))

    with pytest.raises(RetrievalAddressError, match="private"):
        await validate_retrieval_target("https://example.com/", resolver=resolver)


@pytest.mark.asyncio
async def test_unresolvable_target_is_rejected() -> None:
    with pytest.raises(RetrievalAddressError, match="could not be resolved"):
        await validate_retrieval_target("https://example.invalid/", resolver=StaticResolver(()))


@pytest.mark.asyncio
async def test_domain_policy_is_checked_before_dns() -> None:
    resolver = StaticResolver((_PUBLIC_V4,))
    policy = DomainPolicy(allowed=canonicalize_domain_rules(("mozilla.org",)))

    with pytest.raises(RetrievalDomainPolicyError):
        await validate_retrieval_target("https://example.com/", resolver=resolver, policy=policy)

    assert resolver.calls == []


@pytest.mark.asyncio
async def test_domain_policy_matches_canonical_unicode_identity() -> None:
    resolver = StaticResolver((_PUBLIC_V4,))
    policy = DomainPolicy(allowed=canonicalize_domain_rules(("BÜCHER.example",)))

    target = await validate_retrieval_target(
        "https://xn--bcher-kva.example/page",
        resolver=resolver,
        policy=policy,
    )

    assert target.origin.host.value == "xn--bcher-kva.example"


class ScriptedNetworkStream(AsyncNetworkStream):
    def __init__(
        self,
        response: bytes = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK",
    ) -> None:
        self.response = response
        self.read_count = 0
        self.writes: list[bytes] = []
        self.tls_server_names: list[str | None] = []
        self.closed = False

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        del max_bytes, timeout
        if self.read_count:
            return b""
        self.read_count += 1
        return self.response

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        del timeout
        self.writes.append(buffer)

    async def aclose(self) -> None:
        self.closed = True

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        del ssl_context, timeout
        self.tls_server_names.append(server_hostname)
        return self

    def get_extra_info(self, info: str) -> Any:
        del info
        return None


class ScriptedNetworkBackend(AsyncNetworkBackend):
    def __init__(self, *, fail_connect: bool = False) -> None:
        self.fail_connect = fail_connect
        self.connects: list[tuple[str, int]] = []
        self.streams: list[ScriptedNetworkStream] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        del timeout, local_address, socket_options
        self.connects.append((host, port))
        if self.fail_connect:
            raise httpcore.ConnectError("scripted connect failure")
        stream = ScriptedNetworkStream()
        self.streams.append(stream)
        return stream

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        del path, timeout, socket_options
        raise AssertionError("Unix connection was not expected")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class BackendFactory:
    def __init__(self, failures: Sequence[bool] = ()) -> None:
        self.failures = list(failures)
        self.backends: list[ScriptedNetworkBackend] = []

    def __call__(self) -> AsyncNetworkBackend:
        fail = self.failures.pop(0) if self.failures else False
        backend = ScriptedNetworkBackend(fail_connect=fail)
        self.backends.append(backend)
        return backend


async def _request_with_target(
    client: httpx.AsyncClient,
    target: ValidatedTarget,
    *,
    url: str | None = None,
    stream: bool = False,
) -> httpx.Response:
    request = client.build_request("GET", url or str(target.url))
    request.extensions[PINNED_TARGET_EXTENSION] = target
    return await client.send(request, stream=stream)


@pytest.mark.asyncio
async def test_transport_dials_pinned_ip_but_preserves_tls_and_http_authority() -> None:
    factory = BackendFactory()
    transport = PinnedAsyncHTTPTransport(backend_factory=factory)
    target = ValidatedTarget(
        canonical_url=canonicalize_web_url("https://Example.COM.:8443/page?q=1"),
        addresses=(_PUBLIC_V4,),
    )

    async with httpx.AsyncClient(transport=transport) as client:
        response = await _request_with_target(client, target)
        assert await response.aread() == b"OK"

    backend = factory.backends[0]
    stream = backend.streams[0]
    assert backend.connects == [(str(_PUBLIC_V4), 8443)]
    assert stream.tls_server_names == ["example.com"]
    request_bytes = b"".join(stream.writes)
    assert b"GET /page?q=1 HTTP/1.1\r\n" in request_bytes
    assert b"Host: example.com:8443\r\n" in request_bytes


def _write_test_certificate(tmp_path: Path, hostname: str) -> tuple[Path, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, hostname)])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(minutes=5))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    certificate_path = tmp_path / "certificate.pem"
    key_path = tmp_path / "key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path


@pytest.mark.asyncio
async def test_pinned_tls_verifies_certificate_against_canonical_hostname(tmp_path: Path) -> None:
    hostname = "canonical.example"
    certificate_path, key_path = _write_test_certificate(tmp_path, hostname)
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certificate_path, key_path)

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0, ssl=server_context)
    assert server.sockets
    port = int(server.sockets[0].getsockname()[1])
    client_context = ssl.create_default_context(cafile=str(certificate_path))
    transport = PinnedAsyncHTTPTransport(ssl_context=client_context)
    loopback = ipaddress.ip_address("127.0.0.1")
    matching = ValidatedTarget(canonicalize_web_url(f"https://{hostname}:{port}/"), (loopback,))
    wrong_host = ValidatedTarget(canonicalize_web_url(f"https://wrong.example:{port}/"), (loopback,))

    try:
        async with httpx.AsyncClient(transport=transport) as client:
            response = await _request_with_target(client, matching)
            assert await response.aread() == b"OK"
            with pytest.raises(httpx.ConnectError):
                await _request_with_target(client, wrong_host)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_transport_does_not_repeat_dns_after_validation() -> None:
    class RebindingResolver:
        def __init__(self) -> None:
            self.calls = 0

        async def resolve(self, host: str, port: int) -> Sequence[IPAddress]:
            del host, port
            self.calls += 1
            return (_PUBLIC_V4,) if self.calls == 1 else (_PRIVATE_V4,)

    resolver = RebindingResolver()
    target = await validate_retrieval_target("http://example.com/page", resolver=resolver)
    # This is what an ordinary lookup at connection time would see.
    assert await resolver.resolve("example.com", 80) == (_PRIVATE_V4,)

    factory = BackendFactory()
    transport = PinnedAsyncHTTPTransport(backend_factory=factory)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await _request_with_target(client, target)
        assert await response.aread() == b"OK"

    assert resolver.calls == 2
    assert factory.backends[0].connects == [(str(_PUBLIC_V4), 80)]


@pytest.mark.asyncio
async def test_transport_retries_only_validated_addresses() -> None:
    factory = BackendFactory((True, False))
    transport = PinnedAsyncHTTPTransport(backend_factory=factory)
    target = ValidatedTarget(
        canonical_url=canonicalize_web_url("http://example.com/page"),
        addresses=(_PUBLIC_V4, _OTHER_PUBLIC_V4),
    )

    async with httpx.AsyncClient(transport=transport) as client:
        response = await _request_with_target(client, target)
        assert await response.aread() == b"OK"

    assert factory.backends[0].connects == [(str(_PUBLIC_V4), 80)]
    assert factory.backends[1].connects == [(str(_OTHER_PUBLIC_V4), 80)]


def test_transport_requires_positive_pool_bound() -> None:
    with pytest.raises(ValueError, match="max_pools"):
        PinnedAsyncHTTPTransport(max_pools=0)


@pytest.mark.asyncio
async def test_pool_keys_isolate_origin_and_pinned_address() -> None:
    factory = BackendFactory()
    transport = PinnedAsyncHTTPTransport(backend_factory=factory)
    first = ValidatedTarget(canonicalize_web_url("http://example.com/one"), (_PUBLIC_V4,))
    same_origin_new_path = ValidatedTarget(canonicalize_web_url("http://example.com/two"), (_PUBLIC_V4,))
    changed_address = ValidatedTarget(canonicalize_web_url("http://example.com/three"), (_OTHER_PUBLIC_V4,))
    changed_origin = ValidatedTarget(canonicalize_web_url("http://other.example/four"), (_PUBLIC_V4,))

    async with httpx.AsyncClient(transport=transport) as client:
        for target in (first, same_origin_new_path, changed_address, changed_origin):
            response = await _request_with_target(client, target)
            await response.aread()

        assert len(transport._pools) == 3  # noqa: SLF001 - verifies the security pool key


@pytest.mark.asyncio
async def test_pool_bound_evicts_least_recently_used_idle_pool() -> None:
    factory = BackendFactory()
    transport = PinnedAsyncHTTPTransport(backend_factory=factory, max_pools=2)
    first = ValidatedTarget(canonicalize_web_url("http://first.example/"), (_PUBLIC_V4,))
    second = ValidatedTarget(canonicalize_web_url("http://second.example/"), (_PUBLIC_V4,))
    third = ValidatedTarget(canonicalize_web_url("http://third.example/"), (_PUBLIC_V4,))

    async with httpx.AsyncClient(transport=transport) as client:
        await (await _request_with_target(client, first)).aread()
        await (await _request_with_target(client, second)).aread()
        await (await _request_with_target(client, first)).aread()
        await (await _request_with_target(client, third)).aread()

        assert list(transport._pools) == [  # noqa: SLF001 - verifies bounded LRU eviction
            transport._pool_key(first, _PUBLIC_V4),  # noqa: SLF001
            transport._pool_key(third, _PUBLIC_V4),  # noqa: SLF001
        ]


@pytest.mark.asyncio
async def test_pool_bound_never_evicts_an_active_response() -> None:
    factory = BackendFactory()
    transport = PinnedAsyncHTTPTransport(backend_factory=factory, max_pools=1)
    first = ValidatedTarget(canonicalize_web_url("http://first.example/"), (_PUBLIC_V4,))
    second = ValidatedTarget(canonicalize_web_url("http://second.example/"), (_PUBLIC_V4,))

    async with httpx.AsyncClient(transport=transport) as client:
        active_response = await _request_with_target(client, first, stream=True)
        with pytest.raises(PinnedTransportError, match="capacity"):
            await _request_with_target(client, second, stream=True)

        await active_response.aclose()
        second_response = await _request_with_target(client, second, stream=True)
        await second_response.aclose()

        assert list(transport._pools) == [  # noqa: SLF001 - verifies release before eviction
            transport._pool_key(second, _PUBLIC_V4)  # noqa: SLF001
        ]


@pytest.mark.asyncio
async def test_target_with_changed_address_set_cannot_use_old_address_pool() -> None:
    factory = BackendFactory()
    transport = PinnedAsyncHTTPTransport(backend_factory=factory)
    old = ValidatedTarget(canonicalize_web_url("http://example.com/one"), (_PUBLIC_V4,))
    revalidated = ValidatedTarget(canonicalize_web_url("http://example.com/two"), (_OTHER_PUBLIC_V4,))

    async with httpx.AsyncClient(transport=transport) as client:
        await (await _request_with_target(client, old)).aread()
        await (await _request_with_target(client, revalidated)).aread()

    assert factory.backends[0].connects == [(str(_PUBLIC_V4), 80)]
    assert factory.backends[1].connects == [(str(_OTHER_PUBLIC_V4), 80)]


@pytest.mark.asyncio
async def test_transport_requires_exact_validated_url_and_get_method() -> None:
    factory = BackendFactory()
    transport = PinnedAsyncHTTPTransport(backend_factory=factory)
    target = ValidatedTarget(canonicalize_web_url("https://example.com/allowed"), (_PUBLIC_V4,))

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(PinnedTransportError, match="does not match"):
            await _request_with_target(client, target, url="https://example.com/different")

        request = client.build_request("POST", str(target.url))
        request.extensions[PINNED_TARGET_EXTENSION] = target
        with pytest.raises(PinnedTransportError, match="GET"):
            await client.send(request)

        request = client.build_request("GET", str(target.url), headers={"Host": "other.example"})
        request.extensions[PINNED_TARGET_EXTENSION] = target
        with pytest.raises(PinnedTransportError, match="Host"):
            await client.send(request)

        request = client.build_request("GET", str(target.url))
        with pytest.raises(PinnedTransportError, match="missing"):
            await client.send(request)

    assert factory.backends == []


def test_redirect_tracker_resolves_relative_targets_and_preserves_query() -> None:
    tracker = RedirectTracker(canonicalize_web_url("http://example.com/a/start?q=old"))

    target = tracker.advance("../next%2Fpart?signature=a%2Fb&n=2#fragment")

    assert str(target.url) == "http://example.com/next%2Fpart?signature=a%2Fb&n=2"
    assert tracker.redirects_followed == 1


def test_redirect_tracker_permits_http_to_https() -> None:
    tracker = RedirectTracker(canonicalize_web_url("http://example.com/start"))

    target = tracker.advance("https://example.com/secure")

    assert target.origin.scheme == "https"


def test_redirect_tracker_rejects_https_downgrade() -> None:
    tracker = RedirectTracker(canonicalize_web_url("https://example.com/start"))

    with pytest.raises(RedirectValidationError, match="HTTPS-to-HTTP"):
        tracker.advance("http://example.com/insecure")


@pytest.mark.asyncio
async def test_redirect_target_requires_fresh_policy_and_address_validation() -> None:
    tracker = RedirectTracker(canonicalize_web_url("https://public.example/start"))
    redirected = tracker.advance("https://private.example/next")
    policy = DomainPolicy(allowed=canonicalize_domain_rules(("public.example",)))
    resolver = StaticResolver((_PRIVATE_V4,))

    with pytest.raises(RetrievalDomainPolicyError):
        await validate_retrieval_target(redirected, policy=policy, resolver=resolver)

    assert resolver.calls == []


def test_redirect_tracker_detects_equivalent_origin_loop() -> None:
    tracker = RedirectTracker(canonicalize_web_url("https://BÜCHER.example./path?q=1"))

    with pytest.raises(RedirectValidationError, match="loop"):
        tracker.advance("https://xn--bcher-kva.example:443/path?q=1#different")


def test_query_change_is_not_a_loop_even_when_display_urls_match() -> None:
    tracker = RedirectTracker(canonicalize_web_url("https://example.com/path?token=one"))

    target = tracker.advance("?token=two")

    assert target.display_url == tracker.current.display_url == "https://example.com/path"
    assert str(target.url).endswith("?token=two")


def test_redirect_tracker_rejects_missing_location_and_hop_exhaustion() -> None:
    tracker = RedirectTracker(canonicalize_web_url("https://example.com/0"), max_redirects=2)

    with pytest.raises(RedirectValidationError, match="missing"):
        tracker.advance(None)
    tracker.advance("/1")
    tracker.advance("/2")
    with pytest.raises(RedirectValidationError, match="limit"):
        tracker.advance("/3")


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_manual_redirect_statuses(status: int) -> None:
    assert is_redirect_status(status)


@pytest.mark.parametrize("status", [200, 201, 300, 304, 305, 400])
def test_nonredirect_statuses(status: int) -> None:
    assert not is_redirect_status(status)


@pytest.mark.asyncio
async def test_network_deadline_is_shared_across_operations() -> None:
    deadline = NetworkDeadline(0.05)

    await deadline.run(asyncio.sleep(0.03))
    with pytest.raises(NetworkDeadlineExceeded):
        await deadline.run(asyncio.sleep(0.03))


@pytest.mark.asyncio
async def test_dns_resolution_observes_network_deadline() -> None:
    class SlowResolver:
        async def resolve(self, host: str, port: int) -> Sequence[IPAddress]:
            del host, port
            await asyncio.sleep(0.1)
            return (_PUBLIC_V4,)

    with pytest.raises(NetworkDeadlineExceeded):
        await validate_retrieval_target(
            "https://example.com/",
            resolver=SlowResolver(),
            deadline=NetworkDeadline(0.01),
        )


@pytest.mark.asyncio
async def test_capped_body_marks_overflow_without_retaining_it() -> None:
    response = httpx.Response(200, content=b"abcdefghijk")

    body = await read_capped_decoded_body(response, deadline=NetworkDeadline(1), max_bytes=8, chunk_size=3)

    assert body == CappedBody(content=b"abcdefgh", truncated=True)


@pytest.mark.asyncio
async def test_capped_body_closes_stream_after_truncation() -> None:
    stream = OneChunkByteStream(b"abcdefghijk")
    response = httpx.Response(200, stream=stream)

    body = await read_capped_decoded_body(response, deadline=NetworkDeadline(1), max_bytes=8, chunk_size=3)

    assert body == CappedBody(content=b"abcdefgh", truncated=True)
    assert stream.closed


@pytest.mark.asyncio
async def test_capped_body_distinguishes_exact_limit() -> None:
    response = httpx.Response(200, content=b"abcdefgh")

    body = await read_capped_decoded_body(response, deadline=NetworkDeadline(1), max_bytes=8, chunk_size=4)

    assert body == CappedBody(content=b"abcdefgh", truncated=False)


@pytest.mark.asyncio
async def test_capped_body_bounds_compressed_expansion_memory() -> None:
    max_bytes = 5 * 1024 * 1024
    expanded = b"a" * (20 * 1024 * 1024)
    response = httpx.Response(
        200,
        headers={"Content-Encoding": "gzip"},
        stream=OneChunkByteStream(gzip.compress(expanded, compresslevel=9)),
    )

    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        body = await read_capped_decoded_body(response, deadline=NetworkDeadline(1), max_bytes=max_bytes)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert body == CappedBody(content=b"a" * max_bytes, truncated=True)
    assert peak < max_bytes * 4


@pytest.mark.parametrize(
    ("encoding", "compressed"),
    [
        ("gzip", gzip.compress(b"decoded body")),
        ("deflate", zlib.compress(b"decoded body")),
    ],
)
@pytest.mark.asyncio
async def test_capped_body_completes_supported_compressed_streams(encoding: str, compressed: bytes) -> None:
    response = httpx.Response(
        200,
        headers={"Content-Encoding": encoding},
        stream=OneChunkByteStream(compressed),
    )

    body = await read_capped_decoded_body(response, deadline=NetworkDeadline(1))

    assert body == CappedBody(content=b"decoded body", truncated=False)


@pytest.mark.asyncio
async def test_capped_body_rejects_unsupported_content_encoding() -> None:
    response = httpx.Response(
        200,
        headers={"Content-Encoding": "br"},
        stream=OneChunkByteStream(b"not decoded"),
    )

    with pytest.raises(ContentDecodingError, match="unsupported"):
        await read_capped_decoded_body(response, deadline=NetworkDeadline(1))


@pytest.mark.asyncio
async def test_slow_stream_cannot_reset_absolute_deadline_per_chunk() -> None:
    class SlowStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for _ in range(10):
                await asyncio.sleep(0.01)
                yield b"a"

        async def aclose(self) -> None:
            self.closed = True

    stream = SlowStream()
    response = httpx.Response(200, stream=stream)

    with pytest.raises(NetworkDeadlineExceeded):
        await read_capped_decoded_body(response, deadline=NetworkDeadline(0.035), max_bytes=100)

    assert stream.closed


@pytest.mark.parametrize(
    ("value", "max_bytes", "suffix", "expected"),
    [
        ("plain", 5, "", UTF8Truncation("plain", False)),
        ("abcdef", 4, "", UTF8Truncation("abcd", True)),
        ("A😀B", 5, "", UTF8Truncation("A😀", True)),
        ("A😀B", 4, "", UTF8Truncation("A", True)),
        ("abcdef", 5, "…", UTF8Truncation("ab…", True)),
        ("abcdef", 2, "…", UTF8Truncation("ab", True)),
    ],
)
def test_utf8_safe_truncation(
    value: str,
    max_bytes: int,
    suffix: str,
    expected: UTF8Truncation,
) -> None:
    result = truncate_utf8(value, max_bytes, suffix=suffix)

    assert result == expected
    assert result.byte_length <= max_bytes


def test_system_resolver_is_protocol_compatible() -> None:
    resolver: AddressResolver = SystemAddressResolver()

    assert callable(resolver.resolve)
