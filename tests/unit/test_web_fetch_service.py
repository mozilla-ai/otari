"""Tests for shared bounded public-page retrieval and extraction."""

from __future__ import annotations

import ipaddress
from collections.abc import Callable
from typing import cast

import httpx
import pytest

from gateway.services.web_extraction import ExtractedText, ExtractionError, ExtractionSupervisor
from gateway.services.web_fetch_service import (
    UnsupportedContentTypeError,
    WebFetchHTTPStatusError,
    WebFetchService,
)
from gateway.services.web_retrieval_network import (
    MAX_DECODED_BODY_BYTES,
    PINNED_TARGET_EXTENSION,
    RedirectValidationError,
    RetrievalAddressError,
    RetrievalDomainPolicyError,
    ValidatedTarget,
)
from gateway.services.web_retrieval_policy import DomainPolicy, canonicalize_domain_rules


class _Resolver:
    def __init__(self, addresses: dict[str, str] | None = None) -> None:
        self.addresses = addresses or {}
        self.calls: list[tuple[str, int]] = []

    async def resolve(self, host: str, port: int) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
        self.calls.append((host, port))
        return (ipaddress.ip_address(self.addresses.get(host, "93.184.216.34")),)


class _Extraction:
    def __init__(self, *, html: str = "extracted html", pdf: str = "extracted pdf") -> None:
        self.html = html
        self.pdf = pdf
        self.html_inputs: list[str] = []
        self.pdf_inputs: list[bytes] = []

    async def extract_html(self, value: str) -> ExtractedText:
        self.html_inputs.append(value)
        return ExtractedText(self.html)

    async def extract_pdf(self, value: bytes) -> ExtractedText:
        self.pdf_inputs.append(value)
        return ExtractedText(self.pdf)


def _stream_response(
    status: int,
    content: bytes = b"",
    *,
    content_type: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    response_headers = dict(headers or {})
    if content_type is not None:
        response_headers["content-type"] = content_type
    return httpx.Response(status, headers=response_headers, stream=httpx.ByteStream(content))


def _service(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    resolver: _Resolver | None = None,
    extraction: _Extraction | None = None,
) -> tuple[WebFetchService, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    supervisor = cast(ExtractionSupervisor, extraction or _Extraction())
    return WebFetchService(client, resolver=resolver or _Resolver(), extraction_supervisor=supervisor), client


@pytest.mark.asyncio
async def test_fetches_html_with_fixed_headers_and_validated_target() -> None:
    requests: list[httpx.Request] = []
    extraction = _Extraction(html="Article in Markdown")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _stream_response(200, b"<html><body>Article</body></html>", content_type="text/html; charset=utf-8")

    service, client = _service(handler, extraction=extraction)
    async with client:
        result = await service.fetch("https://EXAMPLE.com./article?signature=secret#fragment")

    assert result.text == "Article in Markdown"
    assert result.content_kind == "html"
    assert result.final_url.display_url == "https://example.com/article"
    assert extraction.html_inputs == ["<html><body>Article</body></html>"]
    request = requests[0]
    assert request.headers["host"] == "example.com"
    assert request.headers["user-agent"] == "otari-web-retrieval/1.0"
    assert "text/html" in request.headers["accept"]
    assert isinstance(request.extensions[PINNED_TARGET_EXTENSION], ValidatedTarget)
    assert "signature=secret" in str(request.url)
    assert "fragment" not in str(request.url)


@pytest.mark.asyncio
async def test_textual_content_decodes_and_normalizes_newlines_without_worker() -> None:
    extraction = _Extraction()

    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(200, "café\r\nline".encode("latin-1"), content_type="text/plain; charset=latin-1")

    service, client = _service(handler, extraction=extraction)
    async with client:
        result = await service.fetch("https://example.com/file.txt")

    assert result.text == "café\nline"
    assert result.content_kind == "text"
    assert not extraction.html_inputs
    assert not extraction.pdf_inputs


@pytest.mark.asyncio
async def test_generic_header_can_sniff_html() -> None:
    extraction = _Extraction(html="article")

    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(200, b"<!doctype html><html>body</html>", content_type="application/octet-stream")

    service, client = _service(handler, extraction=extraction)
    async with client:
        result = await service.fetch("https://example.com/no-type")

    assert result.content_type == "text/html"
    assert result.text == "article"


@pytest.mark.asyncio
async def test_pdf_signature_is_extracted_in_worker() -> None:
    extraction = _Extraction(pdf="PDF text")
    body = b"%PDF-1.7\nbody"

    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(200, body, content_type="application/octet-stream")

    service, client = _service(handler, extraction=extraction)
    async with client:
        result = await service.fetch("https://example.com/document")

    assert result.content_kind == "pdf"
    assert result.text == "PDF text"
    assert extraction.pdf_inputs == [body]


@pytest.mark.asyncio
async def test_truncated_pdf_is_rejected_before_parser_execution() -> None:
    extraction = _Extraction(pdf="must not be returned")
    body = b"%PDF-" + b"x" * MAX_DECODED_BODY_BYTES

    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(200, body, content_type="application/pdf")

    service, client = _service(handler, extraction=extraction)
    async with client:
        with pytest.raises(ExtractionError):
            await service.fetch("https://example.com/large.pdf")

    assert not extraction.pdf_inputs


@pytest.mark.asyncio
async def test_unsupported_binary_is_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(200, b"\x89PNG\r\n", content_type="image/png")

    service, client = _service(handler)
    async with client:
        with pytest.raises(UnsupportedContentTypeError):
            await service.fetch("https://example.com/image.png")


@pytest.mark.asyncio
async def test_non_success_status_is_sanitized() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(503, b"secret response body", content_type="text/plain")

    service, client = _service(handler)
    async with client:
        with pytest.raises(WebFetchHTTPStatusError) as raised:
            await service.fetch("https://example.com/failure?token=secret")

    assert raised.value.status_code == 503
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_retrieval_telemetry_omits_urls_content_and_raw_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from gateway.services import web_fetch_service as service_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(service_module, "tracer", provider.get_tracer(service_module.__name__))

    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(502, b"private upstream detail", content_type="text/plain")

    service, client = _service(handler)
    async with client:
        with pytest.raises(WebFetchHTTPStatusError):
            await service.fetch("https://example.com/private?token=sensitive")

    span = exporter.get_finished_spans()[0]
    rendered = repr((span.attributes, span.events, span.status.description))
    assert "example.com" not in rendered
    assert "sensitive" not in rendered
    assert "private upstream detail" not in rendered
    assert not [event for event in span.events if event.name == "exception"]


@pytest.mark.asyncio
async def test_redirect_is_resolved_revalidated_and_pinned() -> None:
    requests: list[httpx.Request] = []
    resolver = _Resolver()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "example.com":
            return _stream_response(302, headers={"location": "https://docs.example.org/final?secret=yes"})
        return _stream_response(200, b"done", content_type="text/plain")

    service, client = _service(handler, resolver=resolver)
    async with client:
        result = await service.fetch("https://example.com/start")

    assert result.text == "done"
    assert result.redirect_count == 1
    assert resolver.calls == [("example.com", 443), ("docs.example.org", 443)]
    assert len(requests) == 2
    assert all(isinstance(request.extensions[PINNED_TARGET_EXTENSION], ValidatedTarget) for request in requests)


@pytest.mark.asyncio
async def test_https_redirect_downgrade_is_rejected_before_resolution() -> None:
    resolver = _Resolver()

    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(302, headers={"location": "http://downgrade.example/final"})

    service, client = _service(handler, resolver=resolver)
    async with client:
        with pytest.raises(RedirectValidationError):
            await service.fetch("https://example.com/start")

    assert resolver.calls == [("example.com", 443)]


@pytest.mark.asyncio
async def test_domain_policy_applies_to_redirect_target() -> None:
    resolver = _Resolver()
    policy = DomainPolicy(allowed=canonicalize_domain_rules(("example.com",)))

    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(302, headers={"location": "https://blocked.example/final"})

    service, client = _service(handler, resolver=resolver)
    async with client:
        with pytest.raises(RetrievalDomainPolicyError):
            await service.fetch("https://example.com/start", policy=policy)

    assert resolver.calls == [("example.com", 443)]


@pytest.mark.asyncio
async def test_search_private_host_override_still_resolves_and_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS", "true")
    resolver = _Resolver({"internal.example": "10.0.0.5"})
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _stream_response(200, b"internal", content_type="text/plain")

    service, client = _service(handler, resolver=resolver)
    async with client:
        result = await service.fetch_for_search("http://internal.example/page")

    assert result.text == "internal"
    target = cast(ValidatedTarget, requests[0].extensions[PINNED_TARGET_EXTENSION])
    assert target.addresses == (ipaddress.ip_address("10.0.0.5"),)
    assert resolver.calls == [("internal.example", 80)]


@pytest.mark.asyncio
async def test_direct_fetch_rejects_private_host_even_when_search_override_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS", "true")
    resolver = _Resolver({"internal.example": "10.0.0.5"})

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("private destination must not be contacted")

    service, client = _service(handler, resolver=resolver)
    async with client:
        with pytest.raises(RetrievalAddressError):
            await service.fetch("http://internal.example/page")
