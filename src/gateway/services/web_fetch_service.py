"""Shared bounded retrieval and extraction for gateway web tools."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import httpx
from opentelemetry import trace

from gateway.core.config import parse_bool_env
from gateway.core.env import otari_env
from gateway.services.web_extraction import ExtractionError, ExtractionSupervisor, get_extraction_supervisor
from gateway.services.web_retrieval_network import (
    PINNED_TARGET_EXTENSION,
    AddressResolver,
    CappedBody,
    NetworkDeadline,
    NetworkDeadlineExceeded,
    RedirectTracker,
    RedirectValidationError,
    RetrievalAddressError,
    RetrievalDomainPolicyError,
    RetrievalTargetError,
    is_redirect_status,
    read_capped_decoded_body,
    validate_retrieval_target,
)
from gateway.services.web_retrieval_policy import CanonicalWebURL, DomainPolicy

if TYPE_CHECKING:
    from opentelemetry.trace import Span

ContentKind = Literal["html", "text", "pdf"]

WEB_RETRIEVAL_ACCEPT = (
    "text/html,application/xhtml+xml,text/plain,text/markdown,application/pdf,"
    "application/json,application/xml;q=0.9,*/*;q=0.1"
)
WEB_RETRIEVAL_USER_AGENT = "otari-web-retrieval/1.0"
WEB_RETRIEVAL_ACCEPT_ENCODING = "gzip, deflate"

_HTML_MIME_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_PDF_MIME_TYPE = "application/pdf"
_TEXT_MIME_TYPES = frozenset(
    {
        "application/json",
        "application/ld+json",
        "application/markdown",
        "application/xml",
        "application/javascript",
        "application/x-javascript",
        "application/ecmascript",
        "application/x-ndjson",
    }
)
_GENERIC_MIME_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream"})

tracer = trace.get_tracer(__name__)


class WebFetchError(RuntimeError):
    """A retrieval failed without exposing destination or parser details."""


class UnsupportedContentTypeError(WebFetchError):
    """The response body is not one of the supported textual formats."""


class WebFetchHTTPStatusError(WebFetchError):
    """The destination returned a non-success HTTP response."""

    def __init__(self, status_code: int) -> None:
        super().__init__("destination returned a non-success status")
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class WebFetchResult:
    """Bounded extracted content and privacy-safe retrieval metadata."""

    text: str
    content_type: str
    content_kind: ContentKind
    requested_url: CanonicalWebURL
    final_url: CanonicalWebURL
    redirect_count: int
    source_truncated: bool = False
    extraction_truncated: bool = False


def _search_allows_private_hosts() -> bool:
    return parse_bool_env(otari_env("WEB_SEARCH_ALLOW_PRIVATE_HOSTS", "false"))


def _normalized_content_type(response: httpx.Response) -> str:
    return str(response.headers.get("content-type", "")).split(";", 1)[0].strip().lower()


def _sniff_content_kind(content: bytes) -> ContentKind | None:
    leading = content[:512].lstrip()
    lowered = leading.lower()
    if leading.startswith(b"%PDF-"):
        return "pdf"
    if lowered.startswith((b"<!doctype html", b"<html", b"<?xml")) and b"html" in lowered[:128]:
        return "html"
    return None


def classify_content(response: httpx.Response, content: bytes) -> tuple[ContentKind, str]:
    """Classify a bounded response without treating arbitrary binary as text."""
    mime_type = _normalized_content_type(response)
    sniffed = _sniff_content_kind(content)
    if sniffed == "pdf":
        return "pdf", _PDF_MIME_TYPE
    if mime_type == _PDF_MIME_TYPE:
        return "pdf", mime_type
    if mime_type in _HTML_MIME_TYPES:
        return "html", mime_type
    if mime_type.startswith("text/"):
        return "text", mime_type
    if mime_type in _TEXT_MIME_TYPES or mime_type.endswith(("+json", "+xml")):
        return "text", mime_type
    if mime_type in _GENERIC_MIME_TYPES and sniffed == "html":
        return "html", "text/html"
    raise UnsupportedContentTypeError("response content type is unsupported")


def _response_encoding(response: httpx.Response) -> str:
    encoding = response.encoding or "utf-8"
    try:
        codecs.lookup(encoding)
    except LookupError:
        return "utf-8"
    return str(encoding)


def _decode_text(response: httpx.Response, content: bytes) -> str:
    return content.decode(_response_encoding(response), errors="replace").replace("\r\n", "\n")


class WebFetchService:
    """Retrieve one URL through a pinned client and return bounded text."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        resolver: AddressResolver | None = None,
        extraction_supervisor: ExtractionSupervisor | None = None,
    ) -> None:
        self._client = client
        self._resolver = resolver
        self._extraction_supervisor = extraction_supervisor

    async def fetch_for_search(
        self,
        url: str,
        *,
        policy: DomainPolicy | None = None,
    ) -> WebFetchResult:
        """Retrieve a Search result, honoring its private-host compatibility flag."""
        return await self.fetch(url, policy=policy, allow_private=_search_allows_private_hosts())

    async def fetch(
        self,
        url: str,
        *,
        policy: DomainPolicy | None = None,
        allow_private: bool = False,
    ) -> WebFetchResult:
        """Retrieve, classify, and extract one URL under fixed network bounds."""
        deadline = NetworkDeadline()
        with tracer.start_as_current_span(
            "web_retrieval",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attribute("tool.type", "web_retrieval")
            try:
                result = await self._fetch(
                    url,
                    policy=policy,
                    allow_private=allow_private,
                    deadline=deadline,
                )
            except BaseException as exc:
                self._record_failure(span, exc)
                raise
            span.set_attribute("web_retrieval.outcome", "success")
            span.set_attribute("web_retrieval.content_kind", result.content_kind)
            span.set_attribute("web_retrieval.content_bytes", len(result.text.encode("utf-8")))
            span.set_attribute("web_retrieval.redirect_count", result.redirect_count)
            span.set_attribute("web_retrieval.source_truncated", result.source_truncated)
            span.set_attribute("web_retrieval.extraction_truncated", result.extraction_truncated)
            return result

    @staticmethod
    def _record_failure(span: Span, exc: BaseException) -> None:
        if isinstance(exc, RetrievalDomainPolicyError):
            outcome = "domain_denied"
        elif isinstance(exc, (RetrievalTargetError, RetrievalAddressError)):
            outcome = "destination_denied"
        elif isinstance(exc, RedirectValidationError):
            outcome = "redirect_denied"
        elif isinstance(exc, (NetworkDeadlineExceeded, httpx.TimeoutException)):
            outcome = "timeout"
        elif isinstance(exc, WebFetchHTTPStatusError):
            outcome = "http_status"
            span.set_attribute("web_retrieval.http_status", exc.status_code)
        elif isinstance(exc, UnsupportedContentTypeError):
            outcome = "unsupported_content"
        elif isinstance(exc, ExtractionError):
            outcome = "extraction_failed"
        else:
            outcome = "network_failed"
        span.set_attribute("web_retrieval.outcome", outcome)
        span.set_status(trace.StatusCode.ERROR, outcome)

    async def _fetch(
        self,
        url: str,
        *,
        policy: DomainPolicy | None,
        allow_private: bool,
        deadline: NetworkDeadline,
    ) -> WebFetchResult:
        target = await validate_retrieval_target(
            url,
            policy=policy,
            resolver=self._resolver,
            deadline=deadline,
            allow_private=allow_private,
        )
        requested_url = target.canonical_url
        tracker = RedirectTracker(requested_url)

        while True:
            request = self._client.build_request(
                "GET",
                target.url,
                headers={
                    "Accept": WEB_RETRIEVAL_ACCEPT,
                    "Accept-Encoding": WEB_RETRIEVAL_ACCEPT_ENCODING,
                    "User-Agent": WEB_RETRIEVAL_USER_AGENT,
                    "Host": target.origin.authority,
                },
                extensions={PINNED_TARGET_EXTENSION: target},
            )
            response = await deadline.run(
                lambda: self._client.send(request, stream=True, follow_redirects=False)
            )
            try:
                if is_redirect_status(response.status_code):
                    location = response.headers.get("location")
                    await response.aclose()
                    next_url = tracker.advance(location)
                    target = await validate_retrieval_target(
                        next_url,
                        policy=policy,
                        resolver=self._resolver,
                        deadline=deadline,
                        allow_private=allow_private,
                    )
                    continue
                if not 200 <= response.status_code < 300:
                    raise WebFetchHTTPStatusError(response.status_code)

                body = await read_capped_decoded_body(response, deadline=deadline)
                return await self._extract_body(
                    response,
                    body,
                    requested_url=requested_url,
                    final_url=target.canonical_url,
                    redirect_count=tracker.redirects_followed,
                )
            finally:
                await response.aclose()

    async def _extract_body(
        self,
        response: httpx.Response,
        body: CappedBody,
        *,
        requested_url: CanonicalWebURL,
        final_url: CanonicalWebURL,
        redirect_count: int,
    ) -> WebFetchResult:
        kind, content_type = classify_content(response, body.content)
        extraction_truncated = False
        if kind == "pdf":
            if body.truncated:
                raise ExtractionError("truncated PDF cannot be extracted")
            supervisor = self._extraction_supervisor or get_extraction_supervisor()
            extracted = await supervisor.extract_pdf(body.content)
            text = extracted.text
            extraction_truncated = extracted.truncated
        else:
            decoded = _decode_text(response, body.content)
            if kind == "html":
                supervisor = self._extraction_supervisor or get_extraction_supervisor()
                extracted = await supervisor.extract_html(decoded)
                text = extracted.text
                extraction_truncated = extracted.truncated
            else:
                text = decoded.strip()
        if not text:
            raise ExtractionError("content has no extractable text")
        return WebFetchResult(
            text=text,
            content_type=content_type,
            content_kind=kind,
            requested_url=requested_url,
            final_url=final_url,
            redirect_count=redirect_count,
            source_truncated=body.truncated,
            extraction_truncated=extraction_truncated,
        )
