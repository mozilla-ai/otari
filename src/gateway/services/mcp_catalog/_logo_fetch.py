"""Fetching one logo candidate from the internet, under the same rules as any other URL Otari did not write.

Every hop is https, resolved by the gateway, refused unless every address it
resolves to is public, and dialed at the address that was checked, so a name
re-pointed between the check and the connection cannot reach an internal host.
At most three redirects, each checked the same way. Five seconds and 256 KB for
the whole fetch. A site's home page may be read too, 512 KB of it, only for the
icon links in its HTML, each of which is then fetched under the same rules. The
type is read from the bytes, never from the response's headers. Nothing is
logged about a failed candidate: the next one is tried.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from gateway.services.mcp_catalog._logos import MAX_LOGO_BYTES, Logo, make_logo, sniff_image
from gateway.services.web_retrieval_network import (
    PINNED_TARGET_EXTENSION,
    AddressResolver,
    ContentDecodingError,
    NetworkDeadline,
    NetworkDeadlineExceeded,
    NoCookieJar,
    PinnedAsyncHTTPTransport,
    RedirectTracker,
    RedirectValidationError,
    RetrievalTargetError,
    TransportFactory,
    is_redirect_status,
    read_capped_decoded_body,
    validate_retrieval_target,
)

FETCH_SECONDS = 5.0
MAX_REDIRECTS = 3
# A site's home page is read only for the icons it declares in its head.
MAX_PAGE_BYTES = 512 * 1024
MAX_DECLARED_ICONS = 3
_USER_AGENT = "otari-gateway (logo fetch)"
_IMAGE_HEADERS = {
    "Accept": "image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "User-Agent": _USER_AGENT,
}
_PAGE_HEADERS = {"Accept": "text/html", "Accept-Encoding": "gzip, deflate", "User-Agent": _USER_AGENT}
_CANDIDATE_FAILURES = (
    httpx.HTTPError,
    RetrievalTargetError,
    RedirectValidationError,
    NetworkDeadlineExceeded,
    ContentDecodingError,
)

FetchLogo = Callable[[str], Awaitable[Logo | None]]


class LogoFetcher:
    """Fetches logo candidates. One per resolution: it holds a connection pool, closed by ``aclose``."""

    def __init__(
        self, *, resolver: AddressResolver | None = None, transport_factory: TransportFactory | None = None
    ) -> None:
        self._resolver = resolver
        self._client = httpx.AsyncClient(
            transport=PinnedAsyncHTTPTransport(transport_factory=transport_factory, max_pools=8),
            follow_redirects=False,
            cookies=NoCookieJar(),
            timeout=httpx.Timeout(FETCH_SECONDS),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch(self, url: str) -> Logo | None:
        """The image at ``url`` if it is one the gateway will serve, else None. Never raises for a bad candidate.

        A site's root (``https://example.com/``) is a page, not an image: the icons its HTML
        declares are tried instead, at most ``MAX_DECLARED_ICONS`` of them, each fetched the same way.
        """
        if urlsplit(url).path == "/":
            try:
                return await self._declared(url)
            except _CANDIDATE_FAILURES:
                return None
        return await self._image(url)

    async def _image(self, url: str) -> Logo | None:
        """The image at ``url``, or None. Never read as a page, so a declared icon cannot start another search."""
        try:
            body = await self._get(url, NetworkDeadline(FETCH_SECONDS), MAX_LOGO_BYTES, _IMAGE_HEADERS)
        except _CANDIDATE_FAILURES:
            return None
        if body is None:
            return None
        content_type = sniff_image(body[0])
        return make_logo(body[0], content_type) if content_type else None

    async def _declared(self, page_url: str) -> Logo | None:
        page = await self._get(page_url, NetworkDeadline(FETCH_SECONDS), MAX_PAGE_BYTES, _PAGE_HEADERS)
        if page is None:
            return None
        content, final_url = page
        # Off the event loop: the page is the site's to write, and half a megabyte of it takes a moment to read.
        icons = await asyncio.to_thread(declared_icons, content.decode("utf-8", "replace"), final_url)
        for icon_url in icons[:MAX_DECLARED_ICONS]:
            found = await self._image(icon_url)
            if found is not None:
                return found
        return None

    async def _get(
        self, url: str, deadline: NetworkDeadline, max_bytes: int, headers: dict[str, str]
    ) -> tuple[bytes, str] | None:
        """The body at ``url`` and the URL it came from after redirects, or None. Raises on a refused hop."""
        target = await validate_retrieval_target(url, resolver=self._resolver, deadline=deadline)
        tracker = RedirectTracker(target.canonical_url, max_redirects=MAX_REDIRECTS)
        while True:
            if target.origin.scheme != "https":
                return None
            request = self._client.build_request(
                "GET",
                target.url,
                headers={**headers, "Host": target.origin.authority},
                extensions={PINNED_TARGET_EXTENSION: target},
            )
            response = await deadline.run(lambda: self._client.send(request, stream=True, follow_redirects=False))
            try:
                if is_redirect_status(response.status_code):
                    location = response.headers.get("location")
                    await response.aclose()
                    target = await validate_retrieval_target(
                        tracker.advance(location), resolver=self._resolver, deadline=deadline
                    )
                    continue
                if response.status_code != 200:
                    return None
                body = await read_capped_decoded_body(response, deadline=deadline, max_bytes=max_bytes)
            finally:
                await response.aclose()
            if body.truncated or not body.content:
                return None
            return body.content, str(target.url)


class _LinkTags(HTMLParser):
    """The attributes of each ``<link>`` tag on a page, read in one pass, so no page can make the scan backtrack."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "link":
            self.links.append({name: value or "" for name, value in attrs})


def declared_icons(html: str, base_url: str) -> list[str]:
    """The https icon URLs a page's ``<link rel=...icon...>`` tags name, touch icons first, then the larger."""
    parser = _LinkTags()
    parser.feed(html[:MAX_PAGE_BYTES])
    parser.close()
    found: list[tuple[tuple[int, int], str]] = []
    for attributes in parser.links:
        rel = attributes.get("rel", "").lower().split()
        href = attributes.get("href", "").strip()
        if "icon" not in rel and "apple-touch-icon" not in rel or not href or href.startswith("data:"):
            continue
        absolute = urljoin(base_url, href)
        if not absolute.startswith("https://"):
            continue
        sizes = re.findall(r"(\d{1,4})x\d{1,4}", attributes.get("sizes", ""))
        largest = max((int(size) for size in sizes), default=0)
        found.append(((0 if "apple-touch-icon" in rel else 1, -largest), absolute))
    return list(dict.fromkeys(url for _, url in sorted(found)))
