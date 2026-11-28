"""Firecrawl scrape: ``POST /v2/scrape``, with the key as a Bearer token.

One page per call, as Markdown. Sends the URL and the native options passed,
with ``formats`` always ``["markdown"]``, since the Markdown is the text. ``None``
counts as not passed. ``parsers`` is ``[{"type": "pdf", "maxPages": 100}, "image"]``
unless passed: Firecrawl's own default, ``["pdf", "image"]``, parses every page of
a PDF at 1 Firecrawl credit each with no ceiling, while the cap still returns
the text of the first 100 pages. Firecrawl has no length cap, so ``max_chars``
cuts the text. Firecrawl renders a page's scripts
in a browser when the page needs them
(https://docs.firecrawl.dev/features/scrape: "js-rendered sites").

``final_url`` is ``metadata.url``, where the page was found after redirects, and
``content_type`` is ``metadata.contentType``. ``title`` is ``metadata.title``, the
first one when Firecrawl sends a list. ``published`` is ``None``: Firecrawl
passes on the page's own meta tags, in whatever form the page wrote them, and
they stay in ``raw``. Firecrawl reports ``creditsUsed``, Firecrawl credits
rather than money, so it reports no cost.

401, 402 and 429 are tagged by status (``unauthorized``, ``payment_required``,
``rate_limited``): their bodies are a sentence. Any other status is tagged with
the body's ``code`` when it carries one, as seen live (``SCRAPE_ALL_ENGINES_FAILED``, a 500,
for a page Firecrawl could not read, such as a zero-byte file, HTML with an
empty body or a private address; ``SCRAPE_TIMEOUT``, a 408), else
``http_error``. The ``error`` sentence, which can quote the URL, is not read.
``success: false`` inside a 200 is ``FetchedPage.error``, tagged with its
``code`` (``SCRAPE_DNS_RESOLUTION_ERROR`` for a host that does not resolve), else
``unsuccessful``. A site's own error page, such as a 404, comes back as that
page's text, with the site's status in ``metadata.statusCode``; one with an
empty body is an empty page.

``success`` must be a boolean. A successful answer must hold
``data.markdown`` as a string and ``data.metadata.url`` as a non-empty string,
since ``final_url`` is what a caller checks a redirect against; ``metadata.title``
must be a string or a list of strings and ``metadata.contentType`` a string when
present. Anything else raises ``unexpected_response``. Unread fields are ignored.

API: https://docs.firecrawl.dev/api-reference/endpoint/scrape. Pricing, read
2026-11-25, https://docs.firecrawl.dev/billing and https://www.firecrawl.dev/pricing:
1 Firecrawl credit a page, a site's error page included, and none when
Firecrawl returns no page; a PDF adds 1 Firecrawl credit per page parsed. A
Firecrawl credit costs from $0.0006 (Scale, billed yearly) to $0.005 (Hobby's
extra credits).
"""

from typing import Any

import httpx

from any_fetch._api import AnyFetch
from any_fetch._errors import ProviderError, safe_tag
from any_fetch._types import FetchedPage, FetchError, OptionSpec, ProviderMetadata

DEFAULT_PARSERS: list[Any] = [{"type": "pdf", "maxPages": 100}, "image"]
"""The ``parsers`` sent when none is passed: Firecrawl's own, with a ceiling on the PDF pages parsed and billed."""

# Native options copied into the request as they are.
_PASSED_THROUGH = (
    "onlyMainContent",
    "includeTags",
    "excludeTags",
    "maxAge",
    "waitFor",
    "mobile",
    "timeout",
    "location",
    "removeBase64Images",
    "blockAds",
    "storeInCache",
)

# The statuses whose body is a sentence, not a code.
_STATUS_TAGS = {401: "unauthorized", 402: "payment_required", 429: "rate_limited"}


class FirecrawlProvider(AnyFetch):
    """Firecrawl's Scrape API."""

    METADATA = ProviderMetadata(
        name="firecrawl",
        doc_url="https://docs.firecrawl.dev/api-reference/endpoint/scrape",
        env_key="FIRECRAWL_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.firecrawl.dev",
        tier="production",
        max_urls_per_call=1,
        # Firecrawl documents scraping "js-rendered sites" (https://docs.firecrawl.dev/features/scrape).
        renders_javascript=True,
        # Firecrawl answers with metadata.url, the page's URL after redirects (checked 2026-11-25).
        reports_final_url=True,
        formats=["markdown"],
        options=[
            OptionSpec(
                name="onlyMainContent",
                type="boolean",
                default=True,
                description="Leave out headers, navigation and footers.",
            ),
            OptionSpec(
                name="includeTags",
                type="array",
                description="HTML tags, classes or ids to keep, such as ['article', '.content'].",
            ),
            OptionSpec(
                name="excludeTags",
                type="array",
                description="HTML tags, classes or ids to leave out, such as ['nav', '#ads'].",
            ),
            OptionSpec(
                name="maxAge",
                type="integer",
                default=172800000,
                description=(
                    "Milliseconds a cached copy may be old, two days unless passed, seven at most; 0 always scrapes "
                    "the page as it is now."
                ),
            ),
            OptionSpec(
                name="waitFor",
                type="integer",
                default=0,
                description="Milliseconds to wait for the page to load before reading it, at most 60000.",
            ),
            OptionSpec(
                name="mobile",
                type="boolean",
                default=False,
                description="Read the page as a mobile device would.",
            ),
            OptionSpec(
                name="timeout",
                type="integer",
                default=60000,
                description=(
                    "Milliseconds Firecrawl may spend, 1000 to 300000. Give the fetcher a longer timeout: afetch's "
                    "timeout argument is the fetcher's, in seconds, so this option goes through fetch."
                ),
            ),
            OptionSpec(
                name="location",
                type="object",
                description='Where to read the page from, as {"country": "DE", "languages": ["de"]}.',
            ),
            OptionSpec(
                name="removeBase64Images",
                type="boolean",
                default=True,
                description="Replace inline base64 images in the Markdown with a placeholder, keeping their alt text.",
            ),
            OptionSpec(
                name="blockAds",
                type="boolean",
                default=True,
                description="Block ads and cookie pop-ups.",
            ),
            OptionSpec(
                name="storeInCache",
                type="boolean",
                default=True,
                description="Keep the page in Firecrawl's cache.",
            ),
            OptionSpec(
                name="parsers",
                type="array",
                default=DEFAULT_PARSERS,
                description=(
                    "How PDFs and images are read. A PDF adds 1 Firecrawl credit per page parsed, so the default "
                    "parses the first 100 pages; a higher maxPages, or a pdf parser without one, reads and bills "
                    "more. A list turns off the parsers it leaves out."
                ),
            ),
        ],
    )

    async def _fetch(self, url: str, *, max_chars: int | None, options: dict[str, Any]) -> FetchedPage:
        payload: dict[str, Any] = {"url": url, "formats": ["markdown"]}
        payload.update({name: options[name] for name in _PASSED_THROUGH if options.get(name) is not None})
        payload["parsers"] = DEFAULT_PARSERS if options.get("parsers") is None else options["parsers"]

        endpoint = f"{(self._api_base or '').rstrip('/')}/v2/scrape"
        headers = {"Authorization": f"Bearer {self._api_key or ''}"}
        response = await self._http.request("POST", endpoint, json=payload, headers=headers)
        body = _body(response)
        success = body.get("success")
        if not isinstance(success, bool):
            raise _unexpected(response)
        if not success:
            return FetchedPage(
                url=url,
                final_url=url,
                text="",
                content_type="",
                cost_source="none",
                error=FetchError(tag=safe_tag(body.get("code")) or "unsuccessful"),
                raw=body,
            )
        data = body.get("data")
        metadata = data.get("metadata") if isinstance(data, dict) else None
        if not isinstance(data, dict) or not isinstance(metadata, dict):
            raise _unexpected(response)
        text, final_url = data.get("markdown"), metadata.get("url")
        content_type = metadata.get("contentType", "")
        if not isinstance(text, str) or not isinstance(final_url, str) or not final_url:
            raise _unexpected(response)
        if not isinstance(content_type, str):
            raise _unexpected(response)
        return FetchedPage(
            url=url,
            final_url=final_url,
            title=_title(metadata.get("title", ""), response),
            text=text[:max_chars],
            content_type=content_type,
            cost_source="none",
            text_truncated=max_chars is not None and len(text) > max_chars,
            raw=body,
        )


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Firecrawl answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    status = response.status_code
    if status >= 400:
        tag = _STATUS_TAGS.get(status) or (safe_tag(body.get("code")) if isinstance(body, dict) else None)
        raise ProviderError(FirecrawlProvider.METADATA.name, status, tag or "http_error")
    if not isinstance(body, dict):
        raise ProviderError(FirecrawlProvider.METADATA.name, status, "invalid_response")
    return body


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field the adapter needs, or carries one of the wrong type."""
    return ProviderError(FirecrawlProvider.METADATA.name, response.status_code, "unexpected_response")


def _title(title: Any, response: httpx.Response) -> str:
    """``metadata.title``, which Firecrawl documents as a string or a list of strings: the first, for a list."""
    if isinstance(title, str):
        return title
    if not isinstance(title, list) or not all(isinstance(part, str) for part in title):
        raise _unexpected(response)
    return title[0] if title else ""
