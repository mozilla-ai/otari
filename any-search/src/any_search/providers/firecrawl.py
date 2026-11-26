"""Firecrawl: ``POST /v2/search``, with the key as a Bearer token.

Sends the query and the native options passed, as Firecrawl's SDK
(``firecrawl-py`` 4.57.0) does. ``None`` is not sent. ``max_results`` becomes
``limit``, at most 100, sent only when passed, as the SDK does; Firecrawl's
own default is 10.
``time_range`` becomes ``tbs`` (``qdr:d``, ``qdr:w``, ``qdr:m``, ``qdr:y``) and
wins over that option. Firecrawl's ``timeout`` is 60 seconds unless passed.

Each result's ``description`` is the snippet: query-relevant highlights by
default, the site's own description under ``highlights=False``. ``markdown``,
present only when ``scrapeOptions`` scrapes the results, is the text, so
``markdown`` is added to ``scrapeOptions.formats`` when they lack it, and they
are ``["markdown"]`` when missing or empty: Firecrawl scrapes nothing without
a format. ``scrapeOptions`` is not a cost tier, so a request can set it; each
page it scrapes adds to the cost of the search. Firecrawl dates no web result,
so ``published`` is ``None``. ``sources`` is not offered: the adapter reads
web results, while news, images and Firecrawl Alexandria's data tools come back
in fields of their own.

401, 402 and 429 are tagged by status (``unauthorized``, ``payment_required``,
``rate_limited``): their bodies are a sentence. Any other status is tagged with
the body's ``code`` when it carries one, else ``http_error``; the codes are
Firecrawl's, seen live, such as ``CATEGORY_REMOVED``. The ``error``
sentence, which can quote the request, is not read. ``success: false`` inside a
200 is ``SearchResult.error``, tagged with its ``code``, else ``unsuccessful``.
That path is unobserved: every failure seen live came with a status of 400 or
more.

``success`` must be a boolean, and ``data`` an object. ``data.web``
must be a list when present; without it there are no hits. Each item read must
have ``url`` as a string, ``title`` and ``description`` as strings when present,
and ``markdown`` as a string or null when present. A result without an http or
https URL is dropped. Anything else raises ``unexpected_response``. Unread
fields are ignored.

API: https://docs.firecrawl.dev/api-reference/endpoint/search. Pricing, read
2026-11-25, https://docs.firecrawl.dev/billing and https://www.firecrawl.dev/pricing:
2 Firecrawl credits per 10 results, rounded up per 10; each result
``scrapeOptions`` scrapes adds 1 Firecrawl credit plus the scrape options' own.
A Firecrawl credit costs from $0.0006 (Scale, billed yearly) to $0.005
(Hobby's extra credits). Firecrawl reports
``creditsUsed``, which stays in ``raw``: it counts Firecrawl credits, not money.
"""

from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError, safe_tag
from any_search._http import is_web_url
from any_search._types import OptionSpec, ProviderMetadata, SearchError, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 100
"""The most results Firecrawl returns for one search."""

DEFAULT_LIMIT = 10
"""The ``limit`` Firecrawl applies when none is sent, as the SDK sends none."""

# Native options copied into the request as they are; time_range wins over tbs.
_PASSED_THROUGH = (
    "tbs",
    "categories",
    "includeDomains",
    "excludeDomains",
    "location",
    "country",
    "safe",
    "highlights",
    "ignoreInvalidURLs",
    "timeout",
    "scrapeOptions",
)

_TIME_RANGES: dict[TimeRange, str] = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}

# The statuses whose body is a sentence, not a code.
_STATUS_TAGS = {401: "unauthorized", 402: "payment_required", 429: "rate_limited"}


class FirecrawlProvider(AnySearch):
    """Firecrawl's Search API."""

    METADATA = ProviderMetadata(
        name="firecrawl",
        doc_url="https://docs.firecrawl.dev/api-reference/endpoint/search",
        env_key="FIRECRAWL_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.firecrawl.dev",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=False,
        key_in_url=False,
        options=[
            OptionSpec(
                name="tbs",
                type="string",
                description=(
                    "Time filter: qdr:h, qdr:d, qdr:w, qdr:m or qdr:y, a range as "
                    "cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY, and sbd:1 to sort by date, joined by commas. "
                    "time_range wins. Firecrawl answers a value it does not take as if none was set."
                ),
            ),
            OptionSpec(
                name="categories",
                type="array",
                description='Categories to search, such as ["research", "pdf"]. developer and gov must be alone.',
            ),
            OptionSpec(
                name="includeDomains",
                type="array",
                description="Hostnames to limit results to, without scheme or path. Refused with excludeDomains.",
            ),
            OptionSpec(
                name="excludeDomains",
                type="array",
                description="Hostnames to leave out, without scheme or path. Refused with includeDomains.",
            ),
            OptionSpec(
                name="location",
                type="string",
                description="Where to search from, such as 'San Francisco,California,United States'. Set country too.",
            ),
            OptionSpec(
                name="country",
                type="string",
                default="US",
                description="Country to search from, a two-letter ISO code.",
            ),
            OptionSpec(
                name="safe",
                type="boolean",
                description="Filter explicit content out of the results.",
            ),
            OptionSpec(
                name="highlights",
                type="boolean",
                default=True,
                description=(
                    "Query-relevant passages of each page as its description, the snippet; false returns the "
                    "site's own description, which is shorter."
                ),
            ),
            OptionSpec(
                name="ignoreInvalidURLs",
                type="boolean",
                default=False,
                description="Leave out results whose URL Firecrawl's other endpoints would refuse.",
            ),
            OptionSpec(
                name="timeout",
                type="integer",
                default=60000,
                description=(
                    "Milliseconds Firecrawl may spend. Give the engine a longer timeout: AnySearch's own timeout "
                    "argument is in seconds and ends the call on this side."
                ),
            ),
            OptionSpec(
                name="scrapeOptions",
                type="object",
                description=(
                    'Scrape each result too, as {"formats": ["markdown"]}; its Markdown is the text, and markdown '
                    "is always among the formats. Each page scraped adds 1 Firecrawl credit, plus the options' own."
                ),
            ),
        ],
    )

    async def _search(
        self,
        query: str,
        *,
        max_results: int | None,
        time_range: TimeRange | None,
        options: dict[str, Any],
    ) -> SearchResult:
        limit = min(DEFAULT_LIMIT if max_results is None else max_results, MAX_RESULTS)
        payload: dict[str, Any] = {"query": query}
        if max_results is not None:
            payload["limit"] = limit
        payload.update({name: options[name] for name in _PASSED_THROUGH if options.get(name) is not None})
        if time_range is not None:
            payload["tbs"] = _TIME_RANGES[time_range]
        if "scrapeOptions" in payload:
            payload["scrapeOptions"] = _with_markdown(payload["scrapeOptions"])

        endpoint = f"{(self._api_base or '').rstrip('/')}/v2/search"
        headers = {"Authorization": f"Bearer {self._api_key or ''}"}
        response = await self._http.request("POST", endpoint, json=payload, headers=headers)
        body = _body(response)
        success = body.get("success")
        if not isinstance(success, bool):
            raise _unexpected(response)
        if not success:
            error = SearchError(tag=safe_tag(body.get("code")) or "unsuccessful")
            return SearchResult(provider=self.METADATA.name, hits=[], cost_source="none", error=error, raw=body)
        data = body.get("data")
        if not isinstance(data, dict):
            raise _unexpected(response)
        web = data.get("web", [])
        if not isinstance(web, list):
            raise _unexpected(response)
        # Only the items read until the count is reached are checked, as only they reach the caller.
        hits: list[SearchHit] = []
        for item in web:
            if len(hits) == limit:
                break
            hit = _hit(item, response)
            if hit is not None:
                hits.append(hit)
        return SearchResult(provider=self.METADATA.name, hits=hits, cost_source="none", raw=body)


def _with_markdown(scrape_options: Any) -> dict[str, Any]:
    """``scrapeOptions`` as given, with ``markdown`` among its formats, since the text is the Markdown.

    Missing or empty formats become ``["markdown"]``: under search, Firecrawl scrapes nothing without a format.
    """
    if not isinstance(scrape_options, dict):
        raise ProviderError(FirecrawlProvider.METADATA.name, None, "invalid_option")
    formats = scrape_options.get("formats") or []
    if not isinstance(formats, list) or any(_is_markdown(entry) for entry in formats):
        return dict(scrape_options)
    return {**scrape_options, "formats": ["markdown", *formats]}


def _is_markdown(entry: Any) -> bool:
    """Whether a format names Markdown, given by name or as an object with its type."""
    return entry == "markdown" or (isinstance(entry, dict) and entry.get("type") == "markdown")


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


def _hit(item: Any, response: httpx.Response) -> SearchHit | None:
    """One web result as a hit, or ``None`` for one without a URL a caller can fetch."""
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, description = item.get("url"), item.get("title", ""), item.get("description", "")
    markdown = item.get("markdown")
    if (
        not isinstance(url, str)
        or not isinstance(title, str)
        or not isinstance(description, str)
        or not (markdown is None or isinstance(markdown, str))
    ):
        raise _unexpected(response)
    if not is_web_url(url):
        return None
    return SearchHit(url=url, title=title, snippet=description, text=markdown or None, raw=item)
