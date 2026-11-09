"""You.com: ``POST /v1/search``, with the key in the ``X-API-Key`` header.

The adapter sends what You.com's Python SDK (``youdotcom`` 3.6.0) sends for a
search: the query, ``count``, ``language`` ``EN``, ``knowledge`` ``core`` and
``extraction`` with ``extraction_mode`` ``highlights``, and the native options
passed. An option passed as ``None`` is not sent, so ``knowledge=None`` and
``extraction=None`` turn those defaults off, as they do in the SDK. ``count``
is ``max_results``, 10 when it is not passed, at most 100, and at most 50 under
highlights, which You.com refuses above that with a 422. Above 100 it answers
no results rather than refusing. ``time_range`` becomes ``freshness``, which
takes You.com's own ``day`` to ``year``, and wins over that option.

Each web result's ``contents.highlights``, the passages of the page that
address the query, joined, become the hit's snippet; a result without
highlights has its ``description`` as the snippet. ``contents.markdown``, which
comes back only with ``extraction_mode`` ``full_page``, is the hit's text.
``page_age`` is the hit's date when it is ISO 8601, taken as UTC since it
carries no zone. You.com documents no format for it, and its SDK has met
``7/29/2024 10:38:56 AM`` and ``171 days ago``, so a value in another form
stays in ``raw``. The ``knowledge`` results, which carry no URL, the
``snippets`` of a search without highlights and ``contents.html`` stay in
``raw`` too.

You.com signals every failure with an HTTP status. A refused key is a 403
with ``{"message": "Forbidden"}``, answered by the gateway in front of the API
rather than the 401 the documentation lists, and a 422's ``error`` is a
sentence that can quote the request, so the tag is the adapter's own, by
status: ``unauthorized`` for a 401, ``forbidden`` for a 403, which a key
without the scope gets too, and ``invalid_request`` for a 422. A 402's
``error`` is a code, such as ``payment_required`` for an account out of
credit, and becomes the tag. Any other status is ``http_error``. A successful
search carries no error of its own. Checked live 2026-11-09: You.com does not
refuse a ``freshness`` or a ``country`` it does not take, or an ``offset`` of
10, and answers ``count`` 0 or 101 with no results.

Every field read from a successful answer is checked, so a change on You.com's
side fails the call: ``results`` must be an object, and its ``web``, which
You.com leaves out of a search with no results, a list when present. Each item
read before the count is reached must be an object whose ``url`` is a string
or null, whose ``title`` and ``description`` are strings, whose ``page_age``
is a string or null, and whose ``contents`` is an object with ``highlights``
a list of strings and ``markdown`` a string, each when present; a
``page_age`` that starts like an ISO 8601 date and does not parse is refused
too. Anything else raises ``unexpected_response``. You.com's schema makes
every field of a result optional, so a result without a ``url`` that is an
http or https URL with a host is dropped, and the count is filled from the
items after it. Fields the adapter does not read are ignored, so a field
You.com adds breaks nothing. You.com reports no cost.

API: https://you.com/docs/api-reference/search/v1-search and its OpenAPI
spec, https://you.com/docs/openapi/web-search.json. Pricing, read 2026-11-09,
https://you.com/docs/administration/billing: $5.00 per 1,000 calls, up to 100
results each, highlights and knowledge results included. Full-page extraction
adds $1.00 per 1,000 pages crawled live: with ``extraction_source`` ``blend``,
the default, only the pages not in You.com's cache, with ``cache`` none, with
``fetch`` every page. So a full-page search for 10 results costs from $0.005
to $0.015. Volume discounts by arrangement.
"""

import re
from datetime import UTC, datetime
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError, safe_tag
from any_search._http import is_web_url
from any_search._types import OptionSpec, ProviderMetadata, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 100
"""The most results You.com returns for one search."""

HIGHLIGHTS_MAX_RESULTS = 50
"""The most results You.com takes ``count`` for under highlights, the default."""

DEFAULT_COUNT = 10
"""The ``count`` You.com's SDK sends when none is passed, and You.com's own default."""

# Native options copied into the request as they are; time_range wins over freshness.
_PASSED_THROUGH = (
    "freshness",
    "offset",
    "country",
    "language",
    "safesearch",
    "knowledge",
    "include_domains",
    "exclude_domains",
    "boost_domains",
    "strict_operators",
    "extraction",
    "crawl_timeout",
)

# The tags of the statuses whose body is a sentence or the gateway's; a 402's comes from its body.
_STATUS_TAGS = {401: "unauthorized", 403: "forbidden", 422: "invalid_request"}

# How an ISO 8601 date starts, to tell a broken one from a date in another form.
_ISO_DATE_START = re.compile(r"\d{4}-\d{2}-\d{2}")

_COUNTRIES = [
    "AR", "AU", "AT", "BE", "BR", "CA", "CL", "DK", "FI", "FR", "DE", "HK", "IN", "ID", "IT", "JP", "KR", "MY",
    "MX", "NL", "NZ", "NO", "CN", "PL", "PT", "PH", "RU", "SA", "ZA", "ES", "SE", "CH", "TW", "TR", "GB", "US",
]  # fmt: skip

_LANGUAGES = [
    "AR", "EU", "BN", "BG", "CA", "ZH-HANS", "ZH-HANT", "HR", "CS", "DA", "NL", "EN", "EN-GB", "ET", "FI", "FR",
    "GL", "DE", "EL", "GU", "HE", "HI", "HU", "IS", "IT", "JA", "KN", "KO", "LV", "LT", "MS", "ML", "MR", "NB",
    "PL", "PT-BR", "PT-PT", "PA", "RO", "RU", "SR", "SK", "SL", "ES", "SV", "TA", "TE", "TH", "TR", "UK", "VI",
]  # fmt: skip


def _defaults() -> dict[str, Any]:
    """What You.com's SDK sends unless told otherwise, built per request so no caller shares the object."""
    return {"language": "EN", "knowledge": "core", "extraction": {"extraction_mode": "highlights"}}


class YouDotComProvider(AnySearch):
    """You.com's Web Search API."""

    METADATA = ProviderMetadata(
        name="youdotcom",
        doc_url="https://you.com/docs/api-reference/search/v1-search",
        env_key="YDC_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://ydc-index.io",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=False,
        key_in_url=False,
        options=[
            OptionSpec(
                name="freshness",
                type="string",
                description=(
                    "Only pages from this span: day, week, month or year, or a range of dates as "
                    "YYYY-MM-DDtoYYYY-MM-DD. time_range, when passed, wins. When the query names a span of its own, "
                    "such as 'this week', You.com searches the broader of the two."
                ),
            ),
            OptionSpec(
                name="offset",
                type="integer",
                default=0,
                description="The page of results to return, counted in pages of max_results, from 0 to 9.",
            ),
            OptionSpec(
                name="country",
                type="string",
                enum=_COUNTRIES,
                description="The country to focus the results on, a two-letter code in capitals.",
            ),
            # language and knowledge name their values in the description, not an enum: a host refuses a
            # value its enum lacks, and null, which turns the default off, is in none.
            OptionSpec(
                name="language",
                type="string",
                default="EN",
                description=(
                    f"The language of the results, one of {', '.join(_LANGUAGES)}. Sent as EN unless passed; "
                    "null sends none. You.com ignores a value it does not take."
                ),
            ),
            OptionSpec(
                name="safesearch",
                type="string",
                enum=["off", "moderate", "strict"],
                default="moderate",
                description="How strictly to filter adult content.",
            ),
            OptionSpec(
                name="knowledge",
                type="string",
                default="core",
                description=(
                    "core, the one value, also returns knowledge results from You.com's licensed data providers. "
                    "They carry no URL, so they never become hits and stay in raw. Sent as core unless passed; "
                    "null sends none."
                ),
            ),
            OptionSpec(
                name="include_domains",
                type="array",
                description=(
                    "Up to 500 domains to limit the results to. You.com refuses it together with exclude_domains "
                    "or boost_domains."
                ),
            ),
            OptionSpec(
                name="exclude_domains",
                type="array",
                description="Up to 5,000 domains to leave out of the results.",
            ),
            OptionSpec(
                name="boost_domains",
                type="array",
                description="Up to 500 domains to rank higher, without leaving out the others.",
            ),
            OptionSpec(
                name="strict_operators",
                type="boolean",
                default=False,
                description=(
                    "Apply search operators such as quoted phrases and site: exactly, rather than relaxing them "
                    "when nothing matches. No effect under highlights."
                ),
            ),
            OptionSpec(
                name="extraction",
                type="object",
                default={"extraction_mode": "highlights"},
                description=(
                    'What to return of each page, sent as given. {"extraction_mode": "highlights"}, sent unless '
                    "passed, returns the passages that address the query, which become the snippet; "
                    '{"extraction_mode": "full_page"} returns the page as Markdown, which becomes the text, and '
                    "You.com bills each page it crawls live on top of the search. null sends none, and the snippet "
                    "is each result's description."
                ),
            ),
            OptionSpec(
                name="crawl_timeout",
                type="integer",
                default=10,
                description=(
                    "Seconds, from 1 to 60, to wait for each page under full_page extraction; keep the search's "
                    "own timeout above it. You.com refuses it under highlights."
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
        defaults = _defaults()
        payload: dict[str, Any] = {"query": query}
        for name in _PASSED_THROUGH:
            value = options[name] if name in options else defaults.get(name)
            if value is not None:
                payload[name] = value
        if time_range is not None:
            payload["freshness"] = time_range
        extraction = payload.get("extraction")
        highlights = isinstance(extraction, dict) and extraction.get("extraction_mode") == "highlights"
        count = min(DEFAULT_COUNT if max_results is None else max_results, MAX_RESULTS)
        payload["count"] = min(count, HIGHLIGHTS_MAX_RESULTS) if highlights else count

        endpoint = f"{(self._api_base or '').rstrip('/')}/v1/search"
        response = await self._http.request("POST", endpoint, json=payload, headers={"X-API-Key": self._api_key or ""})
        body = _body(response)
        results = body.get("results")
        if not isinstance(results, dict):
            raise _unexpected(response)
        web = results.get("web", [])
        if not isinstance(web, list):
            raise _unexpected(response)
        # Only the items read until the count is reached are checked, as only they reach the caller.
        hits: list[SearchHit] = []
        for item in web:
            if len(hits) == payload["count"]:
                break
            hit = _hit(item, response)
            if hit is not None:
                hits.append(hit)
        return SearchResult(provider=self.METADATA.name, hits=hits, cost_source="none", raw=body)


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object You.com answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    status = response.status_code
    if status >= 400:
        tag = _STATUS_TAGS.get(status)
        if status == 402:
            tag = (safe_tag(body.get("error")) if isinstance(body, dict) else None) or "payment_required"
        raise ProviderError(YouDotComProvider.METADATA.name, status, tag or "http_error")
    if not isinstance(body, dict):
        raise ProviderError(YouDotComProvider.METADATA.name, status, "invalid_response")
    return body


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field the adapter needs, or carries one of the wrong type."""
    return ProviderError(YouDotComProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit | None:
    """One web result as a hit, or ``None`` for one without a URL a caller can fetch."""
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, description = item.get("url"), item.get("title", ""), item.get("description", "")
    contents = item.get("contents", {})
    if (
        not (url is None or isinstance(url, str))
        or not isinstance(title, str)
        or not isinstance(description, str)
        or not isinstance(contents, dict)
    ):
        raise _unexpected(response)
    highlights, markdown = contents.get("highlights", []), contents.get("markdown", "")
    if not isinstance(highlights, list) or not all(isinstance(part, str) for part in highlights):
        raise _unexpected(response)
    if not isinstance(markdown, str):
        raise _unexpected(response)
    published = _published(item.get("page_age"), response)
    if url is None or not is_web_url(url):
        return None
    snippet = " ".join(part.strip() for part in highlights if part.strip()) or description
    return SearchHit(url=url, title=title, snippet=snippet, text=markdown or None, published=published, raw=item)


def _published(value: Any, response: httpx.Response) -> datetime | None:
    """``page_age`` as an aware datetime, UTC since it carries no zone.

    ``None`` when You.com sends none, or one in another form, which stays in ``raw``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise _unexpected(response)
    try:
        published = datetime.fromisoformat(value)
    except ValueError:
        if _ISO_DATE_START.match(value):
            raise _unexpected(response) from None
        return None
    return published if published.tzinfo is not None else published.replace(tzinfo=UTC)
