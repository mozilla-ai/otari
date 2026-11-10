"""You.com: ``POST /v1/search``, with the key in the ``X-API-Key`` header.

Sends what You.com's SDK (``youdotcom`` 3.6.0) sends: the query, ``count``,
``language`` ``EN``, ``knowledge`` ``core``, highlights extraction, and the
native options passed. ``None`` is not sent, which turns those defaults off.
``count`` is ``max_results`` (10 when unset), at most 100, and at most 50 under
highlights. ``time_range`` becomes ``freshness`` and wins over that option.

Joined highlights are the snippet, or ``description`` when there are none.
``contents.markdown``, only under ``full_page``, is the text. ``page_age`` is
the date when it parses as ISO 8601, taken as UTC; another form stays in
``raw``.

401, 403 and 422 are tagged by status (``unauthorized``, ``forbidden``,
``invalid_request``): those bodies are a sentence or the gateway's, not a code.
A 402 is tagged from its ``error``, else ``payment_required``. Any other status
is ``http_error``.

``results`` must be an object, and ``web`` a list when present. Each item read
must have ``url`` and ``page_age`` as a string or null, ``title`` and
``description`` as strings, and, when present, ``contents`` as an object whose
``highlights`` is a list of strings and whose ``markdown`` is a string. A
``page_age`` that looks like ISO 8601 but does not parse is refused. A result
without an http or https URL is dropped. Anything else raises
``unexpected_response``. Unread fields are ignored. You.com reports no cost.

API: https://you.com/docs/api-reference/search/v1-search. Pricing, read
2026-11-09, https://you.com/docs/administration/billing: $5.00 per 1,000 calls.
Full-page extraction adds $1.00 per 1,000 pages crawled live.
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
                    "Only pages from day, week, month, year, or YYYY-MM-DDtoYYYY-MM-DD. time_range wins. A span "
                    "named in the query, such as 'this week', widens this to the broader of the two."
                ),
            ),
            OptionSpec(
                name="offset",
                type="integer",
                default=0,
                description="Result page, in pages of max_results, from 0 to 9.",
            ),
            OptionSpec(
                name="country",
                type="string",
                enum=_COUNTRIES,
                description="Country to focus on, a two-letter code in capitals.",
            ),
            # language and knowledge name their values in the description, not an enum: a host refuses a
            # value its enum lacks, and null, which turns the default off, is in none.
            OptionSpec(
                name="language",
                type="string",
                default="EN",
                description=(
                    f"Result language, one of {', '.join(_LANGUAGES)}. EN unless passed; null sends none. "
                    "You.com ignores a value it does not take."
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
                    "core, the only value, also returns licensed knowledge results. They have no URL, so they stay "
                    "in raw. Sent unless passed; null sends none."
                ),
            ),
            OptionSpec(
                name="include_domains",
                type="array",
                description="Up to 500 domains to limit results to. Refused with exclude_domains or boost_domains.",
            ),
            OptionSpec(
                name="exclude_domains",
                type="array",
                description="Up to 5,000 domains to leave out.",
            ),
            OptionSpec(
                name="boost_domains",
                type="array",
                description="Up to 500 domains to rank higher, without leaving the others out.",
            ),
            OptionSpec(
                name="strict_operators",
                type="boolean",
                default=False,
                description="Apply quoted phrases and site: exactly. No effect under highlights.",
            ),
            OptionSpec(
                name="extraction",
                type="object",
                default={"extraction_mode": "highlights"},
                description=(
                    "Sent as given. highlights, sent unless passed, returns the passages that become the snippet; "
                    "full_page returns Markdown as the text and bills each page crawled live. null sends none."
                ),
            ),
            OptionSpec(
                name="crawl_timeout",
                type="integer",
                default=10,
                description="Seconds to wait per page under full_page, 1 to 60. Refused under highlights.",
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
