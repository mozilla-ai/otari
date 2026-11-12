"""SpecStream: ``POST /v1/web-search``, with the key as a Bearer token.

SpecStream's Search API takes its parameters in the query string of a GET or,
under the same names, defaults and price, as a JSON body of a POST. The
adapter sends the POST, so neither the query nor the key reaches a URL.

It sends ``q`` and ``num``, from ``max_results`` and at most 100, else 10:
without ``num`` SpecStream returns every result its answer read, and a call is
charged by the results it returns, so the count is always sent and the hits
are cut to it as well. SpecStream has no recency parameter, but holds every
result to the ``after:`` operator in the query (``YYYY-MM-DD``), leaving out a
result shown to break it; ``time_range`` therefore adds ``after:`` with the day
the range starts on, in UTC. A result with no date says nothing either way, so
it is kept.

``speed`` picks one of SpecStream's two cost tiers: ``default``, which also
reads the results to answer the query, and ``fast``, the engines' results with
their snippets. The answer, its rows and the hand-off SpecStream sends beside
the results stay in ``raw``. Images (``images``) are not offered, since a hit
cannot carry one.

Each source's ``snippet`` is the hit's snippet, and its ``excerpt``, sent with
``passages=true``, the page's text closest to the query, is its text.
``published_at``, a news item's time, or else ``published``, the day the page
states it was published, both ISO 8601, becomes the hit's date; a date without
a zone is UTC.

SpecStream signals every failure with an HTTP status and an ``error`` object,
whose ``code`` becomes the tag: ``missing_api_key``, ``invalid_token`` for a
refused key, ``insufficient_balance``, ``bad_request``, ``rate_limited``. The
``message`` is never read. A successful search carries no error of its own.

Every field read from a successful answer is checked, so a change on
SpecStream's side fails the call: ``data``, its ``sources`` and
``price_usd``, and each source's ``url``, ``title`` and ``snippet``, which the
API reference marks required, must be there with the right type, and
``excerpt``, ``published`` and ``published_at``, when sent, must be strings or
null, the dates ones that parse as ISO 8601. Anything else raises
``unexpected_response``. Fields the adapter does not read are ignored, so a
field SpecStream adds breaks nothing. ``price_usd``, what the call was
charged, is the result's cost.

API: https://specstream.ai/docs/search-api. Pricing, read 2026-11-12,
https://specstream.ai/pricing: with ``speed`` ``default``, $5 per 1,000
searches of up to 10 results, and each started block of 10 more adds $0.006;
with ``fast``, $1 per 1,000 searches of up to 10 results, and each block of 10
more adds $0.0012. A search that returns no results costs nothing.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError, safe_tag
from any_search._types import TIME_RANGE_SPANS, OptionSpec, ProviderMetadata, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 100
"""The most results SpecStream returns for one web search."""

DEFAULT_RESULTS = 10
"""The count sent without ``max_results``: one search at the price of its first block."""

# Native options copied into the request body as they are.
_PASSED_THROUGH = ("speed", "news", "passages", "pages_wait_ms", "language", "country", "safe", "engines", "page")


class SpecStreamProvider(AnySearch):
    """SpecStream's Search API."""

    METADATA = ProviderMetadata(
        name="specstream",
        doc_url="https://specstream.ai/docs/search-api",
        env_key="SPECSTREAM_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.specstream.ai",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=False,
        key_in_url=False,
        reports_cost=True,
        options=[
            OptionSpec(
                name="speed",
                type="string",
                enum=["default", "fast"],
                default="default",
                cost_tier=True,
                description=(
                    "default also reads the results to answer the query; fast returns the engines' results with "
                    "their snippets in about a second. SpecStream bills each at its own price and reports the cost."
                ),
            ),
            OptionSpec(
                name="news",
                type="boolean",
                default=False,
                description="News items instead of web results, newest first, each dated. Not with passages.",
            ),
            OptionSpec(
                name="passages",
                type="boolean",
                default=False,
                description="Also return the text of the top pages closest to the query, as each hit's text.",
            ),
            OptionSpec(
                name="pages_wait_ms",
                type="integer",
                description="With passages: how long, 0 to 3500 milliseconds, the call waits for the pages' text.",
            ),
            OptionSpec(
                name="language",
                type="string",
                description="The language of the results, a two-letter ISO 639-1 code such as de.",
            ),
            OptionSpec(
                name="country",
                type="string",
                description="The country the results are for, a two-letter ISO 3166-1 code such as DE.",
            ),
            OptionSpec(
                name="safe",
                type="string",
                enum=["off", "moderate", "strict"],
                description="The safe search level, sent to every engine that takes one.",
            ),
            OptionSpec(
                name="engines",
                type="string",
                description=(
                    "The engines the search may ask, comma-separated: names, or two-letter country codes of their "
                    "operators, with a leading - to deny one."
                ),
            ),
            OptionSpec(
                name="page",
                type="integer",
                description="The page of results, 1 to 10, each of the count asked for.",
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
        count = DEFAULT_RESULTS if max_results is None else min(max_results, MAX_RESULTS)
        payload: dict[str, Any] = {"q": query if time_range is None else f"{query} after:{_since(time_range)}"}
        payload["num"] = count
        # An option set to null is not sent.
        payload.update({name: options[name] for name in _PASSED_THROUGH if options.get(name) is not None})

        endpoint = f"{(self._api_base or '').rstrip('/')}/v1/web-search"
        response = await self._http.request(
            "POST", endpoint, json=payload, headers={"Authorization": f"Bearer {self._api_key or ''}"}
        )
        body = _body(response)
        data = body.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
            raise _unexpected(response)
        cost = _cost(data, response)
        # Only the items the caller gets are read, so only they are checked.
        hits = [_hit(item, response) for item in data["sources"][:count]]
        return SearchResult(provider=self.METADATA.name, hits=hits, cost=cost, cost_source="reported", raw=body)


def _since(time_range: TimeRange) -> str:
    """The day the range starts on, in UTC, as the ``YYYY-MM-DD`` SpecStream's ``after:`` takes."""
    return (datetime.now(UTC) - TIME_RANGE_SPANS[time_range]).date().isoformat()


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object SpecStream answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        error = body.get("error") if isinstance(body, dict) else None
        tag = safe_tag(error.get("code")) if isinstance(error, dict) else None
        raise ProviderError(SpecStreamProvider.METADATA.name, response.status_code, tag or "http_error")
    if not isinstance(body, dict):
        raise ProviderError(SpecStreamProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field SpecStream documents, or carries one of the wrong type."""
    return ProviderError(SpecStreamProvider.METADATA.name, response.status_code, "unexpected_response")


def _cost(data: dict[str, Any], response: httpx.Response) -> Decimal:
    """SpecStream's ``price_usd``: what the call was charged, which every successful answer carries."""
    price = data.get("price_usd")
    if isinstance(price, bool) or not isinstance(price, int | float) or not 0 <= price < float("inf"):
        raise _unexpected(response)
    return Decimal(str(price))


def _hit(item: Any, response: httpx.Response) -> SearchHit:
    """One source as a hit: its snippet, its excerpt as text, and its date."""
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, snippet, excerpt = item.get("url"), item.get("title"), item.get("snippet"), item.get("excerpt")
    if not isinstance(url, str) or not url or not isinstance(title, str) or not isinstance(snippet, str):
        raise _unexpected(response)
    if excerpt is not None and not isinstance(excerpt, str):
        raise _unexpected(response)
    published = None
    for field in ("published_at", "published"):
        if item.get(field) is None:
            continue
        parsed = _published(item[field])
        if parsed is None:
            raise _unexpected(response)
        published = published or parsed
    return SearchHit(url=url, title=title, snippet=snippet, text=excerpt or None, published=published, raw=item)


def _published(value: Any) -> datetime | None:
    """An ISO 8601 date or time, as SpecStream sends ``published`` and ``published_at``; without a zone, UTC."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
