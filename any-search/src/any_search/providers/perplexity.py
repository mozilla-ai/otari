"""Perplexity: ``POST /search``, with the key as a Bearer token.

The adapter sends what Perplexity's Python SDK sends for a search: the query,
``max_results`` when it is passed, at most 20, and the native options passed,
nothing else. Without ``max_results`` Perplexity returns its default of 10.
``time_range`` becomes ``search_recency_filter``, and wins over that option
when both are passed.

``search_type`` picks one of Perplexity's two cost tiers: ``web``, its
default, and ``fast``. Perplexity's third type, ``people``, is not offered: it
returns people's profiles, takes up to 50 results, and has no Search API price.

Each result's ``snippet`` is text from the page, the passages relevant to the
query, cut to ``max_tokens_per_page``. It becomes the hit's text, and the
hit's snippet is empty; a result whose snippet is empty has no text. ``date``,
the day the page was published, becomes the hit's date; Perplexity leaves it
out when it has none. ``last_updated`` stays in ``raw``.

Perplexity signals every failure with an HTTP status and an OpenAI-shaped
body. Its ``error.type`` becomes the tag: ``invalid_api_key`` for a refused
key, ``invalid_request`` for a request it refuses, ``invalid_country_code``.
The message is never read, since it can quote what the request sent. A
successful search carries no error of its own. Checked live 2026-11-09:
Perplexity refuses with a 400 a ``max_results`` above 20, which the adapter
never sends; a recency filter, ``time_range`` included, together with
``search_after_date_filter`` or ``search_before_date_filter``; and a value
outside an option's list. It accepts a domain filter that mixes domains to
search with domains to exclude, which its documentation says it does not
apply together.

Every field read from a successful answer is checked, so a change on
Perplexity's side fails the call: ``results`` and
each item's ``url``, ``title`` and ``snippet``, which Perplexity documents on
every result, must be there and be strings, and ``date``, when sent, must be a
string or null that parses as an ISO 8601 date. Anything else raises
``unexpected_response``. Fields the adapter does not read are ignored, so a
field Perplexity adds breaks nothing. Perplexity reports no cost.

API: https://docs.perplexity.ai/api-reference/search-post and
https://docs.perplexity.ai/docs/search/overview. Pricing, read 2026-11-09,
https://docs.perplexity.ai/getting-started/pricing: $5 per 1,000 successful
requests with ``search_type`` ``web`` and $1 per 1,000 with ``fast``, whatever
the result count or the page text asked for; a search with no results is
billed, a refused one is not. 50 requests a second per account.
"""

from datetime import UTC, datetime
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError, safe_tag
from any_search._types import OptionSpec, ProviderMetadata, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 20
"""The most results Perplexity's web search returns for one search."""

# Native options copied into the request as they are; time_range wins over search_recency_filter.
_PASSED_THROUGH = (
    "search_type",
    "search_recency_filter",
    "search_domain_filter",
    "country",
    "search_language_filter",
    "max_tokens_per_page",
    "max_tokens",
    "search_after_date_filter",
    "search_before_date_filter",
    "last_updated_after_filter",
    "last_updated_before_filter",
)


def _date_filter(name: str, what: str) -> OptionSpec:
    return OptionSpec(name=name, type="string", description=f"Only pages {what} this date, as MM/DD/YYYY (3/1/2025).")


class PerplexityProvider(AnySearch):
    """Perplexity's Search API."""

    METADATA = ProviderMetadata(
        name="perplexity",
        doc_url="https://docs.perplexity.ai/api-reference/search-post",
        env_key="PERPLEXITY_API_KEY",
        env_api_base="PERPLEXITY_BASE_URL",
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.perplexity.ai",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=False,
        key_in_url=False,
        options=[
            OptionSpec(
                name="search_type",
                type="string",
                enum=["web", "fast"],
                default="web",
                cost_tier=True,
                description=(
                    "web is Perplexity's general search and fast a quicker one. Perplexity bills each at its own "
                    "price per request."
                ),
            ),
            OptionSpec(
                name="search_recency_filter",
                type="string",
                enum=["hour", "day", "week", "month", "year"],
                description=(
                    "Only pages published within this span. time_range, when passed, wins. Perplexity refuses it "
                    "together with search_after_date_filter or search_before_date_filter."
                ),
            ),
            OptionSpec(
                name="search_domain_filter",
                type="array",
                description=(
                    "Up to 20 domains to search, or to leave out with a leading '-'. Perplexity documents a list "
                    "of either kind, not both."
                ),
            ),
            OptionSpec(
                name="country",
                type="string",
                description="The two-letter ISO 3166-1 country code to localize results to.",
            ),
            OptionSpec(
                name="search_language_filter",
                type="array",
                description="Up to 20 two-letter ISO 639-1 language codes to limit results to.",
            ),
            OptionSpec(
                name="max_tokens_per_page",
                type="integer",
                description="The most tokens of text to take from each page.",
            ),
            OptionSpec(
                name="max_tokens",
                type="integer",
                description="The most tokens of page text across all results.",
            ),
            _date_filter("search_after_date_filter", "published after"),
            _date_filter("search_before_date_filter", "published before"),
            _date_filter("last_updated_after_filter", "last updated after"),
            _date_filter("last_updated_before_filter", "last updated before"),
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
        payload: dict[str, Any] = {"query": query}
        limit = None if max_results is None else min(max_results, MAX_RESULTS)
        if limit is not None:
            payload["max_results"] = limit
        payload.update({name: options[name] for name in _PASSED_THROUGH if options.get(name) is not None})
        if time_range is not None:
            payload["search_recency_filter"] = time_range

        endpoint = f"{(self._api_base or '').rstrip('/')}/search"
        response = await self._http.request(
            "POST", endpoint, json=payload, headers={"Authorization": f"Bearer {self._api_key or ''}"}
        )
        body = _body(response)
        results = body.get("results")
        if not isinstance(results, list):
            raise _unexpected(response)
        # Only the items the caller gets are read, so only they are checked.
        hits = [_hit(item, response) for item in results[:limit]]
        return SearchResult(provider=self.METADATA.name, hits=hits, cost_source="none", raw=body)


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Perplexity answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        error = body.get("error") if isinstance(body, dict) else None
        tag = safe_tag(error.get("type")) if isinstance(error, dict) else None
        raise ProviderError(PerplexityProvider.METADATA.name, response.status_code, tag or "http_error")
    if not isinstance(body, dict):
        raise ProviderError(PerplexityProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field Perplexity documents, or carries one of the wrong type."""
    return ProviderError(PerplexityProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit:
    """One result as a hit: its snippet, text from the page, is the hit's text."""
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, snippet = item.get("url"), item.get("title"), item.get("snippet")
    if not isinstance(url, str) or not url or not isinstance(title, str) or not isinstance(snippet, str):
        raise _unexpected(response)
    published = None
    if item.get("date") is not None:
        published = _published(item["date"])
        if published is None:
            raise _unexpected(response)
    return SearchHit(url=url, title=title, snippet="", text=snippet or None, published=published, raw=item)


def _published(value: Any) -> datetime | None:
    """Perplexity's ``date``: an ISO 8601 date, ``YYYY-MM-DD``. A value without a zone is taken as UTC."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
