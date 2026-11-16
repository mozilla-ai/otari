"""Tavily: ``POST /search``, with the key as a Bearer token.

``max_results`` becomes Tavily's ``max_results``, at most 20, and is sent only
when passed; Tavily answers 10 results without it. ``time_range`` is Tavily's
own parameter of that name.

``search_depth`` is Tavily's cost tier: ``basic``, ``fast`` and ``ultra-fast``
cost 1 credit, ``advanced`` 2. ``auto_parameters`` is not offered, since it
always costs 2 credits and can choose ``advanced`` itself.

Tavily returns each page's full text only when ``include_raw_content`` asks
for it. That text becomes the hit's text, and ``content``, the passages
relevant to the query, its snippet.

A failure is an HTTP status with a sentence in ``detail``, which is not read,
since it can quote the request. The tag comes from the status: ``unauthorized``
(401), ``rate_limited`` (429), ``plan_limit_exceeded`` (432),
``pay_as_you_go_limit_exceeded`` (433), else ``http_error``.

The fields checked in a successful answer:

- ``results`` must be a list.
- Each result must have a ``url``, a ``title`` and a ``content``, all strings.
- ``raw_content`` and ``published_date`` can be absent or null, and are
  strings otherwise. ``published_date`` is an RFC 1123 date, such as
  ``Thu, 15 Oct 2026 00:00:00 GMT``; one that does not parse leaves the hit
  undated and stays in ``raw``.

Anything else raises ``unexpected_response``. Tavily reports credits, not
dollars, so it reports no cost.

API: https://docs.tavily.com/documentation/api-reference/endpoint/search.
Pricing, read 2026-11-16, https://docs.tavily.com/documentation/api-credits:
$0.008 a credit pay-as-you-go, $0.005 to $0.0075 on a monthly plan.
"""

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError
from any_search._types import OptionSpec, ProviderMetadata, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 20

_STATUS_TAGS = {
    401: "unauthorized",
    429: "rate_limited",
    432: "plan_limit_exceeded",
    433: "pay_as_you_go_limit_exceeded",
}


class TavilyProvider(AnySearch):
    METADATA = ProviderMetadata(
        name="tavily",
        doc_url="https://docs.tavily.com/documentation/api-reference/endpoint/search",
        env_key="TAVILY_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.tavily.com",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=False,
        key_in_url=False,
        options=[
            OptionSpec(
                name="search_depth",
                type="string",
                enum=["basic", "fast", "ultra-fast", "advanced"],
                default="basic",
                cost_tier=True,
                description="basic, fast and ultra-fast cost 1 credit a search, advanced 2.",
            ),
            OptionSpec(
                name="include_raw_content",
                type="boolean",
                default=False,
                description="Return each page's full text, as Markdown.",
            ),
            OptionSpec(
                name="topic",
                type="string",
                enum=["general", "news", "finance"],
                default="general",
                description="The kind of search. news dates every hit.",
            ),
            OptionSpec(
                name="chunks_per_source",
                type="integer",
                default=3,
                description="The most passages from each page in its snippet, from 1 to 3.",
            ),
            OptionSpec(name="start_date", type="string", description="Only pages from this date on, as YYYY-MM-DD."),
            OptionSpec(name="end_date", type="string", description="Only pages up to this date, as YYYY-MM-DD."),
            OptionSpec(
                name="include_published_date",
                type="boolean",
                default=False,
                description="Date each hit. topic news does so anyway.",
            ),
            OptionSpec(name="include_domains", type="array", description="Only results from these domains."),
            OptionSpec(name="exclude_domains", type="array", description="No results from these domains."),
            OptionSpec(
                name="country",
                type="string",
                description="Rank results from this country higher, by its English name, such as germany.",
            ),
            OptionSpec(
                name="exact_match", type="boolean", default=False, description="Only pages with the exact query."
            ),
            OptionSpec(name="safe_search", type="boolean", default=False, description="Leave out adult content."),
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
        payload.update({name: value for name, value in options.items() if value is not None})
        if time_range is not None:
            payload["time_range"] = time_range

        endpoint = f"{(self._api_base or '').rstrip('/')}/search"
        response = await self._http.request(
            "POST", endpoint, json=payload, headers={"Authorization": f"Bearer {self._api_key or ''}"}
        )
        body = _body(response)
        results = body.get("results")
        if not isinstance(results, list):
            raise _unexpected(response)
        # Only the results the caller gets are read, so only they are checked.
        hits = [_hit(item, response) for item in results[:limit]]
        return SearchResult(provider=self.METADATA.name, hits=hits, cost_source="none", raw=body)


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Tavily answered with, or raise the error it signaled."""
    if response.status_code >= 400:
        tag = _STATUS_TAGS.get(response.status_code, "http_error")
        raise ProviderError(TavilyProvider.METADATA.name, response.status_code, tag)
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise ProviderError(TavilyProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _unexpected(response: httpx.Response) -> ProviderError:
    """The error for a successful answer with a required field missing, or a field of the wrong type."""
    return ProviderError(TavilyProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit:
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, content = item.get("url"), item.get("title"), item.get("content")
    raw_content, published = item.get("raw_content"), item.get("published_date")
    if not isinstance(url, str) or not url or not isinstance(title, str) or not isinstance(content, str):
        raise _unexpected(response)
    if not isinstance(raw_content, str | None) or not isinstance(published, str | None):
        raise _unexpected(response)
    return SearchHit(
        url=url, title=title, snippet=content, text=raw_content or None, published=_published(published), raw=item
    )


def _published(value: str | None) -> datetime | None:
    """``published_date`` as an aware datetime, or ``None`` when Tavily sends none or one that does not parse."""
    if value is None:
        return None
    try:
        published = parsedate_to_datetime(value)
    except ValueError:
        return None
    return published if published.tzinfo is not None else published.replace(tzinfo=UTC)
