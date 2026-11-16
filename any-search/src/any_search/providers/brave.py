"""Brave: ``GET /res/v1/web/search``, with the key in the ``X-Subscription-Token`` header.

The query travels in the URL, as ``q``. The adapter passes it to ``self._http``
as ``params``, and the log filter (``install_log_filter``) redacts the URL in
httpx's log.

``max_results`` becomes ``count``, at most 20, Brave's limit. ``time_range``
becomes ``freshness`` (``pd``, ``pw``, ``pm`` or ``py``) and wins over the
``freshness`` option. Brave ignores a ``freshness`` value it does not know.

A failure is an HTTP status with a code in ``error.code``, such as
``SUBSCRIPTION_TOKEN_INVALID`` for a refused key, which is a 422, or
``VALIDATION`` for a count above 20. The tag is that code in lower case, else
``http_error``. The rest of the body can quote the request and is not read.

The fields checked in a successful answer:

- ``web`` is absent or null when Brave found nothing, which makes a search
  with no hits. Otherwise it must hold a ``results`` list.
- Each result must have a ``url`` and a ``title``, both strings.
- ``description`` and ``page_age`` can be absent or null, and are strings otherwise.
  ``page_age`` is an ISO 8601 date, read as UTC when it has no zone. It comes
  from the page's own metadata, so a value that does not parse leaves the hit
  undated and stays in ``raw``.

Anything else raises ``unexpected_response``. Brave reports no cost.

API: https://api-dashboard.search.brave.com/api-reference/web/search/get.
Pricing, read 2026-11-14, https://brave.com/search/api/: $5 per 1,000
requests, with $5 of free credit a month. Only successful requests are billed.
"""

from datetime import UTC, datetime
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError, safe_tag
from any_search._types import OptionSpec, ProviderMetadata, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 20

_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}


class BraveProvider(AnySearch):
    METADATA = ProviderMetadata(
        name="brave",
        doc_url="https://api-dashboard.search.brave.com/api-reference/web/search/get",
        env_key="BRAVE_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.search.brave.com",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=True,
        key_in_url=False,
        options=[
            OptionSpec(
                name="country",
                type="string",
                default="US",
                description="The country results come from: a two-letter code such as US or DE, or ALL.",
            ),
            OptionSpec(
                name="search_lang",
                type="string",
                default="en",
                description="The language of the results, such as en, fr or pt-br.",
            ),
            OptionSpec(
                name="offset",
                type="integer",
                default=0,
                description="The number of pages of count results to skip, from 0 to 9, to page through results.",
            ),
            OptionSpec(
                name="safesearch",
                type="string",
                enum=["off", "moderate", "strict"],
                default="moderate",
                description=(
                    "Adult content: strict leaves it all out, moderate keeps adult sites but not explicit media."
                ),
            ),
            OptionSpec(
                name="freshness",
                type="string",
                description=(
                    "Only pages from the past day (pd), week (pw), month (pm) or year (py), or a range such as "
                    "2026-01-01to2026-06-30. Brave ignores a value it does not know. time_range, when passed, wins."
                ),
            ),
            OptionSpec(
                name="spellcheck",
                type="boolean",
                default=True,
                description="Search for the spell-checked query, which the answer reports in raw.",
            ),
            OptionSpec(
                name="text_decorations",
                type="boolean",
                default=True,
                description="Mark the query's words in snippets with HTML such as <strong>.",
            ),
            OptionSpec(
                name="extra_snippets",
                type="boolean",
                default=False,
                description="Up to 5 more excerpts from each page, in raw.",
            ),
            OptionSpec(
                name="result_filter",
                type="string",
                description=(
                    "The sections the answer holds, comma-separated, such as web or web,news; all by default. "
                    "Without web, a search has no hits."
                ),
            ),
            OptionSpec(
                name="goggles",
                type="string",
                description="A Goggle, by URL or inline, to re-rank results with.",
            ),
            OptionSpec(
                name="operators",
                type="boolean",
                default=True,
                description="Apply search operators in the query, such as site:.",
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
        count = MAX_RESULTS if max_results is None else min(max_results, MAX_RESULTS)
        params: dict[str, Any] = {"q": query, "count": count}
        params.update({name: value for name, value in options.items() if value is not None})
        if time_range is not None:
            params["freshness"] = _FRESHNESS[time_range]
        headers = {"Accept": "application/json", "X-Subscription-Token": self._api_key or ""}

        endpoint = f"{(self._api_base or '').rstrip('/')}/res/v1/web/search"
        response = await self._http.request("GET", endpoint, params=params, headers=headers)
        body = _body(response)
        hits: list[SearchHit] = []
        # Brave leaves ``web`` out, or sends it as null, when it found nothing.
        web = body.get("web")
        if web is not None:
            results = web.get("results") if isinstance(web, dict) else None
            if not isinstance(results, list):
                raise _unexpected(response)
            # Only the results the caller gets are read, so only they are checked.
            hits = [_hit(item, response) for item in results[:count]]
        return SearchResult(provider=self.METADATA.name, hits=hits, cost_source="none", raw=body)


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Brave answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        error = body.get("error") if isinstance(body, dict) else None
        code = safe_tag(error.get("code")) if isinstance(error, dict) else None
        tag = code.lower() if code else "http_error"
        raise ProviderError(BraveProvider.METADATA.name, response.status_code, tag)
    if not isinstance(body, dict):
        raise ProviderError(BraveProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _unexpected(response: httpx.Response) -> ProviderError:
    """The error for a successful answer with a required field missing, or a field of the wrong type."""
    return ProviderError(BraveProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit:
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, description = item.get("url"), item.get("title"), item.get("description") or ""
    if not isinstance(url, str) or not url or not isinstance(title, str) or not isinstance(description, str):
        raise _unexpected(response)
    return SearchHit(url=url, title=title, snippet=description, published=_published(item, response), raw=item)


def _published(item: dict[str, Any], response: httpx.Response) -> datetime | None:
    """``page_age`` as an aware datetime, or ``None`` when Brave sends none or one that does not parse."""
    value = item.get("page_age")
    if value is None:
        return None
    if not isinstance(value, str):
        raise _unexpected(response)
    try:
        published = datetime.fromisoformat(value)
    except ValueError:
        return None
    return published if published.tzinfo is not None else published.replace(tzinfo=UTC)
