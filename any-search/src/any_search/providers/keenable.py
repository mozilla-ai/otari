"""Keenable: ``POST /v1/search`` with an API key, ``POST /v1/search/public`` without one.

The key is optional, as in Keenable's SDK (``keenable`` 0.3.0). With one,
passed or read from ``KEENABLE_API_KEY``, the adapter calls ``/v1/search`` with
the key in the ``X-API-Key`` header. Without one it calls ``/v1/search/public``,
which Keenable limits to 1,000 searches an hour and 10 a second per IP address,
shared with every other caller from that address. The query travels in the
JSON body.

Every call carries the ``X-Keenable-Title`` header, which the keyless endpoint
requires and Keenable logs with each query to tell apps apart. It is
``any-search`` unless the ``X-Keenable-Title`` option names the app. Only the
operator sets that option, since its value can name the deployment.

``max_results`` becomes ``max_results``, 10 when unset, at most 50.
``time_range`` becomes ``published_after`` as a span back from now, ``1d``,
``7d``, ``30d`` or ``365d``, and wins over that option; Keenable then leaves out
the pages it knows no publish date for. The other native options go into the
body under Keenable's names, and an option set to ``None`` is not sent.
Keenable refuses a field it does not know with a 400.

A hit's snippet is ``snippet``, Markdown Keenable cuts from the page for the
query, or ``description`` when ``snippet`` is blank; Keenable documents
``description`` as deprecated and sends it empty. No page text comes back, so
``text`` is ``None``. ``published_at`` is the publish date, ISO 8601 in UTC,
absent when Keenable knows none.

Any status from 400 raises ``ProviderError``, tagged with Keenable's ``code``,
such as ``invalid_api_key``, ``missing_app_title``, ``invalid_parameter``,
``insufficient_credits`` or ``public_rate_limit_hourly``. A body without one is
``rate_limited`` for a 429 and ``http_error`` for any other status. The rest of
the body can quote the request and is not read.

The fields checked in a successful answer:

- ``results`` must be a list.
- Each result must have ``url``, a non-empty string, and ``title``, a string.
- ``snippet``, ``description`` and ``published_at`` are strings when present.
  A ``published_at`` that is not ISO 8601 raises.

Anything else raises ``unexpected_response``. Unread fields, such as
``acquired_at`` and a keyed answer's ``usage``, stay in ``raw``. ``usage``
reports a keyed call's cost in credits; the adapter does not read it, so the
result's cost is ``None``.

API: https://docs.keenable.ai/api-reference/search, read 2026-11-24. Pricing,
read 2026-11-24, https://docs.keenable.ai/pricing: keyless calls are free; with
a key, 100,000 requests a month are free, then $4 per 1,000 from a prepaid
balance, in ``pro`` and ``realtime`` mode alike.
"""

from datetime import UTC, datetime
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError, safe_tag
from any_search._types import OptionSpec, ProviderMetadata, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 50
"""The most results Keenable returns for one search."""

DEFAULT_COUNT = 10
"""Keenable's own ``max_results`` when none is passed."""

DEFAULT_TITLE = "any-search"
"""The ``X-Keenable-Title`` every call carries unless the option names the app."""

TITLE_HEADER = "X-Keenable-Title"

# Native options sent in the body as they are; time_range wins over published_after.
_BODY_OPTIONS = (
    "mode",
    "site",
    "exclude_domains",
    "published_after",
    "published_before",
    "acquired_after",
    "acquired_before",
    "snippet_max_length",
    "disable_cards",
    "moderation",
    "query_time",
    "objective",
)

_PUBLISHED_AFTER = {"day": "1d", "week": "7d", "month": "30d", "year": "365d"}

_TOO_MANY_REQUESTS = 429


class KeenableProvider(AnySearch):
    """Keenable's search API, keyed or keyless."""

    METADATA = ProviderMetadata(
        name="keenable",
        doc_url="https://docs.keenable.ai/api-reference/search",
        env_key="KEENABLE_API_KEY",
        env_api_base="KEENABLE_API_URL",
        requires_api_key=False,
        requires_api_base=False,
        default_api_base="https://api.keenable.ai",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=False,
        key_in_url=False,
        options=[
            OptionSpec(
                name="mode",
                type="string",
                enum=["pro", "realtime"],
                default="pro",
                description="pro retrieves more deeply; realtime answers fastest. Both cost the same.",
            ),
            OptionSpec(
                name="site",
                type="string",
                description="Only results from one site, its subdomains included, such as arxiv.org or github.com/org.",
            ),
            OptionSpec(
                name="exclude_domains",
                type="array",
                description=(
                    "Hosts to leave out, with their subdomains, such as example.com; a leading dot leaves out a "
                    "suffix, such as .gov."
                ),
            ),
            OptionSpec(
                name="published_after",
                type="string",
                description=(
                    "Only pages published on or after a date (YYYY-MM-DD), an ISO 8601 timestamp, or a span back "
                    "from now such as 7d or 2mo; pages with no known publish date are left out. time_range, when "
                    "passed, wins."
                ),
            ),
            OptionSpec(
                name="published_before",
                type="string",
                description="Only pages published on or before a date, a timestamp, or a span back from now.",
            ),
            OptionSpec(
                name="acquired_after",
                type="string",
                description="Only pages Keenable indexed on or after a date, a timestamp, or a span back from now.",
            ),
            OptionSpec(
                name="acquired_before",
                type="string",
                description="Only pages Keenable indexed on or before a date, a timestamp, or a span back from now.",
            ),
            OptionSpec(
                name="snippet_max_length",
                type="integer",
                description="The most characters of page text in each snippet, from 180 to 10,000.",
            ),
            OptionSpec(
                name="disable_cards",
                type="boolean",
                default=False,
                description=(
                    "Return web pages only, without the cards Keenable writes for queries such as prices or "
                    "exchange rates, whose URLs expire after a day."
                ),
            ),
            OptionSpec(
                name="moderation",
                type="string",
                enum=["moderate", "strict"],
                description="Leave out adult content, moderately or strictly. Not filtered unless passed.",
            ),
            OptionSpec(
                name="query_time",
                type="string",
                description=(
                    "Search the index as it stood at a date, a timestamp, or a span back from now. Needs Keenable's "
                    "Time Machine on the account, else a 403."
                ),
            ),
            OptionSpec(
                name="objective",
                type="string",
                description="What the search is for, in up to 1,000 characters. Keenable logs it with the query.",
            ),
            OptionSpec(
                name=TITLE_HEADER,
                type="string",
                default=DEFAULT_TITLE,
                description=(
                    "The app name sent as a header, up to 256 characters, which Keenable logs with each search "
                    "and its keyless endpoint requires."
                ),
                operator_only=True,
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
        count = min(DEFAULT_COUNT if max_results is None else max_results, MAX_RESULTS)
        payload: dict[str, Any] = {"query": query, "max_results": count}
        payload.update({name: options[name] for name in _BODY_OPTIONS if options.get(name) is not None})
        if time_range is not None:
            payload["published_after"] = _PUBLISHED_AFTER[time_range]
        headers = {TITLE_HEADER: str(options.get(TITLE_HEADER) or DEFAULT_TITLE)}
        path = "/v1/search"
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        else:
            path = "/v1/search/public"

        endpoint = f"{(self._api_base or '').rstrip('/')}{path}"
        response = await self._http.request("POST", endpoint, json=payload, headers=headers)
        body = _body(response)
        results = body.get("results")
        if not isinstance(results, list):
            raise _unexpected(response)
        # Only the results the caller gets are read, so only they are checked.
        hits = [_hit(item, response) for item in results[:count]]
        return SearchResult(provider=self.METADATA.name, hits=hits, cost_source="none", raw=body)


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Keenable answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    status = response.status_code
    if status >= 400:
        code = safe_tag(body.get("code")) if isinstance(body, dict) else None
        tag = code or ("rate_limited" if status == _TOO_MANY_REQUESTS else "http_error")
        raise ProviderError(KeenableProvider.METADATA.name, status, tag)
    if not isinstance(body, dict):
        raise ProviderError(KeenableProvider.METADATA.name, status, "invalid_response")
    return body


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field the adapter needs, or carries one of the wrong type."""
    return ProviderError(KeenableProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit:
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title = item.get("url"), item.get("title")
    snippet, description = item.get("snippet", ""), item.get("description", "")
    if not isinstance(url, str) or not url or not isinstance(title, str):
        raise _unexpected(response)
    if not isinstance(snippet, str) or not isinstance(description, str):
        raise _unexpected(response)
    return SearchHit(
        url=url,
        title=title,
        snippet=snippet if snippet.strip() else description,
        published=_published(item.get("published_at"), response),
        raw=item,
    )


def _published(value: Any, response: httpx.Response) -> datetime | None:
    """``published_at`` as an aware datetime, UTC when it carries no zone; ``None`` when Keenable knows no date."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise _unexpected(response)
    try:
        published = datetime.fromisoformat(value)
    except ValueError:
        raise _unexpected(response) from None
    return published if published.tzinfo is not None else published.replace(tzinfo=UTC)
