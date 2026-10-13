"""Exa: ``POST /search``, with the key in the ``x-api-key`` header.

What it sends by default is what Exa's own Python SDK sends, so moving from
the SDK changes nothing: ``type`` ``auto``, and page text up to 10,000
characters when the caller passes no ``contents``. A caller who wants less, or
something else, passes ``contents`` and it is sent as given; ``false`` sends
none, which leaves only titles and URLs.

Exa signals every failure with an HTTP status and a ``tag`` in the body; a
successful search carries no error of its own.

Every field read from a successful answer is checked, so a change on Exa's
side fails the call rather than degrading it: ``results`` and each item's
``url`` and ``title`` must be there, and ``text``, ``highlights``,
``publishedDate`` and ``costDollars.total`` must have the right type when they
are. Anything else raises ``unexpected_response``. Exa documents the cost as
optional, so a search without one reports none, and the live tests say so. Fields the adapter does not read
are ignored, so a field Exa adds breaks nothing.

API: https://exa.ai/docs/reference/search. Pricing, read 2026-10-08:
https://exa.ai/docs/admin/pricing.
"""

import copy
import math
import re
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError
from any_search._types import OptionSpec, ProviderMetadata, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 100
"""The most results Exa's public API returns for one search."""

DEFAULT_CONTENTS: dict[str, Any] = {"text": {"maxCharacters": 10_000}}
"""What Exa's Python SDK asks for when its caller names no contents."""

_TIME_RANGE = {
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
    "year": timedelta(days=365),
}

# Native options copied into the request as they are. ``type``, ``numResults``
# and ``contents`` have rules of their own, below.
_PASSED_THROUGH = (
    "category",
    "includeDomains",
    "excludeDomains",
    "startPublishedDate",
    "endPublishedDate",
    "userLocation",
    "moderation",
    "additionalQueries",
)

# An error tag is kept only when it looks like one, so no text from the body
# reaches an exception.
_TAG = re.compile(r"[A-Za-z0-9_.-]{1,64}")


class ExaProvider(AnySearch):
    """Exa's search endpoint."""

    METADATA = ProviderMetadata(
        name="exa",
        doc_url="https://exa.ai/docs/reference/search",
        env_key="EXA_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.exa.ai",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=False,
        key_in_url=False,
        options=[
            OptionSpec(
                name="type",
                type="string",
                enum=["instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"],
                default="auto",
                description="The search mode. The deep modes cost more and take longer.",
            ),
            OptionSpec(
                name="numResults",
                type="integer",
                default=10,
                description="How many results to return, used only when max_results is not passed.",
            ),
            OptionSpec(
                name="contents",
                type="object",
                default=copy.deepcopy(DEFAULT_CONTENTS),
                description=(
                    "What to return of each page: text, highlights, summary, extras. Sent as given; false "
                    "sends none. Highlights become the hit's snippet and text its text."
                ),
            ),
            OptionSpec(
                name="category",
                type="string",
                enum=["company", "publication", "news", "personal site", "financial report", "people"],
                description="A category of results to focus on.",
            ),
            OptionSpec(name="includeDomains", type="array", description="Only return results from these domains."),
            OptionSpec(name="excludeDomains", type="array", description="Never return results from these domains."),
            OptionSpec(
                name="startPublishedDate",
                type="string",
                description="Only results published after this ISO 8601 date-time. time_range, when passed, wins.",
            ),
            OptionSpec(
                name="endPublishedDate",
                type="string",
                description="Only results published before this ISO 8601 date-time.",
            ),
            OptionSpec(
                name="userLocation",
                type="string",
                description="The user's two-letter ISO 3166-1 country code, to localize results.",
            ),
            OptionSpec(name="moderation", type="boolean", default=False, description="Filter unsafe content."),
            OptionSpec(
                name="additionalQueries",
                type="array",
                description="Up to 10 more phrasings of the query, for the deep modes only.",
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
        payload: dict[str, Any] = {"query": query, "type": options.get("type") or "auto"}
        count: Any = max_results if max_results is not None else options.get("numResults")
        # A count Exa cannot take is sent as it is, for Exa to refuse; only a usable one caps the hits.
        limit = count if isinstance(count, int) and not isinstance(count, bool) and count >= 1 else None
        if limit is not None:
            count = limit = min(limit, MAX_RESULTS)
        if count is not None:
            payload["numResults"] = count
        payload.update({name: options[name] for name in _PASSED_THROUGH if name in options})
        if time_range is not None:
            payload["startPublishedDate"] = _since(time_range)
        contents = options.get("contents")
        if contents is None:
            payload["contents"] = copy.deepcopy(DEFAULT_CONTENTS)
        elif contents is not False:
            payload["contents"] = contents

        endpoint = f"{(self._api_base or '').rstrip('/')}/search"
        response = await self._http.request("POST", endpoint, json=payload, headers={"x-api-key": self._api_key or ""})
        body = _body(response)
        results = body.get("results")
        if not isinstance(results, list):
            raise _unexpected(response)
        # Only the items the caller gets are read, so only they are checked.
        hits = [_hit(item, response) for item in (results if limit is None else results[:limit])]
        cost = _cost(body, response)
        return SearchResult(
            provider=self.METADATA.name,
            hits=hits,
            cost=cost,
            cost_source="none" if cost is None else "reported",
            raw=body,
        )


def _since(time_range: TimeRange) -> str:
    """The start of the day the range begins on, in UTC.

    Exa often dates a page by its day alone, as midnight, so a start taken to
    the second would leave out pages from the range's first day.
    """
    start = datetime.combine((datetime.now(UTC) - _TIME_RANGE[time_range]).date(), time(), UTC)
    return start.isoformat(timespec="seconds").replace("+00:00", "Z")


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Exa answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        tag = body.get("tag") if isinstance(body, dict) else None
        raise ProviderError(ExaProvider.METADATA.name, response.status_code, _tag(tag) or "http_error")
    if not isinstance(body, dict):
        raise ProviderError(ExaProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _tag(value: Any) -> str | None:
    """Exa's error tag when it looks like a code, else ``None``, so no body text reaches an error."""
    return value if isinstance(value, str) and _TAG.fullmatch(value) else None


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field Exa documents, or carries one of the wrong type."""
    return ProviderError(ExaProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit:
    """One result as a hit: ``url`` and ``title``, which Exa documents on every result, and the rest when sent."""
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title = item.get("url"), item.get("title")
    if not isinstance(url, str) or not url or not isinstance(title, str):
        raise _unexpected(response)
    text = item.get("text", "")
    highlights = item.get("highlights", [])
    if not isinstance(text, str) or not isinstance(highlights, list):
        raise _unexpected(response)
    if not all(isinstance(part, str) for part in highlights):
        raise _unexpected(response)
    published = None
    if "publishedDate" in item:
        published = _published(item["publishedDate"])
        if published is None:
            raise _unexpected(response)
    return SearchHit(
        url=url,
        title=title,
        snippet=" ".join(part.strip() for part in highlights if part.strip()),
        text=text or None,
        published=published,
        raw=item,
    )


def _published(value: Any) -> datetime | None:
    """Exa's ``publishedDate``: a date, or a date-time. A value without a zone is taken as UTC."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _cost(body: dict[str, Any], response: httpx.Response) -> Decimal | None:
    """Exa's ``costDollars.total``, or ``None`` when it sent none."""
    costs = body.get("costDollars")
    if costs is None:
        return None
    if not isinstance(costs, dict):
        raise _unexpected(response)
    total = costs.get("total")
    if total is None:
        return None
    if isinstance(total, bool) or not isinstance(total, int | float) or not 0 <= total < math.inf:
        raise _unexpected(response)
    return Decimal(str(total))
