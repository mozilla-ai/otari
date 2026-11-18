"""Synthetic: ``POST /v2/search``, with the key as a Bearer token.

The request carries the query and nothing else: Synthetic documents no other
parameter, no result count and no recency filter. ``max_results`` therefore
cuts the hits Synthetic returns, and ``time_range`` drops the hits dated
before the range's first day, in UTC. A hit Synthetic sends without a date is
kept, since nothing says it is outside the range.

Synthetic signals every failure with an HTTP status and an ``error`` sentence
in the body, such as ``Invalid API Key.`` for a 401, which is also the
status for a valid key whose account has no subscription. The sentence never
reaches an error: the tag is the adapter's own, by how it starts, else
``http_error``. A successful search carries no error of its own.

Every field read from a successful answer is checked, so a change on
Synthetic's side fails the call rather than degrading it: ``results`` and each
item's ``url`` and ``title`` must be there and be strings, and ``text`` and
``published`` must be a string, or null, when they are. ``published`` must
parse as an ISO 8601 date. Anything else raises ``unexpected_response``.
Fields the adapter does not read are ignored, so a field Synthetic adds breaks
nothing. Synthetic reports no cost.

API: https://dev.synthetic.new/docs/synthetic/search. Pricing, read
2026-11-05, https://synthetic.new/pricing: the search API comes with every
subscription (Go at $9 a month, Standard at $30, Big at $90), and neither the
pricing page nor the documentation gives a price per search or a search quota.
"""

from datetime import UTC, datetime, time
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError
from any_search._types import TIME_RANGE_SPANS, ProviderMetadata, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 10
"""The most results Synthetic returns for one search; it takes no count."""

# The errors Synthetic was seen to send, by how they start, as tags; any other is "http_error".
_ERROR_TAGS = (
    ("invalid api key", "invalid_api_key"),
    ("authorization header required", "missing_api_key"),
    ("the search api is subscription-only", "subscription_required"),
)


class SyntheticProvider(AnySearch):
    """Synthetic's web search API."""

    METADATA = ProviderMetadata(
        name="synthetic",
        doc_url="https://dev.synthetic.new/docs/synthetic/search",
        env_key="SYNTHETIC_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.synthetic.new",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=False,
        key_in_url=False,
        options=[],
    )

    async def _search(
        self,
        query: str,
        *,
        max_results: int | None,
        time_range: TimeRange | None,
        options: dict[str, Any],
    ) -> SearchResult:
        endpoint = f"{(self._api_base or '').rstrip('/')}/v2/search"
        response = await self._http.request(
            "POST", endpoint, json={"query": query}, headers={"Authorization": f"Bearer {self._api_key or ''}"}
        )
        body = _body(response)
        results = body.get("results")
        if not isinstance(results, list):
            raise _unexpected(response)
        since = None if time_range is None else _since(time_range)
        count = MAX_RESULTS if max_results is None else min(max_results, MAX_RESULTS)
        # Only the items read until the count is reached are checked, as only they reach the caller.
        hits: list[SearchHit] = []
        for item in results:
            if len(hits) == count:
                break
            hit = _hit(item, response)
            if since is None or hit.published is None or hit.published >= since:
                hits.append(hit)
        return SearchResult(provider=self.METADATA.name, hits=hits, cost_source="none", raw=body)


def _since(time_range: TimeRange) -> datetime:
    """The start of the day the range begins on, in UTC, so a page dated by its day alone is not left out."""
    return datetime.combine((datetime.now(UTC) - TIME_RANGE_SPANS[time_range]).date(), time(), UTC)


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Synthetic answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        error = body.get("error") if isinstance(body, dict) else None
        raise ProviderError(SyntheticProvider.METADATA.name, response.status_code, _error_tag(error))
    if not isinstance(body, dict):
        raise ProviderError(SyntheticProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _error_tag(error: Any) -> str:
    """The tag for Synthetic's ``error``, from how it starts, so none of Synthetic's text reaches an error."""
    lowered = error.strip().lower() if isinstance(error, str) else ""
    return next((tag for start, tag in _ERROR_TAGS if lowered.startswith(start)), "http_error")


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field Synthetic documents, or carries one of the wrong type."""
    return ProviderError(SyntheticProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit:
    """One result as a hit: ``url`` and ``title``, and ``text`` and ``published`` when sent."""
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, text = item.get("url"), item.get("title"), item.get("text")
    if not isinstance(url, str) or not url or not isinstance(title, str):
        raise _unexpected(response)
    if text is not None and not isinstance(text, str):
        raise _unexpected(response)
    published = None
    if item.get("published") is not None:
        published = _published(item["published"])
        if published is None:
            raise _unexpected(response)
    return SearchHit(url=url, title=title, snippet="", text=text or None, published=published, raw=item)


def _published(value: Any) -> datetime | None:
    """Synthetic's ``published``: an ISO 8601 date-time. A value without a zone is taken as UTC."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
