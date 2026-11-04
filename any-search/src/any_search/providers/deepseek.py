"""DeepSeek: ``POST /v1/tools/search``, with the key in a bearer header.

DeepSeek's Search API is a search engine of its own, apart from its models.
DeepSeek publishes no SDK for it, so the adapter sends what its documentation
shows: the query, ``count`` when ``max_results`` is passed, at most 50, and
nothing else unless the caller passes a native option. Without ``max_results``
DeepSeek returns its default of 10. Each hit's snippet is DeepSeek's
``snippet``; its longer ``summary``, written for a model, stays in ``raw``, and
page text comes back only with ``need_content``.

DeepSeek has no recency parameter, so ``time_range`` becomes its ``after:``
query operator, the day the range starts on, appended to the query. DeepSeek
counts that day in from midnight China Standard Time (UTC+8), eight hours
before the UTC day, so the range covers at least its span. The operator counts
toward DeepSeek's 128-character limit on the query, so a query near that limit
is refused with a 400.

DeepSeek signals every failure with an HTTP status and an OpenAI-shaped body.
Its ``error.type`` becomes the tag (``authentication_error`` for a refused key,
``invalid_request_error`` for a bad request), and ``error.code`` when there is
no type, since DeepSeek sends ``invalid_request_error`` as the code of every
failure. The message is never read: it carries the key's last four characters
for a refused key. A successful search carries no error of its own.

Every field read from a successful answer is checked, so a change on
DeepSeek's side fails the call rather than degrading it: ``results`` and each
item's ``url``, ``title``, ``snippet``, ``content`` and ``date``, all of which
DeepSeek documents on every result, must be there, the first four as strings
and ``date`` as an RFC 3339 date-time or null. Anything else raises
``unexpected_response``. DeepSeek reports no cost. Fields the adapter does not
read are ignored, so a field DeepSeek adds breaks nothing.

API: https://api-docs.deepseek.com/api/search and
https://api-docs.deepseek.com/guides/search. Pricing, read 2026-10-19: $0.75
per 1,000 successful searches, whatever ``count`` is, and none for a search that
returns no results; 100 searches a second per account, apart from fetches.
DeepSeek's free period covers only calls made through its Harness, Claude Code
or Codex, not this adapter's.
"""

from datetime import UTC, datetime
from typing import Any

import httpx

from any_search._api import AnySearch
from any_search._errors import ProviderError, safe_tag
from any_search._types import TIME_RANGE_SPANS, OptionSpec, ProviderMetadata, SearchHit, SearchResult, TimeRange

MAX_RESULTS = 50
"""The most results DeepSeek returns for one search."""

# Native options copied into the request as they are.
_PASSED_THROUGH = ("ori_question", "localization", "need_content")


class DeepSeekProvider(AnySearch):
    """DeepSeek's Search API."""

    METADATA = ProviderMetadata(
        name="deepseek",
        doc_url="https://api-docs.deepseek.com/api/search",
        env_key="DEEPSEEK_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.deepseek.com",
        tier="production",
        max_results=MAX_RESULTS,
        query_in_url=False,
        key_in_url=False,
        options=[
            OptionSpec(
                name="ori_question",
                type="string",
                description="The question behind the search, up to 128 characters, to help DeepSeek read the query.",
            ),
            OptionSpec(
                name="localization",
                type="string",
                default="",
                description="An IP address to localize the results to.",
            ),
            OptionSpec(
                name="need_content",
                type="boolean",
                default=False,
                description="Return each page's text, which becomes the hit's text. The price is the same.",
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
        payload: dict[str, Any] = {"query": query if time_range is None else f"{query} after:{_since(time_range)}"}
        limit = None if max_results is None else min(max_results, MAX_RESULTS)
        if limit is not None:
            payload["count"] = limit
        payload.update({name: options[name] for name in _PASSED_THROUGH if options.get(name) is not None})

        endpoint = f"{(self._api_base or '').rstrip('/')}/v1/tools/search"
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


def _since(time_range: TimeRange) -> str:
    """The day the range starts on, in UTC, as the ``YYYY-MM-DD`` DeepSeek's ``after:`` takes."""
    return (datetime.now(UTC) - TIME_RANGE_SPANS[time_range]).date().isoformat()


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object DeepSeek answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        error = body.get("error") if isinstance(body, dict) else None
        tag = (safe_tag(error.get("type")) or safe_tag(error.get("code"))) if isinstance(error, dict) else None
        raise ProviderError(DeepSeekProvider.METADATA.name, response.status_code, tag or "http_error")
    if not isinstance(body, dict):
        raise ProviderError(DeepSeekProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field DeepSeek documents, or carries one of the wrong type."""
    return ProviderError(DeepSeekProvider.METADATA.name, response.status_code, "unexpected_response")


def _hit(item: Any, response: httpx.Response) -> SearchHit:
    """One result as a hit, from the fields DeepSeek documents on every result."""
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, snippet, content = item.get("url"), item.get("title"), item.get("snippet"), item.get("content")
    if not isinstance(url, str) or not url:
        raise _unexpected(response)
    if not isinstance(title, str) or not isinstance(snippet, str) or not isinstance(content, str):
        raise _unexpected(response)
    if "date" not in item:
        raise _unexpected(response)
    published = None
    if item["date"] is not None:
        published = _published(item["date"])
        if published is None:
            raise _unexpected(response)
    return SearchHit(url=url, title=title, snippet=snippet, text=content or None, published=published, raw=item)


def _published(value: Any) -> datetime | None:
    """DeepSeek's ``date``: an RFC 3339 date-time in UTC. A value without a zone is taken as UTC."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
