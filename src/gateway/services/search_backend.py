"""The direct search endpoint's old client, for the organization keys that do not run on any-search.

The direct search endpoint runs on any-search (``gateway.services.tools``), except
a search on a workspace's Tavily or Brave organization key (#2015), which
any-search has no adapter for: it runs here, through the provider mapping the
in-loop backend uses. It reads the request in the gateway's provider-neutral terms
(``SearchQuery``) and sends through the client its caller passes, the app's web
provider client. It keeps its own ceiling of 20 results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from gateway.services.tools import (
    DIRECT_DEFAULT_MAX_RESULTS,
    SearchQuery,
    filter_hits_by_domain,
    page_budget_chars,
)
from gateway.services.web_search_providers import WebSearchProviderError, provider_search

if TYPE_CHECKING:
    from gateway.models.tools import WebSearchCredential

# The most results this client returns (decision 65): Tavily and Brave stop at 20 anyway.
MAX_RESULTS_CAP = 20


class SearchProviderError(RuntimeError):
    """The search provider could not be reached or returned malformed data.

    ``tag`` names the failure in the search library's terms (``timeout``,
    ``network``, ``http_error``, ``invalid_response``), and ``status`` is the
    provider's HTTP status when it answered with one, for a caller that reports
    the failure without the message, which can carry the upstream body.
    """

    def __init__(self, message: str, *, tag: str = "provider_error", status: int | None = None) -> None:
        """Carry the message for the log, and the tag and status for a caller that reports only those."""
        super().__init__(message)
        self.tag = tag
        self.status = status


@dataclass(frozen=True)
class SearchHit:
    """One normalized result."""

    url: str
    title: str | None = None
    snippet: str | None = None
    date: str | None = None


@dataclass(frozen=True)
class SearchOutcome:
    """A completed provider search. The provider bills the key's owner, so ``cost_usd`` stays ``None``."""

    results: list[SearchHit]
    cost_usd: float | None = None


async def run_keyed_search(
    credential: WebSearchCredential, query: SearchQuery, *, timeout_s: float, client: httpx.AsyncClient
) -> SearchOutcome:
    """Dispatch one search on a workspace's own provider key, in place of the named tool's backend.

    ``timeout_s`` is the named tool's, so the key changes who pays and not how long a search may take.
    The provider bills the key's owner directly, so no cost is reported.
    """
    limit = _result_limit(query)
    options: dict[str, Any] = {"max_results": limit}
    if query.country:
        options["country"] = query.country
    try:
        raw = await provider_search(
            provider=credential.provider,
            api_key=credential.api_key or "",
            query=query.query,
            options=options,
            client=client,
            timeout_s=timeout_s,
        )
    except WebSearchProviderError as exc:
        msg = f"{credential.provider} search request failed: {exc}"
        raise SearchProviderError(msg, tag=exc.tag, status=exc.status) from exc
    except ValueError as exc:
        msg = f"{credential.provider} search request failed: {exc}"
        raise SearchProviderError(msg) from exc
    hits = filter_hits_by_domain(_mapped_hits(raw, page_budget_chars(query)), query.domain_filter)
    return SearchOutcome(results=hits[:limit], cost_usd=None)


def _result_limit(query: SearchQuery) -> int:
    """How many hits a search returns, under this client's own ceiling."""
    return max(1, min(query.max_results or DIRECT_DEFAULT_MAX_RESULTS, MAX_RESULTS_CAP))


def _mapped_hits(raw: list[dict[str, Any]], max_chars: int) -> list[SearchHit]:
    """The provider mapping's results as hits, each snippet cut to the caller's per-page budget.

    The snippet is ``extracted_content``, the page text the mapping sets when
    the provider returned it, else ``content``, the provider's own snippet; the
    date is the mapping's ``published_date``.
    """
    hits: list[SearchHit] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        snippet = _text_or_none(item.get("extracted_content")) or _text_or_none(item.get("content"))
        hits.append(
            SearchHit(
                url=url,
                title=_text_or_none(item.get("title")),
                snippet=snippet[:max_chars] if snippet else None,
                date=_text_or_none(item.get("published_date")),
            )
        )
    return hits


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
