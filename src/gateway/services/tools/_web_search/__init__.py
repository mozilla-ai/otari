"""Web search on the libraries: the resolver, and the direct endpoint's rules and LiteLLM mapping."""

from gateway.services.tools._web_search._direct import (
    DirectSearch,
    DirectSearchResult,
    pick_direct_search_instance,
    resolve_direct_search,
    run_direct_search,
)
from gateway.services.tools._web_search._litellm import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_TOKENS_PER_PAGE,
    DIRECT_DEFAULT_MAX_RESULTS,
    SearchQuery,
    filter_hits_by_domain,
    page_budget_chars,
)
from gateway.services.tools._web_search._resolve import ResolvedFetch, ResolvedSearch, resolve_fetch, resolve_search

__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_MAX_TOKENS_PER_PAGE",
    "DIRECT_DEFAULT_MAX_RESULTS",
    "DirectSearch",
    "DirectSearchResult",
    "ResolvedFetch",
    "ResolvedSearch",
    "SearchQuery",
    "filter_hits_by_domain",
    "page_budget_chars",
    "pick_direct_search_instance",
    "resolve_direct_search",
    "resolve_fetch",
    "resolve_search",
    "run_direct_search",
]
