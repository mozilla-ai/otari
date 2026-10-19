"""Web search on the libraries: the resolver, the in-loop backend, and the direct endpoint's rules and mapping."""

from gateway.services.tools._web_search._backend import WebSearchBackend
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
from gateway.services.tools._web_search._resolve import (
    InLoopSearch,
    ResolvedFetch,
    ResolvedSearch,
    builtin_fetch_resolved,
    resolve_fetch,
    resolve_in_loop_search,
    resolve_search,
)

__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_MAX_TOKENS_PER_PAGE",
    "DIRECT_DEFAULT_MAX_RESULTS",
    "DirectSearch",
    "DirectSearchResult",
    "InLoopSearch",
    "ResolvedFetch",
    "ResolvedSearch",
    "SearchQuery",
    "WebSearchBackend",
    "builtin_fetch_resolved",
    "filter_hits_by_domain",
    "page_budget_chars",
    "pick_direct_search_instance",
    "resolve_direct_search",
    "resolve_fetch",
    "resolve_in_loop_search",
    "resolve_search",
    "run_direct_search",
]
