"""Web search on the libraries: the in-loop default, the resolver, the in-loop backend, and the direct endpoint."""

from gateway.services.tools._web_search._backend import WebSearchBackend
from gateway.services.tools._web_search._default import (
    in_loop_default,
    search_default_to_pin,
    warn_about_tool_instances,
    web_search_configured,
)
from gateway.services.tools._web_search._direct import (
    DirectSearch,
    DirectSearchResult,
    pick_direct_search_instance,
    refuse_unpriced_direct_search,
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
    in_loop_search_for,
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
    "in_loop_default",
    "in_loop_search_for",
    "page_budget_chars",
    "pick_direct_search_instance",
    "refuse_unpriced_direct_search",
    "resolve_direct_search",
    "resolve_fetch",
    "resolve_in_loop_search",
    "resolve_search",
    "run_direct_search",
    "search_default_to_pin",
    "warn_about_tool_instances",
    "web_search_configured",
]
