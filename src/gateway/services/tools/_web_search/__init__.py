"""Web search on the libraries. So far the resolver: which provider, key and options a search or fetch runs with."""

from gateway.services.tools._web_search._resolve import ResolvedFetch, ResolvedSearch, resolve_fetch, resolve_search

__all__ = ["ResolvedFetch", "ResolvedSearch", "resolve_fetch", "resolve_search"]
