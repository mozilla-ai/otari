"""Web fetch on any-fetch: the fetch backend, one page fetched on an instance, and otari's fetcher as ``builtin``."""

from gateway.services.tools._web_fetch._backend import WebFetchBackend
from gateway.services.tools._web_fetch._builtin import register_builtin_fetcher, retrieval_client
from gateway.services.tools._web_fetch._fetch_page import FetchFailure, fetch_page, refused_url

__all__ = [
    "FetchFailure",
    "WebFetchBackend",
    "fetch_page",
    "refused_url",
    "register_builtin_fetcher",
    "retrieval_client",
]
