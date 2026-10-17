"""Web fetch on any-fetch: the fetch backend, and otari's fetcher as the library's ``builtin`` provider."""

from gateway.services.tools._web_fetch._backend import WebFetchBackend, build_web_fetch_backend
from gateway.services.tools._web_fetch._builtin import register_builtin_fetcher, retrieval_client

__all__ = ["WebFetchBackend", "build_web_fetch_backend", "register_builtin_fetcher", "retrieval_client"]
