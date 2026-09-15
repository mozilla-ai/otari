"""Compatibility imports for the renamed web retrieval backend."""

from gateway.services.web_retrieval_backend import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_CAP,
    WEB_RETRIEVAL_RESULT_MAX_BYTES,
    WEB_SEARCH_TOOL_NAME,
    WebRetrievalBackend,
    WebSearchBackend,
    WebSearchNotReachableError,
    web_search_tool_definition,
)

__all__ = [
    "DEFAULT_MAX_RESULTS",
    "MAX_RESULTS_CAP",
    "WEB_RETRIEVAL_RESULT_MAX_BYTES",
    "WEB_SEARCH_TOOL_NAME",
    "WebRetrievalBackend",
    "WebSearchBackend",
    "WebSearchNotReachableError",
    "web_search_tool_definition",
]
