"""One interface to web search providers.

from any_search import AnySearch, asearch

result = await asearch("exa", "latest stable python release", max_results=5)
async with AnySearch.create("exa", timeout=15.0) as exa:
    result = await exa.search("latest stable python release", max_results=5, type="auto")
"""

from any_search._api import DEFAULT_TIMEOUT, AnySearch, asearch
from any_search._errors import (
    AnySearchError,
    MissingCredentialError,
    ProviderError,
    UnsupportedParameterError,
    UnsupportedProviderError,
)
from any_search._logging import install as install_log_filter
from any_search._types import OptionSpec, ProviderMetadata, SearchError, SearchHit, SearchResult, TimeRange

__all__ = [
    "DEFAULT_TIMEOUT",
    "AnySearch",
    "AnySearchError",
    "MissingCredentialError",
    "OptionSpec",
    "ProviderError",
    "ProviderMetadata",
    "SearchError",
    "SearchHit",
    "SearchResult",
    "TimeRange",
    "UnsupportedParameterError",
    "UnsupportedProviderError",
    "asearch",
    "install_log_filter",
]
