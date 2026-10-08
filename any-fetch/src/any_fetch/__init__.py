"""One interface to web fetch providers.

from any_fetch import AnyFetch, afetch

page = await afetch("fake", "https://www.python.org/downloads/")
async with AnyFetch.create("builtin") as fetcher:  # once the host has registered it
    page = await fetcher.fetch("https://www.python.org/downloads/", max_chars=20_000)
"""

from any_fetch._api import DEFAULT_TIMEOUT, AnyFetch, afetch
from any_fetch._errors import (
    AnyFetchError,
    BuiltinNotRegisteredError,
    MissingCredentialError,
    ProviderError,
    UnsupportedParameterError,
    UnsupportedProviderError,
)
from any_fetch._logging import install as install_log_filter
from any_fetch._types import (
    BuiltinFactory,
    BuiltinFetcher,
    FetchedPage,
    FetchError,
    OptionSpec,
    ProviderMetadata,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "AnyFetch",
    "AnyFetchError",
    "BuiltinFactory",
    "BuiltinFetcher",
    "BuiltinNotRegisteredError",
    "FetchError",
    "FetchedPage",
    "MissingCredentialError",
    "OptionSpec",
    "ProviderError",
    "ProviderMetadata",
    "UnsupportedParameterError",
    "UnsupportedProviderError",
    "afetch",
    "install_log_filter",
]
