"""The httpx log filter, which a host installs, installed per test."""

import logging
from collections.abc import Iterator

import pytest

from any_fetch import install_log_filter
from any_fetch._logging import RedactProviderUrls


@pytest.fixture(autouse=True)
def _remove_the_log_filter() -> Iterator[None]:
    """Remove the filter after every test, so no test depends on another having installed it."""
    yield
    logger = logging.getLogger("httpx")
    for existing in [existing for existing in logger.filters if isinstance(existing, RedactProviderUrls)]:
        logger.removeFilter(existing)


@pytest.fixture
def log_filter() -> None:
    """Install the filter, as a host does at startup."""
    install_log_filter()
