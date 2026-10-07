"""Keep provider URLs out of httpx's log.

httpx logs every request's full URL at INFO, and some providers put the query
or the key in that URL. Every provider call runs inside ``provider_call``, and
a filter on the ``httpx`` logger replaces the URL with ``<redacted>`` in the
records emitted while one is in progress. Requests made by other code in the
process are left alone.

Importing the package installs nothing, so it never changes a process's
logging on its own: the host calls ``install`` (exported as
``install_log_filter``) once at startup, and the command line calls it itself.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import httpx

REDACTED = "<redacted>"

_current_provider: ContextVar[str | None] = ContextVar("any_fetch_provider_call", default=None)


@contextmanager
def provider_call(provider: str) -> Iterator[None]:
    """Mark the code inside as a call to ``provider``, for the filter below."""
    token = _current_provider.set(provider)
    try:
        yield
    finally:
        _current_provider.reset(token)


def _is_url(value: object) -> bool:
    return isinstance(value, httpx.URL) or (isinstance(value, str) and "://" in value)


class RedactProviderUrls(logging.Filter):
    """Replace every URL argument of an httpx record emitted during a provider call."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _current_provider.get() is not None and isinstance(record.args, tuple):
            record.args = tuple(REDACTED if _is_url(arg) else arg for arg in record.args)
        return True


def install() -> None:
    """Attach the filter to the ``httpx`` logger, once."""
    logger = logging.getLogger("httpx")
    if not any(isinstance(existing, RedactProviderUrls) for existing in logger.filters):
        logger.addFilter(RedactProviderUrls())
