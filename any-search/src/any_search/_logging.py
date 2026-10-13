"""Keep provider URLs out of httpx's log.

httpx logs every request's full URL at INFO, and some providers put the query
or the key in that URL. Every provider call runs inside ``provider_call``, and
a filter on the ``httpx`` logger replaces the URL with ``<redacted>`` in the
records emitted while one is in progress. Over HTTP/2, hpack also logs each
header it encodes or decodes at DEBUG, the request's path among them, so the
same filter on hpack's loggers replaces every argument of their records but
the numbers.
Requests made by other code in the process are left alone.

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

# httpx's request line, and over HTTP/2 hpack's header records, which carry the request's path.
LOGGERS = ("httpx", "hpack.hpack", "hpack.table")

_current_provider: ContextVar[str | None] = ContextVar("any_search_provider_call", default=None)


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
    """Replace every URL argument of an httpx record, and every argument of an hpack one, during a provider call."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _current_provider.get() is not None and isinstance(record.args, tuple):
            if record.name == "httpx":
                record.args = tuple(REDACTED if _is_url(arg) else arg for arg in record.args)
            else:
                # Numbers stay, for the records' %d; every header, name or value, goes.
                record.args = tuple(arg if isinstance(arg, int | float) else REDACTED for arg in record.args)
        return True


def install() -> None:
    """Attach the filter to each logger in ``LOGGERS``, once."""
    for name in LOGGERS:
        logger = logging.getLogger(name)
        if not any(isinstance(existing, RedactProviderUrls) for existing in logger.filters):
            logger.addFilter(RedactProviderUrls())
