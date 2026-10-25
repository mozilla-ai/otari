import copy
import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from gateway.secret_shapes import mask_credentials

logger = logging.getLogger("gateway")

# Not ``[redacted]``: the handler renders Rich markup, which would read that as
# a style tag and print nothing.
_LOG_PLACEHOLDER = "<redacted>"

# Pass as ``extra`` on a record that prints a credential on purpose (the console
# mail transport's links), which the filter then leaves alone.
CARRIES_CREDENTIAL: dict[str, object] = {"carries_credential": True}


class CredentialMaskingFilter(logging.Filter):
    """Mask credentials in a record's message and traceback before any handler prints them.

    A provider's error can echo the request back, credentials included, and the
    gateway logs provider errors verbatim. A traceback keeps its rich rendering
    unless masking changed its text.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "carries_credential", False):
            return True
        message = record.getMessage()
        masked = mask_credentials(message, _LOG_PLACEHOLDER)
        if masked != message and not _mask_args(record):
            record.msg, record.args = masked, None
        if record.exc_info and record.exc_info[1] is not None:
            text = logging.Formatter().formatException(record.exc_info)
            masked_text = mask_credentials(text, _LOG_PLACEHOLDER)
            if masked_text != text:
                record.exc_info, record.exc_text = None, masked_text
        return True


def _mask_args(record: logging.LogRecord) -> bool:
    """Mask a record's string arguments in place, when that alone masks its message.

    Keeps the arguments a formatter unpacks: uvicorn's access log reads the
    client, method and path back out of ``record.args``.
    """
    if not isinstance(record.msg, str) or not isinstance(record.args, tuple):
        return False
    args = tuple(mask_credentials(arg, _LOG_PLACEHOLDER) if isinstance(arg, str) else arg for arg in record.args)
    try:
        message = record.msg % args
    except (TypeError, ValueError):
        return False
    if mask_credentials(message, _LOG_PLACEHOLDER) != message:
        return False
    record.args = args
    return True


def uvicorn_log_config() -> dict[str, Any]:
    """uvicorn's own logging config, with its handlers masking credentials as the gateway's does.

    uvicorn's loggers print through handlers of their own, so the gateway
    handler's filter never sees them, and an access log line carries the query
    string.
    """
    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    config.setdefault("filters", {})["mask_credentials"] = {"()": CredentialMaskingFilter}
    for handler in config["handlers"].values():
        handler.setdefault("filters", []).append("mask_credentials")
    return config


def _log_console() -> Console:
    """Return the console used by the active RichHandler, or a stderr fallback."""
    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            return handler.console
    return Console(stderr=True)


def log_secret(message: str, secret: str, *, level: int = logging.WARNING) -> None:
    """Log a message, then emit a secret value on its own unbroken line.

    RichHandler hard-wraps long tokens within its narrow message column, which
    splits a value like an API key across multiple physical lines and makes it
    impossible to copy reliably from container logs. We log the contextual
    message normally, then write the secret through a soft-wrapping console so
    it stays on a single line that can be copied verbatim.
    """
    logger.log(level, message)
    if not logger.isEnabledFor(level):
        return
    _log_console().print(secret, soft_wrap=True, crop=False, highlight=False, markup=False)


def setup_logger(
    level: int = logging.WARNING,
    rich_tracebacks: bool = True,
    log_format: str | None = None,
    propagate: bool = False,
    **kwargs: Any,
) -> None:
    """Configure the gateway logger with the specified settings.

    Args:
        level: The logging level to use (default: logging.INFO)
        rich_tracebacks: Whether to enable rich tracebacks (default: True)
        log_format: Optional custom log format string
        propagate: Whether to propagate logs to parent loggers (default: False)
        **kwargs: Additional keyword arguments to pass to RichHandler

    """
    logger.setLevel(level)
    logger.propagate = propagate

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = RichHandler(rich_tracebacks=rich_tracebacks, markup=True, **kwargs)
    handler.addFilter(CredentialMaskingFilter())

    if log_format:
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

    logger.addHandler(handler)

    # A library logger with no handler of its own (any-llm, httpx) prints through the last resort.
    last_resort = logging.lastResort
    if last_resort is not None and not any(isinstance(f, CredentialMaskingFilter) for f in last_resort.filters):
        last_resort.addFilter(CredentialMaskingFilter())


setup_logger()
