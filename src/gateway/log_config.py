import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger("gateway")


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

    if log_format:
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

    logger.addHandler(handler)


setup_logger()
