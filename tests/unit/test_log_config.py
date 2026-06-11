import io
import logging
from collections.abc import Iterator

import pytest
from rich.console import Console
from rich.logging import RichHandler

from gateway import log_config

# A token long enough that RichHandler's narrow message column would hard-wrap it.
LONG_SECRET = "gw-0cfzM-l0TufXyaDVCVE8J-P4dIgCle_lF8SVsF45Ty46Os3djCG8RfmLq8O-Mqwl"


@pytest.fixture
def captured_logger() -> Iterator[io.StringIO]:
    """Point the gateway logger at a fixed-width StringIO console (mimics container logs)."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=80)
    logger = log_config.logger
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    logger.handlers = [RichHandler(console=console, markup=True)]
    logger.setLevel(logging.WARNING)
    try:
        yield buffer
    finally:
        logger.handlers = saved_handlers
        logger.setLevel(saved_level)


def test_log_secret_keeps_value_on_single_unbroken_line(captured_logger: io.StringIO) -> None:
    log_config.log_secret("Save this key now:", LONG_SECRET)

    output = captured_logger.getvalue()
    # The whole secret must appear verbatim (no hard-wrap newlines inserted mid-token).
    assert LONG_SECRET in output
    # And it must occupy its own line with nothing else on it, so it copies cleanly.
    assert any(line.strip() == LONG_SECRET for line in output.splitlines())


def test_log_secret_emits_context_message(captured_logger: io.StringIO) -> None:
    log_config.log_secret("Save this key now:", LONG_SECRET)

    assert "Save this key now:" in captured_logger.getvalue()


def test_log_secret_respects_log_level(captured_logger: io.StringIO) -> None:
    log_config.logger.setLevel(logging.ERROR)

    log_config.log_secret("Save this key now:", LONG_SECRET, level=logging.INFO)

    assert captured_logger.getvalue() == ""
