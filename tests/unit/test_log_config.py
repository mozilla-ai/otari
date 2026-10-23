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


@pytest.fixture
def configured_logger() -> Iterator[io.StringIO]:
    """The gateway logger as ``setup_logger`` builds it, printing to a StringIO console."""
    buffer = io.StringIO()
    logger = log_config.logger
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    log_config.setup_logger(console=Console(file=buffer, width=240))
    try:
        yield buffer
    finally:
        logger.handlers = saved_handlers
        logger.setLevel(saved_level)


# Built at runtime so no literal credential sits in the repository.
GITHUB_TOKEN = "gho_" + "A1b2" * 9


def test_a_credential_in_a_log_message_is_masked(configured_logger: io.StringIO) -> None:
    log_config.logger.warning("Provider call failed: echoed %s api_key=sk-live-123456789", GITHUB_TOKEN)

    output = configured_logger.getvalue()
    assert GITHUB_TOKEN not in output
    assert "sk-live-123456789" not in output
    assert output.count("<redacted>") == 2


def test_masking_leaves_what_an_operator_debugs_with(configured_logger: io.StringIO) -> None:
    line = "Routed 3f2c9a1b7e8d4c6f to https://api.openai.com/v1 at 9f8e7d6c5b4a39281706f5e4d3c2b1a098765432"

    log_config.logger.warning(line)

    assert line in configured_logger.getvalue()


def test_a_password_in_a_url_is_masked_and_the_host_kept(configured_logger: io.StringIO) -> None:
    log_config.logger.warning("Connecting to postgresql://otari:hunter2@db.internal:5432/otari")

    output = configured_logger.getvalue()
    assert "hunter2" not in output
    assert "postgresql://<redacted>@db.internal:5432/otari" in output


def test_a_credential_in_a_traceback_is_masked(configured_logger: io.StringIO) -> None:
    try:
        raise RuntimeError(f"upstream said {GITHUB_TOKEN}")
    except RuntimeError:
        log_config.logger.exception("Request failed")

    output = configured_logger.getvalue()
    assert GITHUB_TOKEN not in output
    assert "RuntimeError: upstream said <redacted>" in output


def test_log_secret_still_prints_the_one_time_key(configured_logger: io.StringIO) -> None:
    """The bootstrap key is printed on purpose, past the logger and so past the mask."""
    log_config.log_secret("Save this key now:", LONG_SECRET)

    assert any(line.strip() == LONG_SECRET for line in configured_logger.getvalue().splitlines())


def test_a_record_marked_as_carrying_a_credential_is_left_alone(configured_logger: io.StringIO) -> None:
    """The console mail transport's links are the point of that transport."""
    log_config.logger.warning(
        "Accept: https://otari.example.com/#/accept-invitation?token=abc123", extra=log_config.CARRIES_CREDENTIAL
    )

    assert "accept-invitation?token=abc123" in configured_logger.getvalue()


def test_a_reference_to_where_a_credential_comes_from_is_kept(configured_logger: io.StringIO) -> None:
    log_config.logger.warning("Name the key it reads (api_key: ${OPENAI_API_KEY})")

    assert "api_key: ${OPENAI_API_KEY}" in configured_logger.getvalue()
