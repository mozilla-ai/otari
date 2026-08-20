"""Validation of the observation transport settings in ``GatewayConfig``.

The four knobs size the bounded queue the observation shipper flushes from.
Each has to be positive: a zero or negative bound would make the queue reject
every record, a zero batch would flush nothing forever, and a non-positive
interval or timeout would spin or abandon every flush. Rejecting them at
config-load time keeps a typo from turning the stream silently empty, which is
indistinguishable from "nothing was observable" in the report the stream feeds.
"""

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig


def test_valid_observation_settings_are_accepted() -> None:
    config = GatewayConfig(
        platform={
            "observation_max_queue": 20000,
            "observation_max_batch": 250,
            "observation_flush_interval_ms": 2000,
            "observation_timeout_ms": 3000,
        }
    )

    assert config.platform["observation_max_queue"] == 20000
    assert config.platform["observation_max_batch"] == 250
    assert config.platform["observation_flush_interval_ms"] == 2000
    assert config.platform["observation_timeout_ms"] == 3000


def test_absent_observation_settings_are_accepted() -> None:
    """Unset is the normal case: the shipper owns the defaults."""
    config = GatewayConfig(platform={"base_url": "http://platform/api/v1"})

    assert "observation_max_queue" not in config.platform


@pytest.mark.parametrize(
    "key",
    [
        "observation_max_queue",
        "observation_max_batch",
        "observation_flush_interval_ms",
        "observation_timeout_ms",
    ],
)
@pytest.mark.parametrize("bad_value", [0, -1, -5000])
def test_non_positive_observation_setting_is_rejected(key: str, bad_value: int) -> None:
    with pytest.raises(ValidationError, match=f"{key} must be > 0"):
        GatewayConfig(platform={key: bad_value})


@pytest.mark.parametrize("key", ["observation_max_queue", "observation_max_batch"])
@pytest.mark.parametrize("bad_value", [0.5, 0.9, 250.5])
def test_a_fractional_record_count_is_rejected(key: str, bad_value: float) -> None:
    """The shipper sizes a queue and a batch with these, via ``int()``.

    A fraction below 1 truncates to 0, and ``asyncio.Queue(maxsize=0)`` is
    unbounded, so a bound of 0.5 would accept records without limit: the one
    property the transport is built around, inverted by a typo.
    """
    with pytest.raises(ValidationError, match=f"{key} must be a whole number of records"):
        GatewayConfig(platform={key: bad_value})


@pytest.mark.parametrize(
    "key",
    [
        "observation_max_queue",
        "observation_max_batch",
        "observation_flush_interval_ms",
        "observation_timeout_ms",
    ],
)
def test_an_empty_observation_setting_is_rejected_by_key(key: str) -> None:
    """An empty YAML value (``observation_timeout_ms:``) parses to None.

    ``float(None)`` raises TypeError, which pydantic does not convert into a
    ValidationError, so this used to escape config load as a bare traceback that
    named no field.
    """
    with pytest.raises(ValidationError, match=f"{key} must be a number"):
        GatewayConfig(platform={key: None})


def test_a_bad_observation_setting_does_not_mask_a_streaming_one() -> None:
    """One validator covers both key sets, so neither shadows the other."""
    with pytest.raises(ValidationError, match="streaming_first_chunk_timeout_ms must be > 0"):
        GatewayConfig(platform={"streaming_first_chunk_timeout_ms": 0, "observation_max_batch": 100})
