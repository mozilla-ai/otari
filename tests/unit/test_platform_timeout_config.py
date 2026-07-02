"""Validation of the streaming first-chunk timeout settings in ``GatewayConfig``.

The per-attempt failover budgets must be positive; the terminal-attempt extra
grace must be non-negative (it is added on top of the budget). Bad values are
rejected at config-load time rather than silently breaking every request.
"""

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig


def test_valid_streaming_timeouts_are_accepted() -> None:
    config = GatewayConfig(
        platform={
            "streaming_first_chunk_timeout_ms": 2000,
            "streaming_first_chunk_timeout_ms_tool_loop": 30000,
            "streaming_final_attempt_extra_first_chunk_timeout_ms": 58000,
        }
    )
    assert config.platform["streaming_final_attempt_extra_first_chunk_timeout_ms"] == 58000


def test_zero_extra_grace_is_allowed() -> None:
    config = GatewayConfig(platform={"streaming_final_attempt_extra_first_chunk_timeout_ms": 0})
    assert config.platform["streaming_final_attempt_extra_first_chunk_timeout_ms"] == 0


def test_absent_keys_are_accepted() -> None:
    config = GatewayConfig(platform={})
    assert config.platform == {}


@pytest.mark.parametrize("bad_value", [0, -1, -2000])
def test_non_positive_first_chunk_budget_is_rejected(bad_value: int) -> None:
    with pytest.raises(ValidationError, match="streaming_first_chunk_timeout_ms must be > 0"):
        GatewayConfig(platform={"streaming_first_chunk_timeout_ms": bad_value})


@pytest.mark.parametrize("bad_value", [0, -1])
def test_non_positive_tool_loop_budget_is_rejected(bad_value: int) -> None:
    with pytest.raises(ValidationError, match="streaming_first_chunk_timeout_ms_tool_loop must be > 0"):
        GatewayConfig(platform={"streaming_first_chunk_timeout_ms_tool_loop": bad_value})


@pytest.mark.parametrize("bad_value", [-1, -0.5])
def test_negative_extra_grace_is_rejected(bad_value: float) -> None:
    with pytest.raises(ValidationError, match="must be >= 0"):
        GatewayConfig(platform={"streaming_final_attempt_extra_first_chunk_timeout_ms": bad_value})


def test_default_terminal_grace_is_zero() -> None:
    """Unconfigured, the terminal grace resolves to 0.0 so the sole/final attempt
    keeps the historical cap; the setting is opt-in and changes no default."""
    from gateway.api.routes._pipeline import stream_final_attempt_extra_seconds

    assert stream_final_attempt_extra_seconds(GatewayConfig()) == 0.0
