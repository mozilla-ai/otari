"""Unit tests for the usage-log latency helper."""

from __future__ import annotations

import time

from gateway.api.routes._pipeline import _elapsed_ms


def test_elapsed_ms_none_start_returns_none() -> None:
    # Write paths with no meaningful request duration pass None, and must
    # record NULL rather than a misleading zero.
    assert _elapsed_ms(None) is None


def test_elapsed_ms_returns_non_negative_int() -> None:
    started = time.monotonic()
    result = _elapsed_ms(started)
    assert isinstance(result, int)
    assert result >= 0


def test_elapsed_ms_grows_with_elapsed_time() -> None:
    # A start further in the past yields a larger elapsed reading.
    now = time.monotonic()
    recent = _elapsed_ms(now)
    older = _elapsed_ms(now - 0.05)  # 50 ms earlier
    assert recent is not None and older is not None
    assert older >= recent
    assert older >= 50
