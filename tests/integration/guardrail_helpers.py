"""Waits shared by the stored-guardrail suites."""

import time
from collections.abc import Callable


def built(knows: Callable[[], bool], *, timeout_s: float = 10.0) -> bool:
    """Wait for the background build a write deliberately does not wait for.

    Returns what ``knows`` last answered, so a negated assertion says "still not
    built after the timeout" rather than raising.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not knows():
        time.sleep(0.05)
    return knows()
