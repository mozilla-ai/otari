"""Where a rate limit's counts are kept.

One store counts in the process it runs in, so each replica of a deployment
admits the full limit on its own. Another keeps the counts where every replica
reads them, so the limit holds for the deployment as a whole. A caller asks the
same question of either.

The store counts and decides; what a refusal looks like to a client is the
caller's to say.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RateLimitWindow:
    """What a store answered for one request against one limit."""

    allowed: bool
    """Whether the request fit, in which case it is now counted."""
    count: int
    """Requests counted in the window, this one included when it was allowed."""
    reset_after: float
    """Seconds until the oldest counted request leaves the window."""


class RateLimitStorePort(Protocol):
    """Sliding-window request counts, keyed by whatever a limit applies to."""

    async def hit(self, key: str, limit: int, window_sec: float) -> RateLimitWindow:
        """Count one request against ``key`` if fewer than ``limit`` fall within the last ``window_sec``.

        Checking and counting are one step, so two callers cannot both take the last slot.
        """
        ...

    async def aclose(self) -> None:
        """Release whatever connection the store holds."""
        ...


__all__ = ["RateLimitStorePort", "RateLimitWindow"]
