"""Where a rate limit's counts are kept.

One store counts in the process it runs in, so each replica of a deployment
admits the full limit on its own. Another keeps the counts where every replica
reads them, so the limit holds for the deployment as a whole. A caller asks the
same question of either.

Two kinds of limit are counted. A window limit bounds what is admitted over
time: requests per minute, or tokens per minute when each entry costs what a
request is expected to use. A concurrency limit bounds what is in flight at once.

The store counts and decides; what a refusal looks like to a client is the
caller's to say.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RateLimitWindow:
    """What a store answered for one entry against one window limit."""

    allowed: bool
    """Whether the entry fit, in which case it is now counted."""
    count: int
    """What the entries in the window cost, this one included when it was allowed."""
    reset_after: float
    """Seconds until the oldest counted entry leaves the window."""
    handle: str | None = None
    """The admitted entry, for :meth:`RateLimitStorePort.settle`; ``None`` when refused."""


class RateLimitStorePort(Protocol):
    """Sliding-window counts and concurrency slots, keyed by whatever a limit applies to."""

    async def hit(self, key: str, limit: int, window_sec: float, cost: int = 1) -> RateLimitWindow:
        """Count an entry costing ``cost`` against ``key`` if the window stays within ``limit`` with it.

        Checking and counting are one step, so two callers cannot both take the last of the limit.
        """
        ...

    async def settle(self, key: str, handle: str, cost: int) -> None:
        """Make an admitted entry count for ``cost`` instead.

        For a limit counted in units known only afterwards: a request is admitted
        on an estimate of its tokens and settled on what it used. Never refuses,
        and does nothing once the entry has left the window.
        """
        ...

    async def acquire(self, key: str, limit: int, lease_sec: float) -> str | None:
        """Take one of ``limit`` concurrent slots on ``key``, or ``None`` when all are taken.

        The slot is held until :meth:`release`, or until ``lease_sec`` passes,
        so a process that dies holding one does not keep it forever.
        """
        ...

    async def release(self, key: str, lease: str) -> None:
        """Give back a slot :meth:`acquire` returned; a no-op once its lease has run out."""
        ...

    async def aclose(self) -> None:
        """Release whatever connection the store holds."""
        ...


__all__ = ["RateLimitStorePort", "RateLimitWindow"]
