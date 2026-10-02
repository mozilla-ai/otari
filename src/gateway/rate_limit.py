"""Sliding-window rate limiting: in-process limiters and the per-user limit counted in a rate-limit store."""

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from gateway.metrics import REGISTRY, Counter
from gateway.ports.rate_limit_store_port import RateLimitStorePort, RateLimitWindow

RATE_LIMIT_HITS = Counter(
    "gateway_rate_limit_hits",
    "Total number of rate limit hits",
    registry=REGISTRY,
)


@dataclass
class RateLimitInfo:
    """Rate limit status returned by a successful check."""

    limit: int
    remaining: int
    reset: float


class SlidingWindowLog:
    """Request timestamps per key, held in this process.

    Exact: a request is admitted when fewer than ``limit`` were admitted in the
    last ``window_sec``. Keys idle for longer than the widest window seen are
    dropped every ``_CLEANUP_INTERVAL`` hits, so the map is bounded by the keys
    that are active.
    """

    _CLEANUP_INTERVAL = 1000

    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._call_count = 0
        self._widest_window = 0.0

    def hit(self, key: str, limit: int, window_sec: float) -> RateLimitWindow:
        """Count one request against ``key`` if it fits under ``limit``."""
        now = time.monotonic()
        cutoff = now - window_sec
        self._widest_window = max(self._widest_window, window_sec)

        self._call_count += 1
        if self._call_count >= self._CLEANUP_INTERVAL:
            self._cleanup(now - self._widest_window)
            self._call_count = 0

        timestamps = self._requests[key]
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

        allowed = len(timestamps) < limit
        if allowed:
            timestamps.append(now)
        return RateLimitWindow(allowed=allowed, count=len(timestamps), reset_after=timestamps[0] + window_sec - now)

    def _cleanup(self, cutoff: float) -> None:
        """Remove entries for keys with no recent requests."""
        stale = [key for key, ts in self._requests.items() if not ts or ts[-1] <= cutoff]
        for key in stale:
            del self._requests[key]


def _info_or_raise(window: RateLimitWindow, limit: int) -> RateLimitInfo:
    """The headers' view of an admitted request, or the 429 for a refused one."""
    if not window.allowed:
        RATE_LIMIT_HITS.inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(math.ceil(window.reset_after))},
        )
    # Wall-clock time for the externally facing reset header.
    return RateLimitInfo(limit=limit, remaining=limit - window.count, reset=time.time() + window.reset_after)


class RateLimiter:
    """A sliding-window limit counted in this process.

    For the limits that guard one process's own surfaces (sign-in, feedback,
    the public catalog). ``window_sec`` widens the window for a limit counted
    over longer than a minute; ``rpm`` is then the allowance per window.
    """

    def __init__(self, rpm: int, *, window_sec: float = 60.0) -> None:
        self._rpm = rpm
        self._window_sec = window_sec
        self._log = SlidingWindowLog()

    def check(self, user_id: str) -> RateLimitInfo:
        """Check whether a request is allowed for the given key.

        Raises:
            HTTPException: 429 if the rate limit has been exceeded

        """
        return _info_or_raise(self._log.hit(user_id, self._rpm, self._window_sec), self._rpm)


class UserRateLimiter:
    """The per-user limit (``rate_limit_rpm``), counted in the store this build bound.

    With a shared store the limit holds for the deployment, however many
    replicas serve it.
    """

    def __init__(self, store: RateLimitStorePort, rpm: int, *, window_sec: float = 60.0) -> None:
        self._store = store
        self._rpm = rpm
        self._window_sec = window_sec

    async def check(self, user_id: str) -> RateLimitInfo:
        """Count one request for ``user_id``.

        Raises:
            HTTPException: 429 if the rate limit has been exceeded

        """
        window = await self._store.hit(f"user:{user_id}", self._rpm, self._window_sec)
        return _info_or_raise(window, self._rpm)

    async def aclose(self) -> None:
        """Release the store's connection."""
        await self._store.aclose()


async def check_rate_limit(request: Request, user_id: str) -> RateLimitInfo | None:
    """Check rate limit for a user, returning info for header injection.

    Returns RateLimitInfo when rate limiting is active, None when disabled.
    """
    rate_limiter: UserRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter is None:
        return None
    return await rate_limiter.check(user_id)
