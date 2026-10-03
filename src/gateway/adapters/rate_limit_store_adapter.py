"""Otari's own rate-limit stores: this process's memory, and Redis.

Each satisfies :class:`gateway.ports.rate_limit_store_port.RateLimitStorePort`.
:func:`build_rate_limit_store` picks the one ``rate_limit_store`` selects.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.ports.rate_limit_store_port import RateLimitStorePort, RateLimitWindow
from gateway.rate_limit import SlidingWindowLog

if TYPE_CHECKING:
    from redis.asyncio import Redis

T = TypeVar("T")

# Every script reads Redis's own clock, so replicas whose clocks disagree still
# count one window, and runs as one step, which is what keeps two replicas from
# both taking the last of a limit.
_NOW_MS = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
"""

# A sliding-window log: KEYS[1] is a sorted set of entry handles scored by when
# they were admitted, KEYS[2] a hash of what each costs plus the running total
# under '#'. An expired entry is subtracted once, as it leaves, so a check
# never sums the window. Returns {allowed, total, milliseconds until the oldest
# entry leaves the window}.
_HIT_SCRIPT = (
    _NOW_MS
    + """
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local cost = tonumber(ARGV[4])
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now - window)
if #expired > 0 then
  local freed = 0
  for _, handle in ipairs(expired) do
    freed = freed + (tonumber(redis.call('HGET', KEYS[2], handle)) or 0)
    redis.call('HDEL', KEYS[2], handle)
  end
  redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
  redis.call('HINCRBY', KEYS[2], '#', -freed)
end
local total = tonumber(redis.call('HGET', KEYS[2], '#')) or 0
local allowed = 0
if total + cost <= limit then
  redis.call('ZADD', KEYS[1], now, ARGV[3])
  redis.call('HSET', KEYS[2], ARGV[3], cost)
  total = redis.call('HINCRBY', KEYS[2], '#', cost)
  redis.call('PEXPIRE', KEYS[1], window)
  redis.call('PEXPIRE', KEYS[2], window)
  allowed = 1
end
local reset = window
local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
if oldest[2] then
  reset = tonumber(oldest[2]) + window - now
end
return {allowed, total, reset}
"""
)

_SETTLE_SCRIPT = """
local old = redis.call('HGET', KEYS[2], ARGV[1])
if not old then
  return 0
end
redis.call('HSET', KEYS[2], ARGV[1], ARGV[2])
redis.call('HINCRBY', KEYS[2], '#', tonumber(ARGV[2]) - tonumber(old))
return 1
"""

# Concurrency slots: a sorted set of leases scored by when each runs out.
_ACQUIRE_SCRIPT = (
    _NOW_MS
    + """
local limit = tonumber(ARGV[1])
local lease = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) >= limit then
  return 0
end
redis.call('ZADD', KEYS[1], now + lease, ARGV[3])
if redis.call('PTTL', KEYS[1]) < lease then
  redis.call('PEXPIRE', KEYS[1], lease)
end
return 1
"""
)

_KEY_PREFIX = "otari:rl:"
# Bounds what an unreachable Redis adds to a request, and how long the store
# then counts locally before it tries Redis again.
_REDIS_TIMEOUT_SEC = 0.5
_REDIS_RETRY_AFTER_SEC = 5.0
# Marks a handle or lease the in-process fallback issued, so it is settled or
# released there even after Redis answers again.
_LOCAL_PREFIX = "local:"


class InMemoryRateLimitStore:
    """Counts in this process, so each replica admits the full limit on its own."""

    def __init__(self) -> None:
        self._log = SlidingWindowLog()
        self._leases: defaultdict[str, dict[str, float]] = defaultdict(dict)

    async def hit(self, key: str, limit: int, window_sec: float, cost: int = 1) -> RateLimitWindow:
        return self._log.hit(key, limit, window_sec, cost)

    async def settle(self, key: str, handle: str, cost: int) -> None:
        self._log.settle(key, handle, cost)

    async def acquire(self, key: str, limit: int, lease_sec: float) -> str | None:
        now = time.monotonic()
        slots = self._leases[key]
        for lease in [lease for lease, expires_at in slots.items() if expires_at <= now]:
            del slots[lease]
        if len(slots) >= limit:
            return None
        lease = uuid.uuid4().hex
        slots[lease] = now + lease_sec
        return lease

    async def release(self, key: str, lease: str) -> None:
        slots = self._leases.get(key)
        if slots is None:
            return
        slots.pop(lease, None)
        if not slots:
            del self._leases[key]

    async def aclose(self) -> None:
        return None


class RedisRateLimitStore:
    """Counts in Redis, so a limit holds across every replica that shares it.

    When Redis cannot answer, the store counts in this process instead rather
    than refusing traffic, and tries Redis again after a few seconds. Each
    replica then admits the full limit on its own until Redis is back.

    A handle or lease goes back to the store that issued it. One Redis issued
    that cannot reach Redis is left to run out on its own.
    """

    def __init__(self, client: Redis) -> None:
        self._client = client
        self._hit = client.register_script(_HIT_SCRIPT)
        self._settle = client.register_script(_SETTLE_SCRIPT)
        self._acquire = client.register_script(_ACQUIRE_SCRIPT)
        self._fallback = InMemoryRateLimitStore()
        self._retry_at: float | None = None

    @classmethod
    def from_url(cls, url: str) -> RedisRateLimitStore:
        from redis.asyncio import Redis

        return cls(Redis.from_url(url, socket_timeout=_REDIS_TIMEOUT_SEC, socket_connect_timeout=_REDIS_TIMEOUT_SEC))

    @staticmethod
    def _keys(key: str) -> tuple[str, str, str]:
        """The log, cost and lease keys for ``key``, hash-tagged so a Redis Cluster keeps them on one node."""
        base = f"{_KEY_PREFIX}{{{key}}}"
        return f"{base}:log", f"{base}:cost", f"{base}:leases"

    async def _noop(self) -> None:
        return None

    async def _redis_or_fallback(self, call: Callable[[], Awaitable[T]], fallback: Callable[[], Awaitable[T]]) -> T:
        from redis.exceptions import RedisError

        if self._retry_at is not None and time.monotonic() < self._retry_at:
            return await fallback()
        try:
            result = await call()
        except (RedisError, OSError) as exc:
            if self._retry_at is None:
                logger.warning("Rate-limit store unreachable, counting per process until it answers: %s", exc)
            self._retry_at = time.monotonic() + _REDIS_RETRY_AFTER_SEC
            return await fallback()
        if self._retry_at is not None:
            logger.info("Rate-limit store reachable again")
            self._retry_at = None
        return result

    async def hit(self, key: str, limit: int, window_sec: float, cost: int = 1) -> RateLimitWindow:
        log_key, cost_key, _ = self._keys(key)
        handle = uuid.uuid4().hex

        async def call() -> RateLimitWindow:
            allowed, total, reset_ms = await self._hit(
                keys=[log_key, cost_key], args=[limit, math.ceil(window_sec * 1000), handle, cost]
            )
            return RateLimitWindow(
                allowed=bool(allowed),
                count=int(total),
                reset_after=int(reset_ms) / 1000,
                handle=handle if allowed else None,
            )

        async def fallback() -> RateLimitWindow:
            window = await self._fallback.hit(key, limit, window_sec, cost)
            return replace(window, handle=window.handle and _LOCAL_PREFIX + window.handle)

        return await self._redis_or_fallback(call, fallback)

    async def settle(self, key: str, handle: str, cost: int) -> None:
        if handle.startswith(_LOCAL_PREFIX):
            await self._fallback.settle(key, handle.removeprefix(_LOCAL_PREFIX), cost)
            return
        log_key, cost_key, _ = self._keys(key)

        async def call() -> None:
            await self._settle(keys=[log_key, cost_key], args=[handle, cost])

        await self._redis_or_fallback(call, self._noop)

    async def acquire(self, key: str, limit: int, lease_sec: float) -> str | None:
        _, _, lease_key = self._keys(key)
        lease = uuid.uuid4().hex

        async def call() -> str | None:
            taken = await self._acquire(keys=[lease_key], args=[limit, math.ceil(lease_sec * 1000), lease])
            return lease if taken else None

        async def fallback() -> str | None:
            local = await self._fallback.acquire(key, limit, lease_sec)
            return local and _LOCAL_PREFIX + local

        return await self._redis_or_fallback(call, fallback)

    async def release(self, key: str, lease: str) -> None:
        if lease.startswith(_LOCAL_PREFIX):
            await self._fallback.release(key, lease.removeprefix(_LOCAL_PREFIX))
            return
        _, _, lease_key = self._keys(key)

        async def call() -> None:
            await self._client.zrem(lease_key, lease)

        await self._redis_or_fallback(call, self._noop)

    async def aclose(self) -> None:
        await self._client.aclose()


def build_rate_limit_store(config: GatewayConfig) -> RateLimitStorePort:
    """The store ``rate_limit_store`` selects."""
    if config.rate_limit_store == "redis":
        if not config.rate_limit_redis_url:
            msg = "rate_limit_store is 'redis' but rate_limit_redis_url is not set"
            raise ValueError(msg)
        return RedisRateLimitStore.from_url(config.rate_limit_redis_url)
    return InMemoryRateLimitStore()
