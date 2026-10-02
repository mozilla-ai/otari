"""Otari's own rate-limit stores: this process's memory, and Redis.

Each satisfies :class:`gateway.ports.rate_limit_store_port.RateLimitStorePort`.
:func:`build_rate_limit_store` picks the one ``rate_limit_store`` selects.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import TYPE_CHECKING

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.ports.rate_limit_store_port import RateLimitStorePort, RateLimitWindow
from gateway.rate_limit import SlidingWindowLog

if TYPE_CHECKING:
    from redis.asyncio import Redis

# A sliding-window log in one sorted set per key, scored by Redis's own clock so
# replicas whose clocks disagree still count one window. Trimming, counting and
# adding run as one script, which is what keeps two replicas from both taking
# the last slot. Returns {allowed, count, milliseconds until the oldest entry
# leaves the window}.
_SLIDING_WINDOW_SCRIPT = """
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
local count = redis.call('ZCARD', KEYS[1])
local allowed = 0
if count < limit then
  redis.call('ZADD', KEYS[1], now, ARGV[3])
  redis.call('PEXPIRE', KEYS[1], window)
  count = count + 1
  allowed = 1
end
local reset = window
local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
if oldest[2] then
  reset = tonumber(oldest[2]) + window - now
end
return {allowed, count, reset}
"""

_KEY_PREFIX = "otari:rl:"
# Bounds what an unreachable Redis adds to a request, and how long the store
# then counts locally before it tries Redis again.
_REDIS_TIMEOUT_SEC = 0.5
_REDIS_RETRY_AFTER_SEC = 5.0


class InMemoryRateLimitStore:
    """Counts in this process, so each replica admits the full limit on its own."""

    def __init__(self) -> None:
        self._log = SlidingWindowLog()

    async def hit(self, key: str, limit: int, window_sec: float) -> RateLimitWindow:
        return self._log.hit(key, limit, window_sec)

    async def aclose(self) -> None:
        return None


class RedisRateLimitStore:
    """Counts in Redis, so a limit holds across every replica that shares it.

    When Redis cannot answer, the store counts in this process instead rather
    than refusing traffic, and tries Redis again after a few seconds. Each
    replica then admits the full limit on its own until Redis is back.
    """

    def __init__(self, client: Redis) -> None:
        self._client = client
        self._script = client.register_script(_SLIDING_WINDOW_SCRIPT)
        self._fallback = InMemoryRateLimitStore()
        self._retry_at: float | None = None

    @classmethod
    def from_url(cls, url: str) -> RedisRateLimitStore:
        from redis.asyncio import Redis

        return cls(Redis.from_url(url, socket_timeout=_REDIS_TIMEOUT_SEC, socket_connect_timeout=_REDIS_TIMEOUT_SEC))

    async def hit(self, key: str, limit: int, window_sec: float) -> RateLimitWindow:
        from redis.exceptions import RedisError

        if self._retry_at is not None and time.monotonic() < self._retry_at:
            return await self._fallback.hit(key, limit, window_sec)
        try:
            allowed, count, reset_ms = await self._script(
                keys=[_KEY_PREFIX + key], args=[limit, math.ceil(window_sec * 1000), uuid.uuid4().hex]
            )
        except (RedisError, OSError) as exc:
            if self._retry_at is None:
                logger.warning("Rate-limit store unreachable, counting per process until it answers: %s", exc)
            self._retry_at = time.monotonic() + _REDIS_RETRY_AFTER_SEC
            return await self._fallback.hit(key, limit, window_sec)
        if self._retry_at is not None:
            logger.info("Rate-limit store reachable again")
            self._retry_at = None
        return RateLimitWindow(allowed=bool(allowed), count=int(count), reset_after=int(reset_ms) / 1000)

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
