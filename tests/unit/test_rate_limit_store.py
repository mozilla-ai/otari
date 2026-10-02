"""The rate-limit store port: its core adapters, the per-user limiter on top, and the startup checks."""

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from redis.exceptions import ConnectionError as RedisConnectionError

from gateway.adapters.rate_limit_store_adapter import (
    InMemoryRateLimitStore,
    RedisRateLimitStore,
    build_rate_limit_store,
)
from gateway.container import ContainerError, build_container
from gateway.core.config import GatewayConfig
from gateway.main import _validate_rate_limit_store
from gateway.ports.rate_limit_store_port import RateLimitStorePort
from gateway.rate_limit import UserRateLimiter


class _FailingRedis:
    """A client whose every script call fails as an unreachable server does."""

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def register_script(self, script: str) -> Any:
        async def run(*, keys: list[str], args: list[Any]) -> list[int]:
            self.calls += 1
            raise RedisConnectionError("connection refused")

        return run

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_memory_store_admits_up_to_the_limit_then_refuses() -> None:
    store = InMemoryRateLimitStore()

    first = await store.hit("k", 2, 60)
    second = await store.hit("k", 2, 60)
    third = await store.hit("k", 2, 60)

    assert (first.allowed, first.count) == (True, 1)
    assert (second.allowed, second.count) == (True, 2)
    assert (third.allowed, third.count) == (False, 2)
    assert 0 < third.reset_after <= 60


@pytest.mark.asyncio
async def test_memory_store_counts_each_key_on_its_own() -> None:
    store = InMemoryRateLimitStore()

    assert (await store.hit("a", 1, 60)).allowed
    assert (await store.hit("b", 1, 60)).allowed
    assert not (await store.hit("a", 1, 60)).allowed


@pytest.mark.asyncio
async def test_user_limiter_reports_headers_and_refuses_with_retry_after() -> None:
    limiter = UserRateLimiter(InMemoryRateLimitStore(), 2)

    info = await limiter.check("u")
    await limiter.check("u")
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check("u")

    assert (info.limit, info.remaining) == (2, 1)
    assert exc_info.value.status_code == 429
    assert exc_info.value.headers is not None
    assert 1 <= int(exc_info.value.headers["Retry-After"]) <= 60


@pytest.mark.asyncio
async def test_unreachable_redis_counts_in_process_rather_than_refusing() -> None:
    """Traffic keeps flowing, still bounded by the limit this process can count."""
    store = RedisRateLimitStore(_FailingRedis())  # type: ignore[arg-type]

    assert (await store.hit("k", 1, 60)).allowed
    assert not (await store.hit("k", 1, 60)).allowed


@pytest.mark.asyncio
async def test_unreachable_redis_is_not_retried_on_every_request() -> None:
    """An outage costs one timeout per retry interval, not one per request."""
    client = _FailingRedis()
    store = RedisRateLimitStore(client)  # type: ignore[arg-type]

    with patch("gateway.adapters.rate_limit_store_adapter.time") as mock_time:
        mock_time.monotonic.return_value = 1000.0
        await store.hit("k", 10, 60)
        await store.hit("k", 10, 60)
        assert client.calls == 1

        mock_time.monotonic.return_value = 1006.0
        await store.hit("k", 10, 60)
        assert client.calls == 2


@pytest.mark.asyncio
async def test_closing_the_limiter_closes_the_client() -> None:
    client = _FailingRedis()

    await UserRateLimiter(RedisRateLimitStore(client), 1).aclose()  # type: ignore[arg-type]

    assert client.closed


def test_build_picks_the_store_the_config_names() -> None:
    assert isinstance(build_rate_limit_store(GatewayConfig()), InMemoryRateLimitStore)
    redis_config = GatewayConfig(rate_limit_store="redis", rate_limit_redis_url="redis://localhost:6379/0")
    assert isinstance(build_rate_limit_store(redis_config), RedisRateLimitStore)


def test_container_resolves_one_store_for_the_whole_app() -> None:
    container = build_container(config=GatewayConfig())

    store = container.resolve(RateLimitStorePort, None)

    assert isinstance(store, InMemoryRateLimitStore)
    assert container.resolve(RateLimitStorePort, None) is store


def test_container_without_config_refuses_to_pick_a_store() -> None:
    with pytest.raises(ContainerError, match="RateLimitStorePort"):
        build_container().resolve(RateLimitStorePort, None)


def test_config_rejects_an_unknown_store() -> None:
    with pytest.raises(ValueError, match="rate_limit_store"):
        GatewayConfig(rate_limit_store="memcached")  # type: ignore[arg-type]


def test_startup_refuses_redis_without_a_url() -> None:
    with pytest.raises(ValueError, match="rate_limit_redis_url"):
        _validate_rate_limit_store(GatewayConfig(rate_limit_store="redis"))


def test_startup_refuses_redis_without_the_extra() -> None:
    config = GatewayConfig(rate_limit_store="redis", rate_limit_redis_url="redis://localhost:6379/0")

    with (
        patch("gateway.main.importlib.util.find_spec", return_value=None),
        pytest.raises(ValueError, match=r"gateway\[redis\]"),
    ):
        _validate_rate_limit_store(config)


def test_startup_accepts_redis_with_a_url_and_the_extra() -> None:
    _validate_rate_limit_store(GatewayConfig(rate_limit_store="redis", rate_limit_redis_url="redis://localhost:6379/0"))
