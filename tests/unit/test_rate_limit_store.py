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

    async def zrem(self, key: str, member: str) -> int:
        self.calls += 1
        raise RedisConnectionError("connection refused")

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


@pytest.mark.asyncio
async def test_memory_store_counts_what_each_entry_costs() -> None:
    """A tokens-per-minute limit admits an estimate only while the window has room for it."""
    store = InMemoryRateLimitStore()

    first = await store.hit("k", 100, 60, cost=60)
    refused = await store.hit("k", 100, 60, cost=50)
    fits = await store.hit("k", 100, 60, cost=40)

    assert (first.allowed, first.count) == (True, 60)
    assert first.handle is not None
    assert (refused.allowed, refused.count, refused.handle) == (False, 60, None)
    assert (fits.allowed, fits.count) == (True, 100)


@pytest.mark.asyncio
async def test_settling_an_entry_frees_what_its_estimate_overstated() -> None:
    store = InMemoryRateLimitStore()
    estimate = await store.hit("k", 100, 60, cost=90)
    assert estimate.handle is not None

    await store.settle("k", estimate.handle, 30)

    assert (await store.hit("k", 100, 60, cost=70)).allowed
    assert not (await store.hit("k", 100, 60, cost=1)).allowed


@pytest.mark.asyncio
async def test_settling_an_entry_that_left_the_window_changes_nothing() -> None:
    store = InMemoryRateLimitStore()

    with patch("gateway.rate_limit.time") as mock_time:
        mock_time.monotonic.return_value = 1000.0
        estimate = await store.hit("k", 100, 60, cost=10)
        assert estimate.handle is not None

        mock_time.monotonic.return_value = 1061.0
        assert (await store.hit("k", 100, 60, cost=100)).allowed
        await store.settle("k", estimate.handle, 50)

        assert not (await store.hit("k", 100, 60, cost=1)).allowed


@pytest.mark.asyncio
async def test_memory_store_hands_out_at_most_limit_concurrent_slots() -> None:
    store = InMemoryRateLimitStore()

    first = await store.acquire("k", 2, 30)
    second = await store.acquire("k", 2, 30)
    third = await store.acquire("k", 2, 30)
    assert first is not None
    assert second is not None
    assert third is None

    await store.release("k", first)
    assert await store.acquire("k", 2, 30) is not None


@pytest.mark.asyncio
async def test_a_slot_nobody_released_comes_back_when_its_lease_runs_out() -> None:
    """A process that dies holding a slot does not keep it forever."""
    store = InMemoryRateLimitStore()

    with patch("gateway.adapters.rate_limit_store_adapter.time") as mock_time:
        mock_time.monotonic.return_value = 1000.0
        assert await store.acquire("k", 1, 30) is not None
        assert await store.acquire("k", 1, 30) is None

        mock_time.monotonic.return_value = 1031.0
        assert await store.acquire("k", 1, 30) is not None


@pytest.mark.asyncio
async def test_unreachable_redis_hands_out_slots_in_process() -> None:
    store = RedisRateLimitStore(_FailingRedis())  # type: ignore[arg-type]

    lease = await store.acquire("k", 1, 30)
    assert lease is not None
    assert await store.acquire("k", 1, 30) is None

    await store.release("k", lease)
    assert await store.acquire("k", 1, 30) is not None


class _RecoveringRedis(_FailingRedis):
    """A client that fails until ``up`` is set, then answers every script and ``zrem``."""

    def __init__(self) -> None:
        super().__init__()
        self.up = False
        self.zrems: list[str] = []

    def register_script(self, script: str) -> Any:
        failing = super().register_script(script)

        async def run(*, keys: list[str], args: list[Any]) -> Any:
            if not self.up:
                return await failing(keys=keys, args=args)
            return [1, args[3], 1000] if len(keys) == 2 and len(args) == 4 else 1

        return run

    async def zrem(self, key: str, member: str) -> int:
        if not self.up:
            return await super().zrem(key, member)
        self.zrems.append(member)
        return 1


@pytest.mark.asyncio
async def test_what_the_fallback_issued_goes_back_to_it_after_redis_recovers() -> None:
    """A slot taken while Redis was down is not stranded in memory once Redis answers again."""
    client = _RecoveringRedis()
    store = RedisRateLimitStore(client)  # type: ignore[arg-type]
    lease = await store.acquire("k", 1, 30)
    estimate = await store.hit("k", 100, 60, cost=90)
    assert lease is not None
    assert estimate.handle is not None

    client.up = True
    store._retry_at = None
    await store.release("k", lease)
    await store.settle("k", estimate.handle, 30)

    assert client.zrems == []
    assert await store._fallback.acquire("k", 1, 30) is not None
    assert (await store._fallback.hit("k", 100, 60, cost=70)).allowed


@pytest.mark.asyncio
async def test_keys_whose_leases_all_ran_out_are_dropped() -> None:
    store = InMemoryRateLimitStore()

    with patch("gateway.adapters.rate_limit_store_adapter.time") as mock_time:
        mock_time.monotonic.return_value = 1000.0
        assert await store.acquire("abandoned", 1, 30) is not None

        mock_time.monotonic.return_value = 1031.0
        for _ in range(InMemoryRateLimitStore._CLEANUP_INTERVAL - 1):
            assert await store.acquire("busy", 10**6, 60) is not None

    assert set(store._leases) == {"busy"}
