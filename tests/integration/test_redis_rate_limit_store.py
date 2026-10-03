"""The Redis rate-limit store against a real Redis: one count shared by every replica."""

import asyncio
import uuid
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from testcontainers.redis import RedisContainer

from gateway.adapters.rate_limit_store_adapter import RedisRateLimitStore
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.rate_limit import UserRateLimiter

from .conftest import build_test_client


@pytest.fixture(scope="module")
def redis_url() -> Generator[str]:
    with RedisContainer("redis:7") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


def _replica(redis_url: str) -> RedisRateLimitStore:
    """A store with a client of its own, as each gateway process has."""
    return RedisRateLimitStore(Redis.from_url(redis_url))


@pytest.mark.asyncio
async def test_two_replicas_share_one_limit(redis_url: str) -> None:
    key = f"user:{uuid.uuid4()}"
    first, second = _replica(redis_url), _replica(redis_url)
    try:
        assert (await first.hit(key, 3, 60)).allowed
        assert (await second.hit(key, 3, 60)).allowed
        assert (await first.hit(key, 3, 60)).allowed

        refused = await second.hit(key, 3, 60)
    finally:
        await first.aclose()
        await second.aclose()

    assert not refused.allowed
    assert refused.count == 3
    assert 0 < refused.reset_after <= 60


@pytest.mark.asyncio
async def test_concurrent_requests_never_overshoot(redis_url: str) -> None:
    """Checking and counting are one step, so a burst admits exactly the limit."""
    key = f"user:{uuid.uuid4()}"
    replicas = [_replica(redis_url) for _ in range(4)]
    try:
        results = await asyncio.gather(*(replicas[i % 4].hit(key, 10, 60) for i in range(40)))
    finally:
        await asyncio.gather(*(replica.aclose() for replica in replicas))

    assert sum(result.allowed for result in results) == 10


@pytest.mark.asyncio
async def test_a_request_leaves_the_window_once_it_has_passed(redis_url: str) -> None:
    key = f"user:{uuid.uuid4()}"
    store = _replica(redis_url)
    try:
        assert (await store.hit(key, 1, 0.3)).allowed
        assert not (await store.hit(key, 1, 0.3)).allowed
        await asyncio.sleep(0.4)
        admitted = await store.hit(key, 1, 0.3)
    finally:
        await store.aclose()

    assert admitted.allowed


@pytest.fixture
def redis_rate_limit_client(postgres_url: str, redis_url: str) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        rate_limit_rpm=2,
        rate_limit_store="redis",
        rate_limit_redis_url=redis_url,
    )
    yield from build_test_client(config)


def test_the_gateway_counts_rate_limit_rpm_in_redis(redis_rate_limit_client: TestClient, redis_url: str) -> None:
    app_limiter = redis_rate_limit_client.app.state.rate_limiter  # type: ignore[attr-defined]
    assert isinstance(app_limiter, UserRateLimiter)
    header = {API_KEY_HEADER: "Bearer test-master-key"}
    user_id = f"redis-rl-{uuid.uuid4().hex[:8]}"
    assert (
        redis_rate_limit_client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=header).status_code == 200
    )

    async def failing_completion(**kwargs: Any) -> None:
        raise RuntimeError

    def chat() -> int:
        return redis_rate_limit_client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "user": user_id},
            headers=header,
        ).status_code

    with patch("gateway.api.routes.chat.acompletion", new=failing_completion):
        statuses = [chat() for _ in range(3)]

    assert statuses[-1] == 429
    assert 429 not in statuses[:2]

    async def counted() -> int:
        client = Redis.from_url(redis_url)
        try:
            return int(await client.zcard(f"otari:rl:{{user:{user_id}}}:log"))
        finally:
            await client.aclose()

    assert asyncio.run(counted()) == 2


@pytest.mark.asyncio
async def test_replicas_share_what_entries_cost_and_what_they_settle(redis_url: str) -> None:
    key = f"tpm:{uuid.uuid4()}"
    first, second = _replica(redis_url), _replica(redis_url)
    try:
        estimate = await first.hit(key, 100, 60, cost=90)
        refused = await second.hit(key, 100, 60, cost=20)
        assert estimate.handle is not None
        await first.settle(key, estimate.handle, 30)
        admitted = await second.hit(key, 100, 60, cost=70)
        full = await first.hit(key, 100, 60, cost=1)
    finally:
        await first.aclose()
        await second.aclose()

    assert (estimate.allowed, estimate.count) == (True, 90)
    assert not refused.allowed
    assert (admitted.allowed, admitted.count) == (True, 100)
    assert not full.allowed


@pytest.mark.asyncio
async def test_an_entry_leaving_the_window_frees_what_it_cost(redis_url: str) -> None:
    key = f"tpm:{uuid.uuid4()}"
    store = _replica(redis_url)
    try:
        assert (await store.hit(key, 100, 0.3, cost=100)).allowed
        assert not (await store.hit(key, 100, 0.3, cost=1)).allowed
        await asyncio.sleep(0.4)
        admitted = await store.hit(key, 100, 0.3, cost=100)
    finally:
        await store.aclose()

    assert (admitted.allowed, admitted.count) == (True, 100)


@pytest.mark.asyncio
async def test_concurrent_acquires_across_replicas_never_overshoot(redis_url: str) -> None:
    key = f"conc:{uuid.uuid4()}"
    replicas = [_replica(redis_url) for _ in range(4)]
    try:
        leases = await asyncio.gather(*(replicas[i % 4].acquire(key, 5, 30) for i in range(20)))
        taken = [lease for lease in leases if lease is not None]
        await replicas[0].release(key, taken[0])
        after_release = await replicas[1].acquire(key, 5, 30)
    finally:
        await asyncio.gather(*(replica.aclose() for replica in replicas))

    assert len(taken) == 5
    assert after_release is not None


@pytest.mark.asyncio
async def test_an_unreleased_slot_returns_when_its_lease_runs_out(redis_url: str) -> None:
    key = f"conc:{uuid.uuid4()}"
    store = _replica(redis_url)
    try:
        assert await store.acquire(key, 1, 0.3) is not None
        assert await store.acquire(key, 1, 0.3) is None
        await asyncio.sleep(0.4)
        reacquired = await store.acquire(key, 1, 0.3)
    finally:
        await store.aclose()

    assert reacquired is not None
