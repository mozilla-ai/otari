"""A shared slot a request holds until its response ends, after waiting briefly for one to free."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from gateway import rate_limit
from gateway.adapters.rate_limit_store_adapter import InMemoryRateLimitStore
from gateway.core.error_codes import ERROR_CODE_HEADER
from gateway.rate_limit import RateLimitGrant, open_rate_limit_grant

KEY = "organization:org-1:concurrent"


async def _hold(grant: RateLimitGrant, limit: int = 1, wait_sec: float = 0.0) -> None:
    await grant.hold_slot(KEY, limit, wait_sec=wait_sec, lease_sec=60.0, rule="organization_concurrency")


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit, "_SLOT_POLL_SEC", 0.01)


@pytest.mark.asyncio
async def test_a_slot_is_held_until_the_grant_releases_it() -> None:
    store = InMemoryRateLimitStore()
    first = RateLimitGrant(store)
    await _hold(first)

    with pytest.raises(HTTPException) as refused:
        await _hold(RateLimitGrant(store))
    assert refused.value.status_code == 429
    assert refused.value.headers is not None
    assert refused.value.headers["Retry-After"] == "1"
    assert refused.value.headers[ERROR_CODE_HEADER] == "rate_limited"
    assert "1 request in flight" in str(refused.value.detail)

    await first.release()
    await _hold(RateLimitGrant(store))


@pytest.mark.asyncio
async def test_a_request_holds_one_slot_however_often_it_asks() -> None:
    store = InMemoryRateLimitStore()
    grant = RateLimitGrant(store)
    await _hold(grant, limit=2)
    await _hold(grant, limit=2)

    # The second slot is still free for another request.
    await _hold(RateLimitGrant(store), limit=2)


@pytest.mark.asyncio
async def test_a_waiting_request_gets_the_slot_another_gives_back() -> None:
    store = InMemoryRateLimitStore()
    first = RateLimitGrant(store)
    await _hold(first)

    async def finish_soon() -> None:
        await asyncio.sleep(0.05)
        await first.release()

    finishing = asyncio.create_task(finish_soon())
    await _hold(RateLimitGrant(store), wait_sec=2.0)
    await finishing


@pytest.mark.asyncio
async def test_a_request_past_the_queue_is_refused_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryRateLimitStore()
    await _hold(RateLimitGrant(store))
    # One request already waits, and the queue is as long as the limit.
    monkeypatch.setitem(rate_limit._slot_waiters, KEY, 1)

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(HTTPException):
        await _hold(RateLimitGrant(store), wait_sec=5.0)
    assert loop.time() - started < 1.0


@pytest.mark.asyncio
async def test_a_refused_wait_leaves_no_waiter_behind() -> None:
    store = InMemoryRateLimitStore()
    await _hold(RateLimitGrant(store))
    with pytest.raises(HTTPException):
        await _hold(RateLimitGrant(store), wait_sec=0.03)
    assert KEY not in rate_limit._slot_waiters


def _request(store: InMemoryRateLimitStore | None) -> Request:
    app = SimpleNamespace(state=SimpleNamespace(rate_limit_store=store))
    return Request({"type": "http", "headers": [], "app": app, "state": {}})


def test_an_empty_grant_is_opened_once_per_request() -> None:
    request = _request(InMemoryRateLimitStore())
    grant = open_rate_limit_grant(request)
    assert grant is not None
    assert open_rate_limit_grant(request) is grant


def test_no_grant_is_opened_where_the_app_counts_nothing() -> None:
    assert open_rate_limit_grant(_request(None)) is None
