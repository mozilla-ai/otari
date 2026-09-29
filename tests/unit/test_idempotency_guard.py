"""A request's hold on its idempotency key survives a failure of the work that keeps it alive."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response

from gateway.api.routes._idempotency import IdempotencyGuard
from gateway.core.unit_of_work import UnitOfWork
from gateway.services.inference import Claimed, IdempotencyService, IdempotentRequest, keep_claim_alive
from gateway.services.inference import _lease as lease_module

_REQUEST = IdempotentRequest(scope="master:u", key="k", request_hash="0" * 64, user_id="u", api_key_id=None)


def _raw_request() -> MagicMock:
    raw = MagicMock()
    raw.body = AsyncMock(return_value=b"{}")
    raw.headers = {}
    return raw


@pytest.mark.asyncio
async def test_a_failed_keep_alive_does_not_lose_the_response() -> None:
    service = MagicMock()
    service.admit = AsyncMock(return_value=Claimed("token"))
    service.complete = AsyncMock(return_value=True)

    async def keep_alive(request: IdempotentRequest, claimed: Claimed) -> None:
        raise OSError("connection refused")

    guard = IdempotencyGuard(_raw_request(), service, "k", keep_alive=keep_alive)
    await guard.admit(endpoint="/v1/chat/completions", user_id="u", api_key_id=None)
    await asyncio.sleep(0)

    await guard.complete({"ok": True}, Response())

    service.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_renewal_goes_on_after_an_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    @asynccontextmanager
    async def unit_of_work() -> AsyncIterator[UnitOfWork]:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("connection refused")
        yield MagicMock(spec=UnitOfWork)

    service = MagicMock(spec=IdempotencyService)
    service.renew = AsyncMock(return_value=False)
    monkeypatch.setattr(lease_module, "create_unit_of_work", unit_of_work)

    await asyncio.wait_for(keep_claim_alive(_REQUEST, Claimed("token"), 0.03, lambda _uow: service), timeout=2)

    assert len(attempts) == 2
    service.renew.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_claim_is_kept_alive_until_the_response_is_stored() -> None:
    ticks = 0

    async def keep_alive(request: IdempotentRequest, claimed: Claimed) -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    ticks_during_store: list[int] = []

    async def slow_store(*_args: object, **_kwargs: object) -> bool:
        before = ticks
        await asyncio.sleep(0.05)
        ticks_during_store.append(ticks - before)
        return True

    service = MagicMock()
    service.admit = AsyncMock(return_value=Claimed("token"))
    service.complete = AsyncMock(side_effect=slow_store)
    guard = IdempotencyGuard(_raw_request(), service, "k", keep_alive=keep_alive)
    await guard.admit(endpoint="/v1/chat/completions", user_id="u", api_key_id=None)

    await guard.complete({"ok": True}, Response())

    assert ticks_during_store[0] > 0
