"""A running request's claim is renewed often enough that a short outage does not lapse it."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.core.unit_of_work import UnitOfWork
from gateway.services.inference import Claimed, IdempotencyService, IdempotentRequest, keep_claim_alive
from gateway.services.inference import _lease as lease_module

_REQUEST = IdempotentRequest(scope="master:u", key="k", request_hash="0" * 64, user_id="u", api_key_id=None)


class _Clock:
    """A clock the renewal loop's sleeps advance, so a loaded runner cannot stretch the schedule."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.now += delay
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_failed_renewals_are_retried_before_the_lease_runs_out(monkeypatch: pytest.MonkeyPatch) -> None:
    lease_sec = 0.6
    clock = _Clock()
    attempts: list[float] = []

    @asynccontextmanager
    async def failing_unit_of_work() -> AsyncIterator[UnitOfWork]:
        attempts.append(clock.monotonic())
        if len(attempts) <= 4:
            raise OSError("connection refused")
        yield MagicMock(spec=UnitOfWork)

    service = MagicMock(spec=IdempotencyService)
    service.renew = AsyncMock(return_value=False)
    monkeypatch.setattr(lease_module, "create_unit_of_work", failing_unit_of_work)
    monkeypatch.setattr(lease_module, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(lease_module, "asyncio", SimpleNamespace(sleep=clock.sleep))

    await asyncio.wait_for(keep_claim_alive(_REQUEST, Claimed("token"), lease_sec, lambda _uow: service), timeout=5)

    assert sum(1 for at in attempts if at < lease_sec * 0.9) >= 4


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
