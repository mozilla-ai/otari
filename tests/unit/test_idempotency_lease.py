"""A running request's claim is renewed often enough that a short outage does not lapse it."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.core.unit_of_work import UnitOfWork
from gateway.services.inference import Claimed, IdempotencyService, IdempotentRequest, keep_claim_alive
from gateway.services.inference import _lease as lease_module

_REQUEST = IdempotentRequest(scope="master:u", key="k", request_hash="0" * 64, user_id="u", api_key_id=None)


@pytest.mark.asyncio
async def test_failed_renewals_are_retried_before_the_lease_runs_out(monkeypatch: pytest.MonkeyPatch) -> None:
    lease_sec = 0.6
    attempts: list[float] = []

    @asynccontextmanager
    async def failing_unit_of_work() -> AsyncIterator[UnitOfWork]:
        attempts.append(time.monotonic())
        if len(attempts) <= 4:
            raise OSError("connection refused")
        yield MagicMock(spec=UnitOfWork)

    service = MagicMock(spec=IdempotencyService)
    service.renew = AsyncMock(return_value=False)
    monkeypatch.setattr(lease_module, "create_unit_of_work", failing_unit_of_work)
    started = time.monotonic()

    await asyncio.wait_for(keep_claim_alive(_REQUEST, Claimed("token"), lease_sec, lambda _uow: service), timeout=5)

    assert sum(1 for at in attempts if at - started < lease_sec * 0.9) >= 4
