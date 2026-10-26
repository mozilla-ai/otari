"""The trace retention worker runs each of its steps on its own."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from gateway.services.traces import _retention
from gateway.services.traces._service import TraceService


class _Service:
    def __init__(self, ran: list[str]) -> None:
        self._ran = ran

    async def expire(self, **kwargs: Any) -> int:
        self._ran.append("traces")
        raise RuntimeError("trace expiry keeps failing")

    async def expire_erasures(self, **kwargs: Any) -> int:
        self._ran.append("erasures")
        return 0


@contextlib.asynccontextmanager
async def _no_unit_of_work() -> AsyncIterator[None]:
    yield None


@pytest.mark.asyncio
async def test_a_step_that_keeps_failing_does_not_hold_back_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    monkeypatch.setattr(_retention, "create_unit_of_work", _no_unit_of_work)
    worker = asyncio.create_task(
        _retention.run_trace_retention(
            lambda uow: cast(TraceService, _Service(ran)),
            retention_days=30,
            max_age_days=90,
            interval=0,
        )
    )
    try:
        async with asyncio.timeout(5):
            while ran.count("erasures") < 2:
                await asyncio.sleep(0)
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    assert ran[:4] == ["traces", "erasures"] * 2
