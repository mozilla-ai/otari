"""The idempotency service's decisions that need no database."""

import asyncio
import itertools
import random
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.inference import IdempotencyService, IdempotentRequest, StillInFlight
from gateway.services.inference._idempotency import _CHANGES_BEFORE_WAITING, _poll_delays
from gateway.services.secret_box import generate_secret_key


@pytest.mark.parametrize(
    ("retention_sec", "secret_set", "enabled"),
    [(86400, True, True), (0, True, False), (86400, False, False)],
)
def test_keys_are_honored_only_with_a_retention_and_a_secret(
    monkeypatch: pytest.MonkeyPatch, retention_sec: int, secret_set: bool, enabled: bool
) -> None:
    if secret_set:
        monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    else:
        monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    config = GatewayConfig(idempotency_retention_sec=retention_sec)

    assert IdempotencyService.is_enabled(config) is enabled


class _NoUnitOfWork:
    async def __aenter__(self) -> "_NoUnitOfWork":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


def test_waiting_retries_back_off_exponentially_with_jitter() -> None:
    delays = list(itertools.islice(_poll_delays(random.Random(0)), 8))

    bases = [0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0, 2.0]
    for delay, base in zip(delays, bases, strict=True):
        assert base / 2 <= delay <= base


@pytest.mark.asyncio
async def test_a_claim_that_keeps_changing_does_not_spin() -> None:
    """A key that vanishes between the insert and the read every time is looked at a bounded number of times."""
    keys = MagicMock()
    keys.insert_claim = AsyncMock(return_value=False)
    keys.find = AsyncMock(return_value=None)
    keys.get_database_time = AsyncMock(return_value=datetime.now(UTC))
    keys.get_caller = AsyncMock(return_value=MagicMock(blocked=False))
    service = IdempotencyService(
        _NoUnitOfWork(),  # type: ignore[arg-type]
        MagicMock(idempotency=keys),
        GatewayConfig(idempotency_wait_sec=0),
    )
    request = IdempotentRequest(scope="master:u", key="k", request_hash="0" * 64, user_id="u", api_key_id=None)

    outcome = await asyncio.wait_for(service.admit(request), timeout=2)

    assert isinstance(outcome, StillInFlight)
    assert keys.find.await_count == _CHANGES_BEFORE_WAITING


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [0, -1])
async def test_a_sweep_needs_a_positive_batch_size(batch_size: int) -> None:
    service = IdempotencyService(
        _NoUnitOfWork(),  # type: ignore[arg-type]
        MagicMock(idempotency=MagicMock()),
        GatewayConfig(),
    )

    with pytest.raises(ValueError, match="batch"):
        await asyncio.wait_for(service.sweep(batch_size=batch_size), timeout=2)
