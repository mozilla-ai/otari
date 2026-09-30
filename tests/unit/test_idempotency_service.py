"""The idempotency service's decisions that need no database."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.core.config import GatewayConfig
from gateway.models.inference import IdempotencyState
from gateway.services.inference import IdempotencyService, IdempotentRequest, StillInFlight
from gateway.services.inference._idempotency import _CHANGES_BEFORE_CONFLICT
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
        GatewayConfig(),
    )
    request = IdempotentRequest(scope="master:u", key="k", request_hash="0" * 64, user_id="u", api_key_id=None)

    outcome = await asyncio.wait_for(service.admit(request), timeout=2)

    assert isinstance(outcome, StillInFlight)
    assert keys.find.await_count == _CHANGES_BEFORE_CONFLICT


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


@pytest.mark.asyncio
async def test_a_retry_of_a_running_request_is_answered_at_once() -> None:
    """A retry does not wait for the original, as the IETF Idempotency-Key draft and Stripe's API answer 409."""
    now = datetime.now(UTC)
    running = MagicMock(
        state=IdempotencyState.IN_PROGRESS,
        request_hash="0" * 64,
        claim_token="original",
        locked_until=now + timedelta(minutes=1),
        expires_at=now + timedelta(days=1),
    )
    keys = MagicMock()
    keys.get_caller = AsyncMock(return_value=MagicMock(blocked=False))
    keys.get_database_time = AsyncMock(return_value=now)
    keys.insert_claim = AsyncMock(return_value=False)
    keys.find = AsyncMock(return_value=running)
    service = IdempotencyService(
        _NoUnitOfWork(),  # type: ignore[arg-type]
        MagicMock(idempotency=keys),
        GatewayConfig(),
    )
    request = IdempotentRequest(scope="master:u", key="k", request_hash="0" * 64, user_id="u", api_key_id=None)

    outcome = await asyncio.wait_for(service.admit(request), timeout=1)

    assert isinstance(outcome, StillInFlight)
    keys.find.assert_awaited_once()
