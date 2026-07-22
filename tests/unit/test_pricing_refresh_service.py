"""Tests for explicit genai-prices snapshot refreshes."""

from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from genai_prices.data_snapshot import DataSnapshot, get_snapshot
from genai_prices.types import ModelPrice
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

import gateway.services.pricing_refresh_service as pricing_refresh_service
from gateway.models.entities import PricingSnapshot

_PERSISTED_SNAPSHOT = (
    '[{"id":"test","name":"Test","api_pattern":"","models":['
    '{"id":"model","match":{"equals":"model"},"prices":{"input_mtok":"1","output_mtok":"2"}}]}]'
)


def _snapshot_with_changed_price() -> tuple[DataSnapshot, str]:
    snapshot = deepcopy(get_snapshot())
    for provider in snapshot.providers:
        for model in provider.models:
            if isinstance(model.prices, ModelPrice) and model.prices.input_mtok is not None:
                model.prices.input_mtok = Decimal("987.654")
                return snapshot, f"{provider.id}:{model.id}"
    raise AssertionError("Expected bundled genai-prices data to include a directly priced model")


@pytest.mark.asyncio
async def test_refresh_requires_confirmation_before_changing_active_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fetched snapshot remains pending until an operator confirms it."""

    active_snapshot = get_snapshot()
    latest_snapshot, model_key = _snapshot_with_changed_price()
    monkeypatch.setattr(
        pricing_refresh_service,
        "_fetch_latest_snapshot",
        lambda: pricing_refresh_service._PendingSnapshot(latest_snapshot, _PERSISTED_SNAPSHOT),
    )

    preview = await pricing_refresh_service.prepare_price_refresh()

    assert preview.changed_count >= 1
    assert any(change.model_key == model_key and change.change == "changed" for change in preview.changes)
    assert get_snapshot() is active_snapshot

    session = AsyncMock(spec=AsyncSession)
    session.get.return_value = None

    assert await pricing_refresh_service.confirm_price_refresh(session) is True
    assert get_snapshot() is latest_snapshot
    session.add.assert_called_once()
    stored = session.add.call_args.args[0]
    assert isinstance(stored, PricingSnapshot)
    assert stored.source == pricing_refresh_service.GENAI_PRICES_SOURCE
    assert stored.snapshot == _PERSISTED_SNAPSHOT
    session.commit.assert_awaited_once()
    assert await pricing_refresh_service.confirm_price_refresh(session) is False


@pytest.mark.asyncio
async def test_startup_loads_the_persisted_genai_prices_snapshot() -> None:
    """An approved upstream catalog is restored after a gateway restart."""

    row = PricingSnapshot(source=pricing_refresh_service.GENAI_PRICES_SOURCE, snapshot=_PERSISTED_SNAPSHOT)
    result = SimpleNamespace(scalar_one_or_none=lambda: row)
    session = AsyncMock(spec=AsyncSession)
    session.execute.return_value = result

    await pricing_refresh_service.load_persisted_price_snapshot(cast(AsyncSession, session))

    provider = get_snapshot().find_provider("model", "test", None)
    assert provider.id == "test"


@pytest.mark.asyncio
async def test_failed_snapshot_persistence_keeps_the_active_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A database failure cannot activate a snapshot that was not saved."""

    active_snapshot = get_snapshot()
    latest_snapshot, _ = _snapshot_with_changed_price()
    monkeypatch.setattr(
        pricing_refresh_service,
        "_fetch_latest_snapshot",
        lambda: pricing_refresh_service._PendingSnapshot(latest_snapshot, _PERSISTED_SNAPSHOT),
    )
    await pricing_refresh_service.prepare_price_refresh()
    session = AsyncMock(spec=AsyncSession)
    session.get.return_value = None
    session.commit.side_effect = SQLAlchemyError("database unavailable")

    with pytest.raises(pricing_refresh_service.PricingRefreshError):
        await pricing_refresh_service.confirm_price_refresh(session)

    assert get_snapshot() is active_snapshot
    session.rollback.assert_awaited_once()
