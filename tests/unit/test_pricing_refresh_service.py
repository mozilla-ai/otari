"""Tests for explicit genai-prices snapshot refreshes."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from genai_prices import data as genai_data
from genai_prices.data_snapshot import DataSnapshot, get_snapshot
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

import gateway.services.pricing_refresh_service as pricing_refresh_service
from gateway.models.entities import PricingSnapshot

_PERSISTED_SNAPSHOT = (
    '[{"id":"test","name":"Test","api_pattern":"","models":['
    '{"id":"model","match":{"equals":"model"},"prices":{"input_mtok":"1","output_mtok":"2"}}]}]'
)


def _snapshot_with_changed_price() -> tuple[DataSnapshot, str, str]:
    raw_snapshot = _PERSISTED_SNAPSHOT.replace('"input_mtok":"1"', '"input_mtok":"987.654"')
    providers = genai_data.providers_schema.validate_json(raw_snapshot)
    return DataSnapshot(providers=providers, from_auto_update=True), "test:model", raw_snapshot


@pytest.mark.asyncio
async def test_refresh_requires_confirmation_before_changing_active_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reviewed snapshot survives into a separate confirmation session."""

    active_snapshot = get_snapshot()
    latest_snapshot, model_key, raw_snapshot = _snapshot_with_changed_price()
    monkeypatch.setattr(
        pricing_refresh_service,
        "_fetch_latest_snapshot",
        lambda: pricing_refresh_service._PendingSnapshot(latest_snapshot, raw_snapshot),
    )

    preview_session = AsyncMock(spec=AsyncSession)
    preview_session.get.return_value = None

    preview = await pricing_refresh_service.prepare_price_refresh(preview_session)

    assert preview.added_count >= 1
    assert get_snapshot() is active_snapshot

    pending = preview_session.add.call_args.args[0]
    assert isinstance(pending, PricingSnapshot)
    assert pending.source == pricing_refresh_service.GENAI_PRICES_PENDING_SOURCE
    assert pending.snapshot == raw_snapshot
    preview_session.commit.assert_awaited_once()

    result = SimpleNamespace(scalar_one_or_none=lambda: pending)
    confirmation_session = AsyncMock(spec=AsyncSession)
    confirmation_session.execute.return_value = result
    confirmation_session.get.return_value = None

    assert await pricing_refresh_service.confirm_price_refresh(confirmation_session) is True
    assert pricing_refresh_service._snapshot_prices(get_snapshot(), preview.fetched_at)[model_key] == (
        pricing_refresh_service._snapshot_prices(latest_snapshot, preview.fetched_at)[model_key]
    )
    confirmation_session.add.assert_called_once()
    stored = confirmation_session.add.call_args.args[0]
    assert isinstance(stored, PricingSnapshot)
    assert stored.source == pricing_refresh_service.GENAI_PRICES_SOURCE
    assert stored.snapshot == raw_snapshot
    confirmation_session.delete.assert_awaited_once_with(pending)
    confirmation_session.commit.assert_awaited_once()

    no_pending = AsyncMock(spec=AsyncSession)
    no_pending.execute.return_value = SimpleNamespace(scalar_one_or_none=lambda: None)
    assert await pricing_refresh_service.confirm_price_refresh(no_pending) is False


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
    latest_snapshot, _, _ = _snapshot_with_changed_price()
    monkeypatch.setattr(
        pricing_refresh_service,
        "_fetch_latest_snapshot",
        lambda: pricing_refresh_service._PendingSnapshot(latest_snapshot, _PERSISTED_SNAPSHOT),
    )
    preview_session = AsyncMock(spec=AsyncSession)
    preview_session.get.return_value = None
    await pricing_refresh_service.prepare_price_refresh(preview_session)
    pending = preview_session.add.call_args.args[0]

    result = SimpleNamespace(scalar_one_or_none=lambda: pending)
    session = AsyncMock(spec=AsyncSession)
    session.execute.return_value = result
    session.get.return_value = None
    session.commit.side_effect = SQLAlchemyError("database unavailable")

    with pytest.raises(pricing_refresh_service.PricingRefreshError):
        await pricing_refresh_service.confirm_price_refresh(session)

    assert get_snapshot() is active_snapshot
    session.rollback.assert_awaited_once()
