"""Preview and apply explicit updates to the genai-prices snapshot."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

import httpx2
from genai_prices import data as genai_data
from genai_prices.data_snapshot import DataSnapshot, get_snapshot, set_custom_snapshot
from genai_prices.update_prices import DEFAULT_UPDATE_URL, UpdatePrices
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.models.entities import PricingSnapshot
from gateway.services.pricing_service import reset_price_cache

_PREVIEW_CHANGE_LIMIT = 100
GENAI_PRICES_SOURCE = "genai-prices"
_refresh_lock = Lock()


@dataclass(frozen=True)
class _PendingSnapshot:
    snapshot: DataSnapshot
    raw_snapshot: str


_pending_snapshot: _PendingSnapshot | None = None


class PricingRefreshError(Exception):
    """The latest genai-prices snapshot could not be prepared."""


@dataclass(frozen=True)
class PricingRefreshChange:
    """One model added, changed, or removed by a snapshot refresh."""

    model_key: str
    change: str


@dataclass(frozen=True)
class PricingRefreshPreview:
    """Summary of a pending genai-prices snapshot update."""

    fetched_at: datetime
    added_count: int
    changed_count: int
    removed_count: int
    changes: list[PricingRefreshChange]
    changes_truncated: bool


def _snapshot_prices(snapshot: DataSnapshot, as_of: datetime) -> dict[str, object]:
    """Return active prices keyed by their upstream provider and model ids."""

    return {
        f"{provider.id}:{model.id}": model.get_prices(as_of)
        for provider in snapshot.providers
        for model in provider.models
    }


def _build_preview(current: DataSnapshot, latest: DataSnapshot, fetched_at: datetime) -> PricingRefreshPreview:
    """Compare active model prices from two snapshots."""

    current_prices = _snapshot_prices(current, fetched_at)
    latest_prices = _snapshot_prices(latest, fetched_at)
    changes: list[PricingRefreshChange] = []
    added_count = 0
    changed_count = 0
    removed_count = 0

    for model_key in sorted(current_prices.keys() | latest_prices.keys()):
        if model_key not in current_prices:
            added_count += 1
            change = "added"
        elif model_key not in latest_prices:
            removed_count += 1
            change = "removed"
        elif current_prices[model_key] != latest_prices[model_key]:
            changed_count += 1
            change = "changed"
        else:
            continue

        if len(changes) < _PREVIEW_CHANGE_LIMIT:
            changes.append(PricingRefreshChange(model_key=model_key, change=change))

    return PricingRefreshPreview(
        fetched_at=fetched_at,
        added_count=added_count,
        changed_count=changed_count,
        removed_count=removed_count,
        changes=changes,
        changes_truncated=(added_count + changed_count + removed_count) > len(changes),
    )


def _fetch_latest_snapshot() -> _PendingSnapshot:
    """Fetch the upstream snapshot without activating it."""

    updater = UpdatePrices()
    response = httpx2.get(DEFAULT_UPDATE_URL, timeout=updater.request_timeout)
    response.raise_for_status()
    raw_snapshot = response.content.decode("utf-8")
    providers = genai_data.providers_schema.validate_json(raw_snapshot)
    return _PendingSnapshot(
        snapshot=DataSnapshot(providers=providers, from_auto_update=True),
        raw_snapshot=raw_snapshot,
    )


async def prepare_price_refresh() -> PricingRefreshPreview:
    """Fetch and hold a new snapshot until an operator confirms it."""

    try:
        latest = await asyncio.to_thread(_fetch_latest_snapshot)
    except Exception as exc:
        raise PricingRefreshError("Unable to fetch the latest genai-prices data") from exc

    fetched_at = datetime.now(UTC)
    with _refresh_lock:
        global _pending_snapshot
        _pending_snapshot = latest
        return _build_preview(get_snapshot(), latest.snapshot, fetched_at)


async def confirm_price_refresh(session: AsyncSession) -> bool:
    """Persist and activate the pending snapshot, returning false when absent."""

    global _pending_snapshot
    with _refresh_lock:
        pending = _pending_snapshot
        if pending is None:
            return False

    row = await session.get(PricingSnapshot, GENAI_PRICES_SOURCE)
    if row is None:
        session.add(PricingSnapshot(source=GENAI_PRICES_SOURCE, snapshot=pending.raw_snapshot))
    else:
        row.snapshot = pending.raw_snapshot
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise PricingRefreshError("Unable to save the latest genai-prices data") from exc

    with _refresh_lock:
        set_custom_snapshot(pending.snapshot)
        if _pending_snapshot is pending:
            _pending_snapshot = None
    reset_price_cache()
    return True


def reject_price_refresh() -> bool:
    """Discard the pending snapshot without changing active pricing."""

    global _pending_snapshot
    with _refresh_lock:
        if _pending_snapshot is None:
            return False
        _pending_snapshot = None
    return True


async def load_persisted_price_snapshot(session: AsyncSession) -> None:
    """Load the last approved genai-prices snapshot during standalone startup."""

    snapshot_row = (
        await session.execute(select(PricingSnapshot).where(PricingSnapshot.source == GENAI_PRICES_SOURCE).limit(1))
    ).scalar_one_or_none()
    if snapshot_row is None:
        return
    try:
        providers = genai_data.providers_schema.validate_json(snapshot_row.snapshot)
    except ValueError:
        logger.warning("Ignoring invalid persisted %s pricing snapshot", GENAI_PRICES_SOURCE)
        return
    set_custom_snapshot(DataSnapshot(providers=providers, from_auto_update=True))
    reset_price_cache()
    logger.info("Loaded persisted %s pricing snapshot", GENAI_PRICES_SOURCE)


def reset_price_refresh_state() -> None:
    """Restore bundled pricing and discard a pending refresh, for app tests."""

    global _pending_snapshot
    with _refresh_lock:
        _pending_snapshot = None
        set_custom_snapshot(None)
    reset_price_cache()
