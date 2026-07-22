"""Preview and apply explicit updates to the genai-prices snapshot."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from genai_prices import data as genai_data
from genai_prices.data_snapshot import DataSnapshot, get_snapshot, set_custom_snapshot
from genai_prices.update_prices import DEFAULT_UPDATE_URL, UpdatePrices
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import PricingSnapshot
from gateway.services.pricing_service import reset_price_cache

_PREVIEW_CHANGE_LIMIT = 100
GENAI_PRICES_SOURCE = "genai-prices"
GENAI_PRICES_PENDING_SOURCE = "genai-prices-pending"
# Reloading the accepted snapshot is cheap only when it changed, so the refresher
# compares against this before touching the price cache. Matches the alias and
# provider refreshers' cadence.
PRICE_SNAPSHOT_REFRESH_TTL_SECONDS = 30.0

# The raw snapshot this worker has applied in-memory. A confirm refreshes the
# worker that served it; the refresher uses this to converge sibling workers and
# replicas without re-applying (and clearing the price cache) on every tick.
_applied_snapshot_raw: str | None = None


@dataclass(frozen=True)
class _PendingSnapshot:
    snapshot: DataSnapshot
    raw_snapshot: str


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
    # genai-prices exposes its default timeout as an httpx2 object; mirror it onto
    # httpx (the client the rest of the gateway uses) so behavior is unchanged.
    upstream_timeout = updater.request_timeout
    timeout = httpx.Timeout(
        connect=upstream_timeout.connect,
        read=upstream_timeout.read,
        write=upstream_timeout.write,
        pool=upstream_timeout.pool,
    )
    response = httpx.get(DEFAULT_UPDATE_URL, timeout=timeout)
    response.raise_for_status()
    raw_snapshot = response.content.decode("utf-8")
    providers = genai_data.providers_schema.validate_json(raw_snapshot)
    return _PendingSnapshot(
        snapshot=DataSnapshot(providers=providers, from_auto_update=True),
        raw_snapshot=raw_snapshot,
    )


async def prepare_price_refresh(session: AsyncSession) -> PricingRefreshPreview:
    """Fetch and persist a new snapshot until an operator confirms it."""

    try:
        latest = await asyncio.to_thread(_fetch_latest_snapshot)
    except Exception as exc:
        raise PricingRefreshError("Unable to fetch the latest genai-prices data") from exc

    fetched_at = datetime.now(UTC)
    preview = _build_preview(get_snapshot(), latest.snapshot, fetched_at)
    pending_row = await session.get(PricingSnapshot, GENAI_PRICES_PENDING_SOURCE)
    if pending_row is None:
        session.add(PricingSnapshot(source=GENAI_PRICES_PENDING_SOURCE, snapshot=latest.raw_snapshot))
    else:
        pending_row.snapshot = latest.raw_snapshot
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise PricingRefreshError("Unable to save the latest genai-prices data") from exc
    return preview


async def confirm_price_refresh(session: AsyncSession) -> bool:
    """Persist and activate the pending snapshot, returning false when absent."""

    # Best-effort row lock so a second confirm/reject serializes behind this one.
    # It is a no-op on SQLite (the standalone default) and only actually serializes
    # on Postgres; the path is master-key-only and single-writer in practice, so the
    # lock is defensive rather than load-bearing.
    pending_row = (
        await session.execute(
            select(PricingSnapshot)
            .where(PricingSnapshot.source == GENAI_PRICES_PENDING_SOURCE)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if pending_row is None:
        return False
    # Read the raw snapshot before delete/commit: expire_on_commit would otherwise
    # make this attribute access re-fetch a row that no longer exists.
    raw_snapshot = pending_row.snapshot
    try:
        providers = genai_data.providers_schema.validate_json(raw_snapshot)
    except ValueError as exc:
        raise PricingRefreshError("The pending genai-prices data is invalid") from exc

    snapshot = DataSnapshot(providers=providers, from_auto_update=True)
    active_row = await session.get(PricingSnapshot, GENAI_PRICES_SOURCE)
    if active_row is None:
        session.add(PricingSnapshot(source=GENAI_PRICES_SOURCE, snapshot=raw_snapshot))
    else:
        active_row.snapshot = raw_snapshot
    await session.delete(pending_row)
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise PricingRefreshError("Unable to save the latest genai-prices data") from exc

    global _applied_snapshot_raw
    set_custom_snapshot(snapshot)
    reset_price_cache()
    _applied_snapshot_raw = raw_snapshot
    return True


async def reject_price_refresh(session: AsyncSession) -> bool:
    """Discard the pending snapshot without changing active pricing."""

    # See confirm_price_refresh: best-effort lock, a no-op on SQLite.
    pending_row = (
        await session.execute(
            select(PricingSnapshot)
            .where(PricingSnapshot.source == GENAI_PRICES_PENDING_SOURCE)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if pending_row is None:
        return False
    await session.delete(pending_row)
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise PricingRefreshError("Unable to discard the pending genai-prices data") from exc
    return True


def _apply_active_snapshot(raw_snapshot: str) -> None:
    """Activate a raw snapshot in this worker and record what was applied."""

    global _applied_snapshot_raw
    providers = genai_data.providers_schema.validate_json(raw_snapshot)
    set_custom_snapshot(DataSnapshot(providers=providers, from_auto_update=True))
    reset_price_cache()
    _applied_snapshot_raw = raw_snapshot


async def _get_active_snapshot_row(session: AsyncSession) -> PricingSnapshot | None:
    result = await session.execute(
        select(PricingSnapshot).where(PricingSnapshot.source == GENAI_PRICES_SOURCE).limit(1)
    )
    return result.scalar_one_or_none()


async def load_persisted_price_snapshot(session: AsyncSession) -> None:
    """Load the last approved genai-prices snapshot during standalone startup."""

    snapshot_row = await _get_active_snapshot_row(session)
    if snapshot_row is None:
        return
    try:
        _apply_active_snapshot(snapshot_row.snapshot)
    except ValueError:
        logger.warning("Ignoring invalid persisted %s pricing snapshot", GENAI_PRICES_SOURCE)
        return
    logger.info("Loaded persisted %s pricing snapshot", GENAI_PRICES_SOURCE)


async def refresh_price_snapshot(session: AsyncSession) -> None:
    """Re-apply the accepted snapshot when a confirm on another worker changed it.

    The active row only ever appears or advances via ``confirm_price_refresh``, so
    a snapshot equal to what this worker already applied is skipped, leaving the
    price cache untouched.
    """

    snapshot_row = await _get_active_snapshot_row(session)
    if snapshot_row is None or snapshot_row.snapshot == _applied_snapshot_raw:
        return
    try:
        _apply_active_snapshot(snapshot_row.snapshot)
    except ValueError:
        logger.warning("Ignoring invalid persisted %s pricing snapshot", GENAI_PRICES_SOURCE)
        return
    logger.info("Applied updated %s pricing snapshot accepted on another worker", GENAI_PRICES_SOURCE)


async def run_price_snapshot_refresher(interval: float = PRICE_SNAPSHOT_REFRESH_TTL_SECONDS) -> None:
    """Reload the accepted snapshot forever, so a confirm on another worker arrives.

    ``confirm_price_refresh`` refreshes the worker that served it, which covers a
    single-process gateway. This covers the rest: sibling workers and other replicas
    pick up an accepted snapshot within ``interval``, mirroring the alias and provider
    refreshers. Cancelled at shutdown.

    Every error is swallowed and retried on the next tick. A database blip must not
    kill the refresher, because nothing would restart it and the worker would then
    serve frozen prices for as long as it stayed up.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_session() as db:
                await refresh_price_snapshot(db)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Pricing snapshot refresh failed; retrying in %ss", interval, exc_info=True)


def reset_price_refresh_state() -> None:
    """Restore bundled pricing for app tests."""

    global _applied_snapshot_raw
    set_custom_snapshot(None)
    reset_price_cache()
    _applied_snapshot_raw = None
