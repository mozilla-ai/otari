"""Preview and apply explicit updates to the models.dev price snapshot, and keep the stored models.dev catalog.

The price snapshot changes only when an operator, or the ``auto`` policy,
accepts it. The stored catalog carries the model facts, which change no price,
so the catalog refresher replaces it on each download that differs.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, create_session, describe_database_error
from gateway.log_config import logger
from gateway.models.pricing import PricingSnapshot, PricingSnapshotHistory
from gateway.services.model_catalog_service import (
    ModelsDevFetchError,
    ModelsDevRefusedError,
    background_catalog_enabled,
    catalog_refresh_interval,
    clear_catalog_cache,
    fetch_models_dev_document,
    serve_catalog,
)
from gateway.services.pricing import (
    MAX_RESIDENT_GENERATIONS,
    ModelsDevPriceIndex,
    PriceGeneration,
    add_accepted_generation,
    current_index,
    invalid_rate_count,
    reset_generations,
    set_accepted_generations,
)
from gateway.services.pricing_service import normalize_effective_at
from gateway.types.models_dev import ModelsDevCatalog, catalog_document, read_catalog_document, snapshot_document

_PREVIEW_CHANGE_LIMIT = 100
# A candidate that loses more than this share of the active priced models, or
# removes more than this share of them, is held for a person even under ``auto``.
MAX_PRICED_MODELS_DROP_FRACTION = 0.10
MAX_REMOVED_MODELS_FRACTION = 0.10
MODELS_DEV_SOURCE = "models.dev"
MODELS_DEV_PENDING_SOURCE = "models.dev-pending"
# Not a snapshot: the row every worker's poller claims a tick on. It lives in the
# same table because a claim is exactly one row with a timestamp, and the table
# already is that.
MODELS_DEV_POLL_CLAIM_SOURCE = "models.dev-poll-claim"
# Matches the alias and provider refreshers' cadence.
PRICE_SNAPSHOT_REFRESH_TTL_SECONDS = 30.0
# The whole models.dev catalog, for the model facts. Not a price snapshot: its
# changes take effect without review, and it keeps no history.
MODELS_DEV_CATALOG_SOURCE = "models.dev-catalog"
# The row every worker's catalog refresher claims a download on.
MODELS_DEV_CATALOG_CLAIM_SOURCE = "models.dev-catalog-claim"
# How soon a failed catalog download is tried again, instead of a whole interval later.
CATALOG_RETRY_SECONDS = 60.0

# When the active row this worker has applied was last written. A confirm
# refreshes the worker that served it; the refresher compares this against the
# row's timestamp alone, so converging sibling workers and replicas reads no
# payload until the row has changed.
_applied_updated_at: datetime | None = None
# The same for the stored catalog row.
_catalog_applied_at: datetime | None = None


def _parse_snapshot(raw_snapshot: str) -> ModelsDevPriceIndex:
    """Index a stored snapshot, refusing one that carries no models."""
    try:
        index = ModelsDevPriceIndex.from_document(json.loads(raw_snapshot))
    except Exception as exc:
        raise ValueError("Invalid models.dev snapshot") from exc
    if len(index) == 0:
        raise ValueError("Invalid models.dev snapshot")
    return index


@dataclass(frozen=True)
class _PendingSnapshot:
    index: ModelsDevPriceIndex
    raw_snapshot: str


class PricingRefreshError(Exception):
    """The latest models.dev snapshot could not be prepared."""


class PendingSnapshotChanged(PricingRefreshError):
    """The pending snapshot is no longer the one the caller reviewed."""


@dataclass(frozen=True)
class PricingRefreshChange:
    """One model added, changed, or removed by a snapshot refresh."""

    model_key: str
    change: Literal["added", "changed", "removed"]


@dataclass(frozen=True)
class PricingRefreshPreview:
    """Summary of a pending models.dev snapshot update."""

    fetched_at: datetime
    added_count: int
    changed_count: int
    removed_count: int
    changes: list[PricingRefreshChange]
    changes_truncated: bool
    digest: str = ""
    needs_review: bool = False
    review_reason: str | None = None

    @property
    def changes_nothing(self) -> bool:
        """Whether the fetched prices match the active ones."""
        return self.added_count + self.changed_count + self.removed_count == 0


def _snapshot_prices(index: ModelsDevPriceIndex) -> dict[str, tuple[object, ...]]:
    """Each priced model's rates, keyed by its models.dev provider and model ids."""

    return {
        f"{entry.provider_id}:{entry.model_id}": entry.price_signature() for entry in index.entries() if entry.priced
    }


def _review_reason(active_count: int, latest_count: int, removed_count: int) -> str | None:
    """Why a candidate must not be applied without a person, or ``None``."""
    if active_count == 0:
        return None
    reasons: list[str] = []
    if (active_count - latest_count) / active_count > MAX_PRICED_MODELS_DROP_FRACTION:
        reasons.append(f"priced models fall from {active_count} to {latest_count}")
    if removed_count / active_count > MAX_REMOVED_MODELS_FRACTION:
        reasons.append(f"{removed_count} of {active_count} priced models are removed")
    return "; ".join(reasons) or None


def snapshot_digest(raw_snapshot: str) -> str:
    """The identity of a stored snapshot, for binding a confirm to what was previewed."""
    return hashlib.sha256(raw_snapshot.encode("utf-8")).hexdigest()


def _build_preview(
    current: ModelsDevPriceIndex, latest: ModelsDevPriceIndex, fetched_at: datetime, raw_snapshot: str
) -> PricingRefreshPreview:
    """Compare the priced models of two snapshots."""

    current_prices = _snapshot_prices(current)
    latest_prices = _snapshot_prices(latest)
    changes: list[PricingRefreshChange] = []
    added_count = 0
    changed_count = 0
    removed_count = 0

    for model_key in sorted(current_prices.keys() | latest_prices.keys()):
        change: Literal["added", "changed", "removed"]
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

    reason = _review_reason(len(current_prices), len(latest_prices), removed_count)
    return PricingRefreshPreview(
        fetched_at=fetched_at,
        added_count=added_count,
        changed_count=changed_count,
        removed_count=removed_count,
        changes=changes,
        changes_truncated=(added_count + changed_count + removed_count) > len(changes),
        digest=snapshot_digest(raw_snapshot),
        needs_review=reason is not None,
        review_reason=reason,
    )


def _prepare_document(catalog: ModelsDevCatalog) -> _PendingSnapshot:
    if (invalid := invalid_rate_count(catalog)) > 0:
        raise PricingRefreshError(f"The models.dev data carries {invalid} implausible rates")
    raw_snapshot = json.dumps(snapshot_document(catalog), separators=(",", ":"), ensure_ascii=False)
    index = _parse_snapshot(raw_snapshot)
    if not any(entry.priced for entry in index.entries()):
        raise PricingRefreshError("The models.dev data prices no models")
    return _PendingSnapshot(index=index, raw_snapshot=raw_snapshot)


async def _fetch_latest_snapshot(config: GatewayConfig | None, reuse_within: float) -> _PendingSnapshot:
    """Fetch the upstream catalog and cut it down to the price fields, without activating it."""

    try:
        catalog = await fetch_models_dev_document(
            reuse_within=reuse_within, fill_cache=config is not None and config.models_dev_metadata
        )
    except ModelsDevRefusedError as exc:
        raise PricingRefreshError("The models.dev data has a shape the gateway cannot read") from exc
    return await asyncio.to_thread(_prepare_document, catalog)


async def prepare_price_refresh(
    session: AsyncSession, config: GatewayConfig | None = None, *, reuse_within: float = 0.0
) -> PricingRefreshPreview:
    """Fetch and persist a new snapshot until an operator confirms it.

    Works whether or not ``models_dev_metadata`` is on; ``config`` only decides
    whether the download also warms the metadata cache.
    """

    try:
        latest = await _fetch_latest_snapshot(config, reuse_within)
    except PricingRefreshError:
        logger.warning("Refusing the fetched models.dev data as implausible", exc_info=True)
        raise
    except Exception as exc:
        raise PricingRefreshError("Unable to fetch the latest models.dev data") from exc

    fetched_at = datetime.now(UTC)
    preview = _build_preview(current_index(), latest.index, fetched_at, latest.raw_snapshot)
    pending_row = await session.get(PricingSnapshot, MODELS_DEV_PENDING_SOURCE)
    if pending_row is None:
        session.add(PricingSnapshot(source=MODELS_DEV_PENDING_SOURCE, snapshot=latest.raw_snapshot))
    else:
        pending_row.snapshot = latest.raw_snapshot
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise PricingRefreshError("Unable to save the latest models.dev data") from exc
    return preview


async def _lock_pending(session: AsyncSession, digest: str | None) -> PricingSnapshot | None:
    """The pending row, locked where the database supports it, checked against ``digest``."""
    pending_row = (
        await session.execute(
            select(PricingSnapshot).where(PricingSnapshot.source == MODELS_DEV_PENDING_SOURCE).with_for_update()
        )
    ).scalar_one_or_none()
    if pending_row is not None and digest is not None and digest != snapshot_digest(pending_row.snapshot):
        await session.rollback()
        raise PendingSnapshotChanged("The pending models.dev data changed since it was reviewed")
    return pending_row


async def _delete_pending(session: AsyncSession, raw_snapshot: str) -> bool:
    """Delete the pending row only if it still holds ``raw_snapshot``, so one caller wins."""
    result = await session.execute(
        delete(PricingSnapshot).where(
            PricingSnapshot.source == MODELS_DEV_PENDING_SOURCE, PricingSnapshot.snapshot == raw_snapshot
        )
    )
    # ``rowcount`` lives on CursorResult, and mypy sees the result of ``execute()`` as Result.
    return getattr(result, "rowcount", 0) == 1


async def confirm_price_refresh(
    session: AsyncSession, *, accepted_by: str = "operator", digest: str | None = None
) -> bool:
    """Persist and activate the pending snapshot, returning false when absent.

    ``accepted_by`` is recorded on the history row: ``operator`` for a confirm
    from the dashboard, ``schedule`` when the scheduled catalog update
    applied it under the ``auto`` policy. With ``digest``, only the snapshot
    that digest names is accepted; a replaced one raises
    :class:`PendingSnapshotChanged` and nothing changes.

    Reading, validating and deleting the pending row happen in one transaction,
    and the delete is conditional on the row's content, so two confirms never
    both apply even where ``FOR UPDATE`` is a no-op (SQLite).
    """

    pending_row = await _lock_pending(session, digest)
    if pending_row is None:
        return False
    raw_snapshot = pending_row.snapshot
    try:
        index = await asyncio.to_thread(_parse_snapshot, raw_snapshot)
    except ValueError as exc:
        await session.rollback()
        raise PricingRefreshError("The pending models.dev data is invalid") from exc
    if not await _delete_pending(session, raw_snapshot):
        await session.rollback()
        return False

    accepted_at = datetime.now(UTC)
    active_row = await session.get(PricingSnapshot, MODELS_DEV_SOURCE)
    if active_row is None:
        session.add(PricingSnapshot(source=MODELS_DEV_SOURCE, snapshot=raw_snapshot))
    else:
        active_row.snapshot = raw_snapshot
    session.add(
        PricingSnapshotHistory(
            source=MODELS_DEV_SOURCE,
            accepted_at=accepted_at,
            accepted_by=accepted_by,
            model_count=len(index),
            snapshot=raw_snapshot,
        )
    )
    pruned = await _prune_history(session)
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise PricingRefreshError("Unable to save the latest models.dev data") from exc

    global _applied_updated_at
    add_accepted_generation(PriceGeneration(effective_at=accepted_at, index=index), history_pruned=pruned)
    _applied_updated_at = await _active_updated_at(session)
    return True


async def preview_pending_refresh(session: AsyncSession) -> PricingRefreshPreview | None:
    """The change a stored pending snapshot would make, without fetching again.

    What the dashboard shows when the scheduled refresh has left an update
    waiting under the ``review`` policy: the same summary ``prepare_price_refresh``
    produced when it fetched, recomputed against whatever is active now.
    ``None`` when nothing is pending.
    """
    pending_row = await session.get(PricingSnapshot, MODELS_DEV_PENDING_SOURCE)
    if pending_row is None:
        return None
    try:
        latest = await asyncio.to_thread(_parse_snapshot, pending_row.snapshot)
    except ValueError as exc:
        raise PricingRefreshError("The pending models.dev data is invalid") from exc
    fetched_at = normalize_effective_at(pending_row.updated_at)
    return _build_preview(current_index(), latest, fetched_at, pending_row.snapshot)


@dataclass(frozen=True)
class AcceptedSnapshot:
    """One row of the accepted-snapshot history, without the payload."""

    id: uuid.UUID
    accepted_at: datetime
    accepted_by: str
    model_count: int


# How many accepted snapshots are kept, payload included. Each one is the whole
# upstream price dataset, over a megabyte, and under the auto policy one can
# land every day, so the history is a window rather than a ledger: enough to
# answer what a rate was a month ago, not enough to grow without bound.
PRICING_SNAPSHOT_HISTORY_KEEP = 30


async def _prune_history(session: AsyncSession) -> bool:
    """Drop the accepted snapshots older than the newest ``PRICING_SNAPSHOT_HISTORY_KEEP``.

    Returns whether the history is now at its limit, so older accepts may be gone.
    """
    keep = (
        select(PricingSnapshotHistory.id)
        .where(PricingSnapshotHistory.source == MODELS_DEV_SOURCE)
        .order_by(PricingSnapshotHistory.accepted_at.desc())
        .limit(PRICING_SNAPSHOT_HISTORY_KEEP)
    )
    kept = {row for row in (await session.execute(keep)).scalars()}
    if len(kept) < PRICING_SNAPSHOT_HISTORY_KEEP:
        return False
    await session.execute(
        delete(PricingSnapshotHistory).where(
            PricingSnapshotHistory.source == MODELS_DEV_SOURCE,
            PricingSnapshotHistory.id.not_in(kept),
        )
    )
    return True


async def list_accepted_snapshots(session: AsyncSession, limit: int = 50) -> list[AcceptedSnapshot]:
    """The accepted snapshots, newest first, payloads left in the database."""
    stmt = (
        select(
            PricingSnapshotHistory.id,
            PricingSnapshotHistory.accepted_at,
            PricingSnapshotHistory.accepted_by,
            PricingSnapshotHistory.model_count,
        )
        .where(PricingSnapshotHistory.source == MODELS_DEV_SOURCE)
        .order_by(PricingSnapshotHistory.accepted_at.desc())
        .limit(limit)
    )
    return [
        AcceptedSnapshot(
            id=row.id,
            accepted_at=normalize_effective_at(row.accepted_at),
            accepted_by=row.accepted_by,
            model_count=row.model_count,
        )
        for row in (await session.execute(stmt)).all()
    ]


async def _discard_unchanged(session: AsyncSession, preview: PricingRefreshPreview) -> bool:
    """Discard a fetch that changes nothing; false when a later fetch replaced it first."""
    try:
        await reject_price_refresh(session, digest=preview.digest)
    except PendingSnapshotChanged:
        return False
    return True


async def poll_price_updates(session: AsyncSession, policy: str, config: GatewayConfig | None = None) -> str:
    """One tick of the scheduled refresh: fetch, then hold or apply per ``policy``.

    Returns what happened, for the log: ``unchanged`` (the fetch matched what is
    active, and nothing is left pending), ``pending`` (held for review) or
    ``applied``. A ``manual`` policy never reaches here.
    """
    # A catalog the metadata refresher fetched within half an interval is reused.
    reuse_within = config.pricing_refresh_interval_seconds / 2 if config is not None else 0.0
    preview = await prepare_price_refresh(session, config, reuse_within=reuse_within)
    if preview.changes_nothing:
        return "unchanged" if await _discard_unchanged(session, preview) else "pending"
    if policy == "auto" and preview.needs_review:
        logger.warning("Leaving the models.dev price update pending for review: %s", preview.review_reason)
        return "pending"
    if policy == "auto":
        try:
            applied = await confirm_price_refresh(session, accepted_by="schedule", digest=preview.digest)
        except PendingSnapshotChanged:
            return "pending"
        return "applied" if applied else "pending"
    return "pending"


async def claim_poll_tick(
    session: AsyncSession, interval_seconds: float, source: str = MODELS_DEV_POLL_CLAIM_SOURCE
) -> bool:
    """Whether this worker owns the next scheduled check, claimed on the row ``source`` names.

    Every worker and every replica runs the poller, and nothing else separates
    them: each compares the fetch against its own in-memory snapshot, so under
    ``auto`` they would each confirm the same update and write a history row for
    it. The claim is one conditional ``UPDATE``, so the first statement to land
    owns the tick and the rest skip it, and it is honored only while it is older
    than one interval, so a worker that dies holding it costs one tick.
    """
    now = datetime.now(UTC)
    result = await session.execute(
        update(PricingSnapshot)
        .where(
            PricingSnapshot.source == source,
            PricingSnapshot.updated_at <= now - timedelta(seconds=interval_seconds),
        )
        .values(updated_at=now)
    )
    # ``rowcount`` lives on CursorResult, and mypy sees the result of ``execute()`` as Result.
    if getattr(result, "rowcount", 0) == 1:
        await session.commit()
        return True
    # Either a sibling holds the tick or the row has never been written; the
    # insert decides which, and losing that race is a sibling's claim too.
    session.add(PricingSnapshot(source=source, snapshot="", updated_at=now))
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return False
    return True


class ModelsDevPriceUpdates:
    """The models.dev price steps of a catalog update.

    What the hosted catalog update calls for its price half. Each step opens a
    session of its own and commits there, as the price refresh functions do, so
    the caller hands over no session and holds none across a step.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config

    async def claim(self, interval_seconds: float) -> bool:
        """Whether this worker owns the next scheduled update."""
        async with create_session() as session:
            return await claim_poll_tick(session, interval_seconds)

    async def fetch(self) -> PricingRefreshPreview | None:
        """Fetch the latest prices and hold them as pending, or ``None`` when nothing changed.

        Raises:
            PricingRefreshError: the data could not be fetched, or was implausible.
        """
        async with create_session() as session:
            preview = await prepare_price_refresh(session, self.config)
            if not preview.changes_nothing:
                return preview
            await _discard_unchanged(session, preview)
            return None

    async def poll(self, policy: str) -> str:
        """One scheduled price check under ``policy``: ``unchanged``, ``pending`` or ``applied``.

        Raises:
            PricingRefreshError: the data could not be fetched, or was implausible.
        """
        async with create_session() as session:
            return await poll_price_updates(session, policy, self.config)

    async def approve(self, digest: str) -> bool:
        """Apply the pending prices an operator reviewed under ``digest``; false when none is pending.

        Raises:
            PendingSnapshotChanged: the pending prices are no longer the reviewed ones.
            PricingRefreshError: they could not be applied.
        """
        async with create_session() as session:
            return await confirm_price_refresh(session, digest=digest)


async def reject_price_refresh(session: AsyncSession, *, digest: str | None = None) -> bool:
    """Discard the pending snapshot without changing active pricing.

    Reject means "not now", not "never": nothing records what was rejected, so
    the next scheduled check under ``review`` re-pends the same update while
    upstream still differs from the active snapshot. With ``digest``, only the
    snapshot that digest names is discarded.
    """

    pending_row = await _lock_pending(session, digest)
    if pending_row is None:
        return False
    deleted = await _delete_pending(session, pending_row.snapshot)
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise PricingRefreshError("Unable to discard the pending models.dev data") from exc
    return deleted


async def _get_active_snapshot_row(session: AsyncSession) -> PricingSnapshot | None:
    result = await session.execute(select(PricingSnapshot).where(PricingSnapshot.source == MODELS_DEV_SOURCE).limit(1))
    return result.scalar_one_or_none()


def _build_generations(
    history: list[tuple[datetime, str]], active: tuple[datetime, str] | None
) -> list[PriceGeneration]:
    """Index the stored snapshots, skipping one that no longer parses.

    The active row normally equals the newest history row; it is added on its
    own only when the history does not carry it.
    """
    rows = list(history)
    if active is not None and all(raw != active[1] for _, raw in rows):
        rows.append(active)
    generations: list[PriceGeneration] = []
    for effective_at, raw in rows:
        try:
            generations.append(
                PriceGeneration(effective_at=normalize_effective_at(effective_at), index=_parse_snapshot(raw))
            )
        except ValueError:
            logger.warning("Ignoring invalid persisted %s pricing snapshot", MODELS_DEV_SOURCE)
    return generations


async def _active_updated_at(session: AsyncSession) -> datetime | None:
    """The active row's last write, read without its payload; ``None`` when absent or unreadable."""
    try:
        return (
            await session.execute(select(PricingSnapshot.updated_at).where(PricingSnapshot.source == MODELS_DEV_SOURCE))
        ).scalar_one_or_none()
    except SQLAlchemyError:
        return None


async def _apply_persisted_snapshots(session: AsyncSession, active_row: PricingSnapshot) -> None:
    """Rebuild the process-wide generations from the history window and the active row."""
    global _applied_updated_at
    active = (active_row.updated_at, active_row.snapshot)
    marker = active_row.updated_at
    rows = (
        await session.execute(
            select(PricingSnapshotHistory.accepted_at, PricingSnapshotHistory.snapshot)
            .where(PricingSnapshotHistory.source == MODELS_DEV_SOURCE)
            .order_by(PricingSnapshotHistory.accepted_at.desc())
            .limit(MAX_RESIDENT_GENERATIONS + 1)
        )
    ).all()
    # A full history window or a longer one means older accepts may be missing.
    complete = len(rows) <= MAX_RESIDENT_GENERATIONS and len(rows) < PRICING_SNAPSHOT_HISTORY_KEEP
    history = [(row.accepted_at, row.snapshot) for row in rows]
    generations = await asyncio.to_thread(_build_generations, history, active)
    _applied_updated_at = marker
    if generations:
        set_accepted_generations(generations, complete=complete)


async def load_persisted_price_snapshot(session: AsyncSession) -> None:
    """Load the accepted models.dev snapshots during startup."""

    active_row = await _get_active_snapshot_row(session)
    if active_row is None:
        return
    await _apply_persisted_snapshots(session, active_row)
    logger.info("Loaded persisted %s pricing snapshot", MODELS_DEV_SOURCE)


async def refresh_price_snapshot(session: AsyncSession) -> None:
    """Re-apply the accepted snapshots when a confirm on another worker changed them.

    The active row only ever appears or advances via ``confirm_price_refresh``, so
    a timestamp equal to what this worker already applied is skipped, and only
    that column is read until it differs.
    """

    updated_at = await _active_updated_at(session)
    if updated_at is None or updated_at == _applied_updated_at:
        return
    active_row = await _get_active_snapshot_row(session)
    if active_row is None:
        return
    await _apply_persisted_snapshots(session, active_row)
    logger.info("Applied updated %s pricing snapshot accepted on another worker", MODELS_DEV_SOURCE)


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
    """Restore the bundled snapshot for app tests."""

    global _applied_updated_at
    reset_generations()
    _applied_updated_at = None


def reset_catalog_refresh_state() -> None:
    """Forget the catalog this worker serves and the stored one it read, so the next tick reads it again."""
    global _catalog_applied_at
    clear_catalog_cache()
    _catalog_applied_at = None


def _catalog_text(catalog: ModelsDevCatalog) -> str:
    return json.dumps(catalog_document(catalog), separators=(",", ":"), ensure_ascii=False)


def _read_stored_catalog(raw: str) -> ModelsDevCatalog:
    return read_catalog_document(json.loads(raw))


async def store_models_dev_catalog(session: AsyncSession, catalog: ModelsDevCatalog) -> bool:
    """Store a downloaded catalog for every worker to serve; false when the stored one is the same.

    An unchanged catalog is not written again, so the other workers read the
    row only when it has changed.
    """
    global _catalog_applied_at
    raw = await asyncio.to_thread(_catalog_text, catalog)
    row = await session.get(PricingSnapshot, MODELS_DEV_CATALOG_SOURCE)
    if row is not None and row.snapshot == raw:
        return False
    now = datetime.now(UTC)
    if row is None:
        session.add(PricingSnapshot(source=MODELS_DEV_CATALOG_SOURCE, snapshot=raw, updated_at=now))
    else:
        row.snapshot = raw
        row.updated_at = now
    await session.commit()
    _catalog_applied_at = now
    return True


async def load_stored_models_dev_catalog(session: AsyncSession) -> bool:
    """Serve the stored catalog when it changed since this worker last read it; false when it did not.

    A row this version cannot read is logged once and skipped, so the worker
    keeps what it serves.
    """
    global _catalog_applied_at
    updated_at = (
        await session.execute(
            select(PricingSnapshot.updated_at).where(PricingSnapshot.source == MODELS_DEV_CATALOG_SOURCE)
        )
    ).scalar_one_or_none()
    if updated_at is None:
        return False
    updated_at = normalize_effective_at(updated_at)
    if updated_at == _catalog_applied_at:
        return False
    raw = (
        await session.execute(
            select(PricingSnapshot.snapshot).where(PricingSnapshot.source == MODELS_DEV_CATALOG_SOURCE)
        )
    ).scalar_one()
    _catalog_applied_at = updated_at
    try:
        catalog = await asyncio.to_thread(_read_stored_catalog, raw)
    except ValueError as exc:
        logger.warning("Ignoring the stored models.dev catalog: %s", exc)
        return False
    serve_catalog(catalog, updated_at)
    return True


async def _retry_catalog_download_soon(session: AsyncSession, interval_seconds: float) -> None:
    """Move the download claim back, so the next download is tried ``CATALOG_RETRY_SECONDS`` from now."""
    retry_at = datetime.now(UTC) - timedelta(seconds=interval_seconds - CATALOG_RETRY_SECONDS)
    await session.execute(
        update(PricingSnapshot)
        .where(PricingSnapshot.source == MODELS_DEV_CATALOG_CLAIM_SOURCE)
        .values(updated_at=retry_at)
    )
    await session.commit()


async def download_models_dev_catalog_if_due(config: GatewayConfig) -> bool:
    """Download and store the catalog when this worker claims the download; false when it did not.

    One worker in the deployment downloads per interval. A failed download is
    logged, keeps what is stored, and is tried again ``CATALOG_RETRY_SECONDS``
    later.
    """
    interval = catalog_refresh_interval(config)
    async with create_session() as session:
        # Read first, so a tick with no download due writes nothing.
        claimed_at = (
            await session.execute(
                select(PricingSnapshot.updated_at).where(PricingSnapshot.source == MODELS_DEV_CATALOG_CLAIM_SOURCE)
            )
        ).scalar_one_or_none()
        if claimed_at is not None and datetime.now(UTC) - normalize_effective_at(claimed_at) < timedelta(
            seconds=interval
        ):
            return False
        if not await claim_poll_tick(session, interval, MODELS_DEV_CATALOG_CLAIM_SOURCE):
            return False
    try:
        catalog = await fetch_models_dev_document(reuse_within=interval / 2, fill_cache=True)
    except ModelsDevFetchError as exc:
        level = logging.ERROR if isinstance(exc, ModelsDevRefusedError) else logging.WARNING
        logger.log(level, "%s; keeping the stored copy", exc)
        async with create_session() as session:
            await _retry_catalog_download_soon(session, interval)
        return False
    async with create_session() as session:
        await store_models_dev_catalog(session, catalog)
    return True


async def run_catalog_refresher(config: GatewayConfig, interval: float = PRICE_SNAPSHOT_REFRESH_TTL_SECONDS) -> None:
    """Keep the models.dev catalog every worker serves current, so no read waits on a download.

    Each tick serves the stored catalog when another worker changed it, then
    downloads and stores a new one when this worker claims the download. It
    starts at once, so a fresh worker serves the stored catalog within one tick.

    Re-checks ``background_catalog_enabled`` every tick rather than being started
    only when it holds: ``models_dev_metadata`` and ``models_dev_cache_ttl_seconds``
    are both runtime-settable, so the loop has to be able to start and stop
    downloading without a restart. The stored catalog is served either way.
    Every error is logged and retried on the next tick. Cancelled at shutdown.
    """
    while True:
        try:
            async with create_session() as session:
                await load_stored_models_dev_catalog(session)
            if background_catalog_enabled(config):
                await download_models_dev_catalog_if_due(config)
        except asyncio.CancelledError:
            raise
        except DATABASE_ERRORS as exc:
            logger.warning(
                "models.dev catalog refresh failed (%s); retrying in %ss", describe_database_error(exc), interval
            )
        except Exception:
            logger.warning("models.dev catalog refresh failed; retrying in %ss", interval, exc_info=True)
        await asyncio.sleep(interval)
