"""Persistence for batch jobs: ownership records and idempotent accounting.

The gateway stores one row per created batch (:class:`BatchRecord`). That record
is the durable anchor the route uses to enforce strict ownership and to bill and
log a completed batch's usage exactly once. Batches created before this table
existed carry no record; the route keeps the metadata-anchored fallback for them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from gateway.log_config import logger
from gateway.models.entities import BatchRecord

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession


async def record_batch(
    db: AsyncSession,
    *,
    batch_id: str,
    provider: str,
    user_id: str,
    api_key_id: str | None,
    model: str,
) -> None:
    """Persist an ownership record for a newly created batch.

    Best-effort: the provider has already accepted the batch, so a failure to
    persist the record must not fail the request. Ownership then degrades to the
    metadata-anchored fallback (the ``otari_user_id`` marker) for that batch.
    """
    db.add(
        BatchRecord(
            id=batch_id,
            provider=provider,
            user_id=user_id,
            api_key_id=api_key_id,
            model=model,
        )
    )
    try:
        await db.commit()
    except SQLAlchemyError as e:
        await db.rollback()
        logger.warning("Failed to persist batch record for '%s': %s", batch_id, e)


async def get_batch_record(db: AsyncSession, batch_id: str) -> BatchRecord | None:
    """Return the stored record for ``batch_id`` (or None for legacy batches)."""
    return (await db.execute(select(BatchRecord).where(BatchRecord.id == batch_id))).scalar_one_or_none()


async def get_batch_records(db: AsyncSession, batch_ids: Iterable[str]) -> dict[str, BatchRecord]:
    """Return stored records for ``batch_ids`` keyed by id, for list filtering."""
    ids = list(batch_ids)
    if not ids:
        return {}
    rows = (await db.execute(select(BatchRecord).where(BatchRecord.id.in_(ids)))).scalars().all()
    return {row.id: row for row in rows}


async def claim_batch_accounting(db: AsyncSession, batch_id: str) -> bool:
    """Atomically claim the one-time results accounting for a batch.

    Returns True to the single caller that transitions ``results_accounted_at``
    from NULL to now, and False for every later retrieval (retries, polling
    clients). Only the winning caller logs usage and folds the batch cost into
    the user's spend, so repeated retrievals of the same completed batch do not
    double-count.

    This does not commit: the caller commits the claim in the same transaction
    as the spend fold, so the two are atomic. If the spend fold fails (or the
    process dies) before that commit, the claim rolls back and a later retrieval
    re-accounts, rather than leaving the batch marked-accounted-but-unbilled. The
    UPDATE still takes the row lock immediately, so a concurrent claim on the same
    batch blocks and then sees ``results_accounted_at`` set, returning False.
    """
    result = await db.execute(
        update(BatchRecord)
        .where(BatchRecord.id == batch_id, BatchRecord.results_accounted_at.is_(None))
        .values(results_accounted_at=datetime.now(UTC))
        .execution_options(synchronize_session=False)
    )
    # getattr with a default: mypy sees .execute() as Result (no rowcount);
    # rowcount lives on CursorResult. Matches reserve_budget/_cas_reset_user_budget.
    return bool(getattr(result, "rowcount", 0))
