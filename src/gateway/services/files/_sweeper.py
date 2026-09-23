"""Reclaiming the bytes behind expired and soft-deleted files.

Expiry alone only hides a file (``fetch_file`` answers 404 for it, and so does
a listing); this is what gives its storage back. Runs as one of the lifespan
workers ``main.py`` starts, standalone only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from gateway.core.unit_of_work import create_unit_of_work
from gateway.log_config import logger
from gateway.ports.file_storage_port import FileStoragePort
from gateway.repositories.files import FileRepositories, FileRepository

# Passes one tick may make before waiting again, so a large backlog drains over
# several ticks instead of holding one session open until it is done.
_MAX_SWEEP_PASSES = 10


@dataclass(frozen=True)
class SweepBatch:
    """What one pass of :func:`sweep_files` did, and where the next one starts."""

    reclaimed: int
    # Rows the pass looked at, reclaimed or not. A short batch means the
    # backlog is drained.
    seen: int
    # The last row's ``(created_at, id)``, so a following pass in the same tick
    # starts past it rather than re-reading rows whose blob would not delete.
    cursor: tuple[datetime, str] | None


async def sweep_files(
    files: FileRepository,
    file_store: FileStoragePort,
    *,
    batch_size: int,
    after: tuple[datetime, str] | None = None,
) -> SweepBatch:
    """Reclaim one batch of expired or soft-deleted files: their bytes, then their rows.

    Does not commit: the caller's Unit of Work block does. A blob that is
    already gone is not an error (the delete route removes the bytes
    best-effort before this ever sees the row); any other storage failure
    leaves the row in place for a later tick rather than orphaning bytes
    nothing references.
    """
    records = await files.reclaimable(batch_size=batch_size, after=after)
    reclaimed: list[str] = []
    for record in records:
        try:
            # A row with no stored bytes has no blob to remove.
            if record.storage_ref is not None:
                await file_store.delete(record.storage_ref)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("file sweep: could not remove blob %s for %s: %s", record.storage_ref, record.id, exc)
            continue
        reclaimed.append(record.id)
    if reclaimed:
        await files.remove_all(reclaimed)
        logger.info("file sweep: reclaimed %d file(s)", len(reclaimed))
    cursor = (records[-1].created_at, records[-1].id) if records else None
    return SweepBatch(reclaimed=len(reclaimed), seen=len(records), cursor=cursor)


async def run_file_sweeper(interval: float, file_store: FileStoragePort, *, batch_size: int = 200) -> None:
    """Reclaim expired and deleted files on a timer, forever. Cancelled at shutdown.

    Every error is swallowed and retried on the next tick, matching the other
    lifespan tasks: a storage or database blip must not kill the sweeper,
    because nothing would restart it.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_unit_of_work() as uow:
                files = FileRepositories.on(uow).files
                cursor: tuple[datetime, str] | None = None
                for _ in range(_MAX_SWEEP_PASSES):
                    async with uow:
                        batch = await sweep_files(files, file_store, batch_size=batch_size, after=cursor)
                    if batch.seen < batch_size:
                        break
                    cursor = batch.cursor
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("File sweep failed; retrying in %ss", interval, exc_info=True)
