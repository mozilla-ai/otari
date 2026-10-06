"""Schedule bounded cleanup of expired and soft-deleted files."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.files._service import FileService

# Bound work per tick so a large backlog drains across ticks rather than
# keeping one job's session open until every candidate has been scanned.
_MAX_SWEEP_PASSES = 10


async def run_file_sweeper(
    interval: float, build_service: Callable[[UnitOfWork], FileService], *, batch_size: int = 200
) -> None:
    """Run bounded cleanup jobs until shutdown, retrying failures on the next tick."""
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_unit_of_work() as uow:
                files = build_service(uow)
                cursor: tuple[datetime, str] | None = None
                for _ in range(_MAX_SWEEP_PASSES):
                    batch = await files.sweep(batch_size=batch_size, after=cursor)
                    if batch.seen < batch_size:
                        break
                    cursor = batch.cursor
        except asyncio.CancelledError:
            raise
        except Exception:
            # An escaping error would end this worker, and nothing restarts it.
            logger.warning("File sweep failed; retrying in %ss", interval, exc_info=True)
