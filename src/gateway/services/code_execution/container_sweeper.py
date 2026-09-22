"""Dropping the rows of code-execution containers whose clocks ran out.

The sandbox behind a row is reclaimed by the provider, which was told the idle
timeout when the request released it; a resume past either clock is already
refused and drops the row it read. This is the third path, for the rows nobody
resumes again: without it a busy deployment accumulates one dead row per
conversation forever. Runs as one of the lifespan workers ``main.py`` starts,
standalone only, and only when reuse is on.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.repositories.code_execution import delete_container_rows, expired_container_ids

# Rows are small and the provider does the real reclaiming, so the tick is
# unhurried: a row outliving its clock by a few minutes costs nothing.
SWEEP_INTERVAL_S = 300.0
# Passes one tick may make before waiting again, so a large backlog drains over
# several ticks instead of holding one session open until it is done.
_MAX_SWEEP_PASSES = 10


async def sweep_expired_containers(uow: UnitOfWork, *, batch_size: int) -> int:
    """Drop one batch of rows past either clock. Does not commit: the caller's block does."""
    ids = await expired_container_ids(uow, now=datetime.now(UTC), batch_size=batch_size)
    if ids:
        await delete_container_rows(uow, ids)
        logger.info("container sweep: dropped %d expired container(s)", len(ids))
    return len(ids)


async def run_sandbox_container_sweeper(interval: float = SWEEP_INTERVAL_S, *, batch_size: int = 200) -> None:
    """Drop expired container rows on a timer, forever. Cancelled at shutdown.

    Every error is swallowed and retried on the next tick, matching the other
    lifespan tasks: a database blip must not kill the sweeper, because nothing
    would restart it.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_unit_of_work() as uow:
                for _ in range(_MAX_SWEEP_PASSES):
                    async with uow:
                        dropped = await sweep_expired_containers(uow, batch_size=batch_size)
                    if dropped < batch_size:
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Container sweep failed; retrying in %ss", interval, exc_info=True)
