"""Memory's background loop: every few seconds, decide what agents proposed; once a day, apply retention."""

import asyncio
from collections.abc import Callable
from time import monotonic

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.memory._service import MemoryService

POLL_SECONDS = 10.0
SWEEP_SECONDS = 24 * 60 * 60.0
# Proposals one workspace has decided in one pass, so a busy workspace cannot hold up the rest.
PROPOSALS_PER_PASS = 20


async def run_consolidator(
    build_service: Callable[[UnitOfWork], MemoryService],
    interval: float = POLL_SECONDS,
    sweep_every: float = SWEEP_SECONDS,
) -> None:
    """Consolidate forever, and sweep once per ``sweep_every``, the first time on the first pass.

    Every error is logged and retried on the next pass. Cancelled at shutdown. Every replica runs it:
    a proposal another replica decided first is skipped, and a sweep deletes nothing twice.
    """
    next_sweep = monotonic()
    while True:
        await asyncio.sleep(interval)
        try:
            await consolidate_once(build_service)
            if monotonic() >= next_sweep:
                await sweep_once(build_service)
                next_sweep = monotonic() + sweep_every
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Memory pass failed; retrying in %ss", interval, exc_info=True)


async def consolidate_once(build_service: Callable[[UnitOfWork], MemoryService]) -> int:
    """One pass over every workspace with memory on; how many proposals were decided."""
    async with create_unit_of_work() as uow:
        workspaces = await build_service(uow).workspaces_with_memory()
    decided = 0
    for workspace_id in workspaces:
        # A Unit of Work per workspace, so one workspace's failure leaves the others' decisions standing.
        async with create_unit_of_work() as uow:
            decided += await build_service(uow).consolidate(workspace_id, limit=PROPOSALS_PER_PASS)
    if decided:
        logger.info("Memory consolidation decided %s proposals", decided)
    return decided


async def sweep_once(build_service: Callable[[UnitOfWork], MemoryService]) -> int:
    """One retention pass over every workspace with memory settings; how many entries it deleted."""
    async with create_unit_of_work() as uow:
        workspaces = await build_service(uow).workspaces_with_settings()
    deleted = 0
    for workspace_id in workspaces:
        async with create_unit_of_work() as uow:
            swept = await build_service(uow).sweep(workspace_id)
        deleted += swept.entries
        if swept.entries or swept.proposals or swept.keys:
            logger.info(
                "Memory retention in workspace %s deleted %s entries, %s proposals, %s read records and %s keys",
                workspace_id,
                swept.entries,
                swept.proposals,
                swept.reads,
                swept.keys,
            )
    return deleted
