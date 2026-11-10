"""The consolidator's loop: every few seconds, decide what each workspace's agents proposed."""

import asyncio
from collections.abc import Callable

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.memory._service import MemoryService

POLL_SECONDS = 10.0
# Proposals one workspace has decided in one pass, so a busy workspace cannot hold up the rest.
PROPOSALS_PER_PASS = 20


async def run_consolidator(
    build_service: Callable[[UnitOfWork], MemoryService], interval: float = POLL_SECONDS
) -> None:
    """Consolidate forever. Every error is logged and retried on the next pass. Cancelled at shutdown."""
    while True:
        await asyncio.sleep(interval)
        try:
            await consolidate_once(build_service)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Memory consolidation pass failed; retrying in %ss", interval, exc_info=True)


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
