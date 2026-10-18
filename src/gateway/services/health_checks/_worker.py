"""The poll every replica runs. Only the replica that claims a due run carries it out."""

import asyncio
from collections.abc import Callable

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.health_checks._service import HealthCheckService, ListOfferings

POLL_SECONDS = 60.0


async def run_health_check_worker(
    build_service: Callable[[UnitOfWork], HealthCheckService],
    list_offerings: ListOfferings,
    interval: float = POLL_SECONDS,
) -> None:
    """Poll for a due run forever. Every error is logged and retried on the next poll. Cancelled at shutdown."""
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_unit_of_work() as uow:
                run = await build_service(uow).run_if_due(list_offerings)
            if run is not None:
                logger.info("Health checks: %s of %s answered", run.passed, run.total)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Health check run failed; retrying in %ss", interval, exc_info=True)
