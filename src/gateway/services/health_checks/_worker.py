"""The poll every replica runs. Only the replica that claims a due run carries it out."""

import asyncio
from collections.abc import Callable

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.health_checks._offerings import list_hosted_offerings
from gateway.services.health_checks._service import HealthCheckOffering, HealthCheckService
from gateway.services.providers import HostedProviderService

POLL_SECONDS = 60.0


async def run_health_check_worker(
    build_service: Callable[[UnitOfWork], HealthCheckService],
    build_hosted: Callable[[UnitOfWork], HostedProviderService],
    config: GatewayConfig,
    interval: float = POLL_SECONDS,
) -> None:
    """Poll for a due run forever. Every error is logged and retried on the next poll. Cancelled at shutdown."""

    async def list_offerings() -> list[HealthCheckOffering]:
        # Closed before the run probes anything, so no transaction is held across a minute of dialing.
        async with create_unit_of_work() as uow:
            return await list_hosted_offerings(build_hosted(uow), config)

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
