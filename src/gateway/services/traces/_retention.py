"""Delete traces once their retention has passed."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.traces._service import TraceService


async def run_trace_retention(
    build_service: Callable[[UnitOfWork], TraceService], *, retention_days: int, max_age_days: int, interval: float
) -> None:
    """Expire traces idle longer than ``retention_days`` or older than ``max_age_days``, once per ``interval``."""
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_unit_of_work() as uow:
                now = datetime.now(UTC)
                removed = await build_service(uow).expire(
                    idle_before=now - timedelta(days=retention_days), started_before=now - timedelta(days=max_age_days)
                )
            if removed:
                logger.info("Trace retention removed %d traces", removed)
        except asyncio.CancelledError:
            raise
        except Exception:
            # An escaping error would end this worker, and nothing restarts it.
            logger.warning("Trace retention failed; retrying in %ss", interval, exc_info=True)
