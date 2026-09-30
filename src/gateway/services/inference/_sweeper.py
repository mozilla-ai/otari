"""Schedule the deletion of expired idempotency records."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.inference._idempotency import IdempotencyService


async def run_idempotency_sweeper(interval: float, build_service: Callable[[UnitOfWork], IdempotencyService]) -> None:
    """Delete expired idempotency records until shutdown, retrying failures on the next tick."""
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_unit_of_work() as uow:
                deleted = await build_service(uow).sweep()
            if deleted:
                logger.info("Idempotency sweep deleted %d expired records", deleted)
        except asyncio.CancelledError:
            raise
        except Exception:
            # An escaping error would end this worker, and nothing restarts it.
            logger.warning("Idempotency sweep failed; retrying in %ss", interval, exc_info=True)
