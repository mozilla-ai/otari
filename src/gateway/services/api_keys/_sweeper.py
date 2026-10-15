"""Schedule the deletion of expired keys a service key minted."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.api_keys._service import ApiKeyService


async def run_end_user_key_sweeper(
    interval: float, retention_sec: int, build_service: Callable[[UnitOfWork], ApiKeyService]
) -> None:
    """Delete minted keys ``retention_sec`` past their expiry until shutdown, retrying failures on the next tick."""
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_unit_of_work() as uow:
                deleted = await build_service(uow).sweep_expired_end_user_keys(retention_sec)
            if deleted:
                logger.info("End-user key sweep deleted %d expired keys", deleted)
        except asyncio.CancelledError:
            raise
        except Exception:
            # An escaping error would end this worker, and nothing restarts it.
            logger.warning("End-user key sweep failed; retrying in %ss", interval, exc_info=True)
