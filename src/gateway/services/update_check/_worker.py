"""The poll every replica runs. Only the replica that claims a due check sends it."""

import asyncio
from collections.abc import Callable

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.update_check._service import UpdateCheckService

POLL_SECONDS = 60.0


async def run_update_check_worker(
    build_service: Callable[[UnitOfWork], UpdateCheckService], interval: float = POLL_SECONDS
) -> None:
    """Poll for a due check forever. Every error is logged and retried on the next poll. Cancelled at shutdown."""
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_unit_of_work() as uow:
                await build_service(uow).check_if_due()
        except asyncio.CancelledError:
            raise
        except DATABASE_ERRORS as exc:
            logger.warning("Update check poll failed: %s; retrying in %ss", describe_database_error(exc), interval)
        except Exception as exc:
            # The class only: an error a rolled-back unit of work raises chains the database error, and its row.
            logger.warning("Update check poll failed (%s); retrying in %ss", type(exc).__name__, interval)
