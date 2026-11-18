"""Delete traces once their retention has passed."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.traces._service import TraceService


async def run_trace_retention(
    build_service: Callable[[UnitOfWork], TraceService],
    *,
    retention_days: int,
    max_age_days: int,
    interval: float,
) -> None:
    """Expire traces idle longer than ``retention_days`` or older than ``max_age_days``, once per ``interval``.

    The ids of erased users go too, once they can no longer be needed. Each step runs on its
    own, so one that keeps failing never holds back the others.
    """

    async def traces(service: TraceService) -> int:
        now = datetime.now(UTC)
        return await service.expire(
            idle_before=now - timedelta(days=retention_days), started_before=now - timedelta(days=max_age_days)
        )

    async def erasures(service: TraceService) -> int:
        return await service.expire_erasures(now=datetime.now(UTC))

    steps: tuple[tuple[str, Callable[[TraceService], Awaitable[int]]], ...] = (
        ("traces", traces),
        ("user ids", erasures),
    )
    while True:
        await asyncio.sleep(interval)
        for name, step in steps:
            await _run_step(build_service, name, step, interval)


async def _run_step(
    build_service: Callable[[UnitOfWork], TraceService],
    name: str,
    step: Callable[[TraceService], Awaitable[int]],
    interval: float,
) -> None:
    try:
        async with create_unit_of_work() as uow:
            removed = await step(build_service(uow))
    except asyncio.CancelledError:
        raise
    except DATABASE_ERRORS as exc:
        logger.warning(
            "Trace retention could not remove %s (%s); retrying in %ss", name, describe_database_error(exc), interval
        )
        return
    except Exception as exc:
        # An escaping error would end this worker, and nothing restarts it.
        logger.warning("Trace retention could not remove %s (%s); retrying in %ss", name, type(exc).__name__, interval)
        return
    if removed:
        logger.info("Trace retention removed %d %s", removed, name)
