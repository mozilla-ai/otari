"""Keep a running request's idempotency claim from lapsing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from gateway.core.database import DATABASE_ERRORS
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.inference._idempotency import Claimed, IdempotencyService, IdempotentRequest


async def keep_claim_alive(
    request: IdempotentRequest,
    claimed: Claimed,
    lease_sec: float,
    build_service: Callable[[UnitOfWork], IdempotencyService],
) -> None:
    """Renew the claim every third of its lease until canceled, or until the claim is no longer this request's.

    A retry can then take the key over only once the worker running the request is gone.
    Each renewal runs in a Unit of Work of its own, because the request's belongs to the request's task.
    """
    while True:
        await asyncio.sleep(lease_sec / 3)
        try:
            async with create_unit_of_work() as uow:
                if not await build_service(uow).renew(request, claimed):
                    return
        except DATABASE_ERRORS:
            logger.warning("Could not renew an idempotency claim; retrying", exc_info=True)
