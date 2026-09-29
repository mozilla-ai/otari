"""Keep a running request's idempotency claim from lapsing."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

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
    After a failed renewal the next attempt comes sooner, and sooner again as the lease runs out,
    so the claim lapses only when the database stays unreachable for about the whole lease.
    Each renewal runs in a Unit of Work of its own, because the request's belongs to the request's task.
    """
    cadence = lease_sec / 3
    shortest_retry = lease_sec / 60
    lease_ends = time.monotonic() + lease_sec
    delay = cadence
    while True:
        await asyncio.sleep(delay)
        try:
            async with create_unit_of_work() as uow:
                if not await build_service(uow).renew(request, claimed):
                    return
            lease_ends = time.monotonic() + lease_sec
            delay = cadence
        except Exception:
            # A lapsed lease lets a retry run the request again, so no error ends the renewal.
            logger.warning("Could not renew an idempotency claim; retrying", exc_info=True)
            delay = min(cadence, max((lease_ends - time.monotonic()) / 4, shortest_retry))
