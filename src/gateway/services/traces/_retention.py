"""Delete traces once their retention has passed."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from gateway.log_config import logger
from gateway.ports.trace_storage_port import TraceStoragePort


async def run_trace_retention(store: Callable[[], TraceStoragePort], *, retention_days: int, interval: float) -> None:
    """Expire traces idle longer than ``retention_days``, once per ``interval``, until shutdown.

    The store is resolved per tick, as a request resolves it, so a binding that
    cannot be built is a logged, retried failure here rather than a startup error.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            removed = await store().expire(datetime.now(UTC) - timedelta(days=retention_days))
            if removed:
                logger.info("Trace retention removed %d traces", removed)
        except asyncio.CancelledError:
            raise
        except Exception:
            # An escaping error would end this worker, and nothing restarts it.
            logger.warning("Trace retention failed; retrying in %ss", interval, exc_info=True)
