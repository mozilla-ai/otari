"""Delete traces, and the content captured in them, once their retention has passed."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from gateway.log_config import logger
from gateway.ports.trace_storage_port import TraceStoragePort
from gateway.services.traces._content_keys import ContentKeys


async def run_trace_retention(
    store: Callable[[], TraceStoragePort],
    *,
    retention_days: int,
    content_retention_days: int,
    keys: Callable[[], ContentKeys],
    interval: float,
) -> None:
    """Expire traces idle longer than ``retention_days``, and content older than its own retention, until shutdown.

    The store is resolved per tick, as a request resolves it, so a binding that
    cannot be built is a logged, retried failure here rather than a startup error.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            now = datetime.now(UTC)
            removed = await store().expire(now - timedelta(days=retention_days))
            content_cutoff = now - timedelta(days=content_retention_days)
            removed_content = await store().expire_content(content_cutoff)
            # Expiry deletes each session's key with the session; this catches keys
            # minted for sessions the writer never stored.
            destroyed = await keys().destroy_orphaned(now - timedelta(days=1))
            if removed or removed_content or destroyed:
                logger.info(
                    "Trace retention removed %d traces, %d content blobs and %d orphaned keys",
                    removed,
                    removed_content,
                    destroyed,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            # An escaping error would end this worker, and nothing restarts it.
            logger.warning("Trace retention failed; retrying in %ss", interval, exc_info=True)
