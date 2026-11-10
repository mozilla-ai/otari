"""The poll every replica runs. Only the replica that claims a due sync carries it out."""

import asyncio
from collections.abc import Callable

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.services.mcp_catalog._registry import RegistryClient
from gateway.services.mcp_catalog._service import McpCatalogService

POLL_SECONDS = 60.0


async def run_mcp_catalog_sync_worker(
    build_service: Callable[[UnitOfWork], McpCatalogService],
    config: GatewayConfig,
    interval: float = POLL_SECONDS,
) -> None:
    """Poll for a due sync forever. Every error is logged and retried on the next poll. Cancelled at shutdown.

    While ``mcp_catalog_sync`` is ``off`` a poll neither reads the database nor opens a connection anywhere.
    """
    while True:
        await asyncio.sleep(interval)
        if config.mcp_catalog_sync == "off":
            continue
        try:
            async with create_unit_of_work() as uow, RegistryClient() as registry:
                status = await build_service(uow).sync_if_due(registry.fetch_page)
            if status is not None:
                logger.info(
                    "MCP catalog sync %s: %s pages, %s kept, %s removed",
                    status.outcome,
                    status.pages,
                    status.kept,
                    status.removed,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("MCP catalog sync failed; retrying in %ss", interval, exc_info=True)
