"""How a hosted write rebuilds the catalog selector index once it commits."""

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import get_config
from gateway.api.deps.ports import RequestSessionModelProviderPortDep
from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error, get_db
from gateway.log_config import logger
from gateway.services.selector_index_service import rebuild_selector_index


def get_selector_reindex(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    model_provider: RequestSessionModelProviderPortDep,
) -> Callable[[], Awaitable[None]]:
    """Rebuild the catalog selector index on the request's session, for a route to call once its write commits.

    Without it a model a write switches on is listed by ``GET /models`` at once
    but answers ``invalid_model`` by its catalog id until the refresher's next
    tick. A failed rebuild is logged rather than raised: the write has already
    committed, and the next tick retries it.
    """

    async def reindex() -> None:
        try:
            await rebuild_selector_index(db, config, model_provider=model_provider)
        except DATABASE_ERRORS as exc:
            logger.warning(
                "catalog selector index rebuild after a write failed (%s); the next tick retries",
                describe_database_error(exc),
            )
        except Exception:
            logger.warning("catalog selector index rebuild after a write failed; the next tick retries", exc_info=True)

    return reindex


SelectorReindexDep = Annotated[Callable[[], Awaitable[None]], Depends(get_selector_reindex)]
