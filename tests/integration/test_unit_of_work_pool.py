"""A Unit of Work hands its connection back to the pool when a block ends.

A session opens a transaction on its first statement and holds a pooled
connection until the transaction ends. The count lives on the real
``AsyncAdaptedQueuePool``; SQLite runs on ``NullPool``, which keeps none, so
this is PostgreSQL-only.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager

import pytest
import pytest_asyncio
from sqlalchemy import text

from gateway.api.deps import get_unit_of_work
from gateway.core.config import GatewayConfig
from gateway.core.database import LOG_POOL, REQUEST_POOL, dispose_db, get_db, init_db, pool_stats
from gateway.core.unit_of_work import UnitOfWork, create_log_unit_of_work, create_unit_of_work, session_for


@pytest_asyncio.fixture
async def engines(test_config: GatewayConfig, clean_database: None) -> AsyncIterator[None]:
    init_db(test_config)
    try:
        yield
    finally:
        await dispose_db()


def _checked_out(pool: str) -> int:
    return pool_stats()[pool].checked_out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("helper", "pool"),
    [(create_unit_of_work, REQUEST_POOL), (create_log_unit_of_work, LOG_POOL)],
    ids=["request pool", "log pool"],
)
async def test_a_block_that_commits_returns_its_connection(
    engines: None, helper: Callable[[], AbstractAsyncContextManager[UnitOfWork]], pool: str
) -> None:
    async with helper() as uow:
        async with uow:
            await session_for(uow).execute(text("SELECT 1"))
            assert _checked_out(pool) == 1

        assert _checked_out(pool) == 0, "the block ended but its session kept the connection"


@pytest.mark.asyncio
async def test_a_block_that_rolls_back_returns_its_connection(engines: None) -> None:
    async with create_unit_of_work() as uow:
        with pytest.raises(ValueError, match="step failed"):
            async with uow:
                await session_for(uow).execute(text("SELECT 1"))
                raise ValueError("step failed")

        assert _checked_out(REQUEST_POOL) == 0, "the block rolled back but its session kept the connection"


@pytest.mark.asyncio
async def test_the_request_unit_of_work_returns_its_connection_before_the_request_ends(engines: None) -> None:
    requests = get_db()
    session = await anext(requests)
    try:
        uow = get_unit_of_work(session)
        async with uow:
            await session_for(uow).execute(text("SELECT 1"))

        assert _checked_out(REQUEST_POOL) == 0, "the request session held its connection past the block"
    finally:
        await requests.aclose()
