"""The switched-on conflict on SQLite, which names the column and not the index.

The migration test covers the index itself. This covers the repository's
reading of the refusal, which differs by dialect: PostgreSQL names the index,
SQLite the table and column.
"""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.routing_exceptions import AgentRecommenderSwitchConflictError
from gateway.repositories.routing import AgentRecommenderRepository

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def store() -> AsyncIterator[tuple[UnitOfWork, AgentRecommenderRepository]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            uow = UnitOfWork(session)
            yield uow, AgentRecommenderRepository(uow)
    finally:
        await engine.dispose()


async def test_a_second_router_switched_on_is_a_conflict(
    store: tuple[UnitOfWork, AgentRecommenderRepository],
) -> None:
    uow, rows = store
    async with uow:
        await rows.insert(provider="typesafe", model="a", enabled=True)

    with pytest.raises(AgentRecommenderSwitchConflictError):
        async with uow:
            await rows.insert(provider="typesafe", model="b", enabled=True)


async def test_a_second_router_switched_off_fits(store: tuple[UnitOfWork, AgentRecommenderRepository]) -> None:
    uow, rows = store
    async with uow:
        await rows.insert(provider="typesafe", model="a", enabled=True)
        await rows.insert(provider="typesafe", model="b", enabled=False)
        assert [row.model for row in await rows.list_all()] == ["a", "b"]
