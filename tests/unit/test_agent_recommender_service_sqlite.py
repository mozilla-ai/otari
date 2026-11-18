"""Which stored router a create switches on, run against SQLite.

The first router added is switched on. A later one waits, also where an
operator has switched every stored router off: that state is theirs, and a
new row does not undo it.
"""

from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.routing import AgentRecommenderRepository
from gateway.schemas.routing import AgentRecommenderCreate, AgentRecommenderUpdate
from gateway.services.routing import AgentRecommenderService, reset_agent_recommenders
from gateway.services.secret_box import generate_secret_key

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_agent_recommenders()
    yield
    reset_agent_recommenders()


@pytest_asyncio.fixture
async def service() -> AsyncIterator[AgentRecommenderService]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            uow = UnitOfWork(session)
            yield AgentRecommenderService(uow, AgentRecommenderRepository(uow))
    finally:
        await engine.dispose()


def _create(model: str) -> AgentRecommenderCreate:
    return AgentRecommenderCreate(
        provider="typesafe", model=model, api_key="sk-decider-wxyz", input_price_per_million=1.0
    )


async def test_the_first_router_is_switched_on_and_a_later_one_waits(service: AgentRecommenderService) -> None:
    first = await service.create(_create("a"))
    second = await service.create(_create("b"))

    assert (first.enabled, second.enabled) == (True, False)


async def test_a_router_added_while_every_one_is_off_stays_off(service: AgentRecommenderService) -> None:
    first = await service.create(_create("a"))
    await service.update(first.id, AgentRecommenderUpdate(enabled=False))

    second = await service.create(_create("b"))

    assert second.enabled is False
    assert [row.enabled for row in (await service.list()).data] == [False, False]
