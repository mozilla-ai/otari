"""Which addresses the deployment's email blocklist matches, read on SQLite."""

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.identity import BlockedEmailRepository
from gateway.services.identity import EmailBlocklist

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def store() -> AsyncIterator[tuple[UnitOfWork, BlockedEmailRepository]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            uow = UnitOfWork(session)
            yield uow, BlockedEmailRepository(uow)
    finally:
        await engine.dispose()


async def _block(store: tuple[UnitOfWork, BlockedEmailRepository], *patterns: str) -> None:
    uow, repository = store
    async with uow:
        for pattern in patterns:
            await repository.add(pattern=pattern, reason=None, created_by_user_id=uuid.uuid4())


async def test_an_address_is_matched_by_its_own_entry_or_by_an_entry_for_a_domain_above_it(
    store: tuple[UnitOfWork, BlockedEmailRepository],
) -> None:
    await _block(store, "ada@example.com", "blocked.test")
    blocklist = EmailBlocklist(*store)

    matched = await blocklist.blocked_among(
        ["Ada@Example.com", "grace@example.com", "bob@sub.blocked.test", "eve@other.test"]
    )

    assert matched == {"Ada@Example.com", "bob@sub.blocked.test"}


async def test_a_value_with_no_domain_is_never_matched(
    store: tuple[UnitOfWork, BlockedEmailRepository],
) -> None:
    await _block(store, "example.com")
    blocklist = EmailBlocklist(*store)

    assert await blocklist.blocked_among(["not-an-address"]) == set()
    assert await blocklist.blocked_among([]) == set()
