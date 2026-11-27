"""A fold's reads of the people signed in: one read for several servers, and the ids alone where only they are read."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401 - registers every table on the metadata
from gateway.models.tools import McpUserConnection
from gateway.repositories.tools import McpUserConnectionRepository

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield session
    finally:
        await engine.dispose()


def _connection(server_id: uuid.UUID, user_id: str) -> McpUserConnection:
    return McpUserConnection(
        server_id=server_id,
        user_id=user_id,
        encrypted_access_token=f"encrypted-{user_id}",
        client_id="client",
        client_auth_method="none",
        token_endpoint="https://93.184.216.34/token",
        server_url="https://93.184.216.34/mcp",
    )


async def test_several_servers_are_read_in_one_call_by_server(db: AsyncSession) -> None:
    first, second, alone = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db.add_all([_connection(first, "ada"), _connection(first, "grace"), _connection(second, "alan")])
    await db.commit()

    grouped = await McpUserConnectionRepository(db).every_for_servers([first, second, alone])

    assert {server_id: sorted(row.user_id for row in rows) for server_id, rows in grouped.items()} == {
        first: ["ada", "grace"],
        second: ["alan"],
    }, "a server nobody signed in to is absent rather than empty"
    assert await McpUserConnectionRepository(db).every_for_servers([]) == {}


async def test_the_people_signed_in_are_read_as_ids(db: AsyncSession) -> None:
    server_id, other = uuid.uuid4(), uuid.uuid4()
    db.add_all([_connection(server_id, "ada"), _connection(server_id, "grace"), _connection(other, "alan")])
    await db.commit()

    assert await McpUserConnectionRepository(db).signed_in_users(server_id) == {"ada", "grace"}
    assert await McpUserConnectionRepository(db).signed_in_users(uuid.uuid4()) == set()
