"""A stored server's URL is matched with its surrounding whitespace aside, on SQLite as on PostgreSQL."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401 - registers every table on the metadata
from gateway.models.tools import WorkspaceMcpServer
from gateway.repositories.tools import WorkspaceMcpServerRepository

URL = "https://93.184.216.34/mcp"


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


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [f"{URL}\n", f"\t{URL}/", f" {URL}/\r\n", URL])
async def test_a_stored_url_matches_whatever_whitespace_surrounds_it(db: AsyncSession, stored: str) -> None:
    workspace_id = uuid.uuid4()
    db.add(WorkspaceMcpServer(workspace_id=workspace_id, name="drive", url=stored))
    await db.commit()

    servers = WorkspaceMcpServerRepository(db)

    assert await servers.has_url(workspace_id, URL)
    assert await servers.has_url(workspace_id, f"{URL}/ ")
    assert not await servers.has_url(workspace_id, "https://93.184.216.35/mcp")
