"""Tests on SQLite that a session losing the race to create an MCP catalog row reads the winner's."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.models.mcp_catalog import LOGO_OK, SYNC_STATE_ROW_ID, McpCatalogSyncState, McpLogo
from gateway.repositories.mcp_catalog import McpCatalogSyncStateRepository, McpLogoRepository

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 11, 10, 9, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def test_the_sync_state_row_a_loser_reads_is_the_winners(
    sessions: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    async with sessions() as winner:
        won = await McpCatalogSyncStateRepository(winner).get_or_create()
        won.cursor = "winner"
        await winner.commit()

    async with sessions() as loser:
        get = loser.get
        lookups = 0

        async def missed_once(*args: Any, **kwargs: Any) -> Any:
            # The loser looked before the winner committed, so its first lookup found nothing.
            nonlocal lookups
            lookups += 1
            return None if lookups == 1 else await get(*args, **kwargs)

        monkeypatch.setattr(loser, "get", missed_once)
        read = await McpCatalogSyncStateRepository(loser).get_or_create()
        await loser.commit()

    assert (read.id, read.cursor) == (SYNC_STATE_ROW_ID, "winner")
    async with sessions() as check:
        assert len((await check.execute(select(McpCatalogSyncState))).scalars().all()) == 1


async def test_a_logo_a_loser_stores_updates_the_winners_row(
    sessions: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    async with sessions() as winner:
        await McpLogoRepository(winner).store_not_found("host:mcp.example.com", fetched_at=_NOW, retry_at=_NOW)
        await winner.commit()

    async with sessions() as loser:
        repository = McpLogoRepository(loser)
        find = repository.find
        lookups = 0

        async def missed_once(key: str) -> McpLogo | None:
            nonlocal lookups
            lookups += 1
            return None if lookups == 1 else await find(key)

        monkeypatch.setattr(repository, "find", missed_once)
        await repository.store_found(
            "host:mcp.example.com",
            content=b"<svg/>",
            content_type="image/svg+xml",
            etag="e1",
            luminance=None,
            source_url="https://mcp.example.com/icon.svg",
            fetched_at=_NOW,
            retry_at=_NOW,
        )
        await loser.commit()

    async with sessions() as check:
        rows = (await check.execute(select(McpLogo))).scalars().all()
    assert [(row.key, row.status, row.content) for row in rows] == [("host:mcp.example.com", LOGO_OK, b"<svg/>")]
