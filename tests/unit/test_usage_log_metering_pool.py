"""The usage-log writer draws from the metering pool, not the request pool.

A saturated request pool used to time the writer out as well, and a dropped
usage row cannot be recovered: spend, budget reconciliation and the activity
log are all rebuilt from it. Metering must not be the first thing a busy
gateway loses.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from gateway.core.config import GatewayConfig
from gateway.core.database import init_db, reset_db
from gateway.models.entities import UsageLog
from gateway.services import log_writer as log_writer_module
from gateway.services.log_writer import BatchLogWriter, SingleLogWriter


class _RecordingSession:
    """Enough of AsyncSession for a writer to think it wrote something."""

    def __init__(self) -> None:
        self.added: list[Any] = []
        self.committed = False

    def add(self, row: Any) -> None:
        self.added.append(row)

    def add_all(self, rows: Any) -> None:
        self.added.extend(rows)

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:  # pragma: no cover - not reached here
        pass


@pytest.fixture
def metering_session(monkeypatch: pytest.MonkeyPatch) -> _RecordingSession:
    """Stand in for the metering pool and record what the writer sends it."""
    session = _RecordingSession()

    @asynccontextmanager
    async def _fake() -> AsyncIterator[_RecordingSession]:
        yield session

    monkeypatch.setattr(log_writer_module, "create_log_session", _fake)
    return session


@pytest.mark.asyncio
async def test_single_writer_writes_through_the_metering_pool(
    metering_session: _RecordingSession,
) -> None:
    row = UsageLog(endpoint="/v1/chat/completions", model="m", provider="p")
    await SingleLogWriter().put(row)
    assert metering_session.added == [row]
    assert metering_session.committed


@pytest.mark.asyncio
async def test_batch_writer_flushes_through_the_metering_pool(
    metering_session: _RecordingSession,
) -> None:
    rows = [UsageLog(endpoint="/v1/embeddings", model="m", provider="p") for _ in range(3)]
    await BatchLogWriter()._flush(rows)
    assert metering_session.added == rows
    assert metering_session.committed


@pytest.mark.asyncio
async def test_metering_pool_is_a_separate_engine_on_postgres() -> None:
    from gateway.core import database

    reset_db()
    # Not connected to: init_db only builds the engines, and auto_migrate off
    # keeps it from reaching for the server.
    init_db(
        GatewayConfig(
            database_url="postgresql+asyncpg://u:p@localhost:5432/db",
            auto_migrate=False,
        )
    )
    try:
        assert database._log_engine is not None
        assert database._log_engine is not database._engine
        assert database._log_engine.url == database._engine.url  # type: ignore[union-attr]
        # No overflow above the reserved size, so metering never bursts against
        # the request pool it was separated from.
        assert database._log_engine.pool._max_overflow == 0  # type: ignore[attr-defined]
    finally:
        reset_db()


@pytest.mark.asyncio
async def test_sqlite_keeps_one_engine(tmp_path: Path) -> None:
    from gateway.core import database

    reset_db()
    init_db(
        GatewayConfig(
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'otari.db'}",
            auto_migrate=False,
        )
    )
    try:
        # A second engine would gain nothing on SQLite (NullPool ignores the
        # pool size) and would contend for the single writer without the
        # busy-timeout and foreign-key pragmas, which are applied to the
        # request engine only.
        assert database._log_engine is None
        assert database._LogSessionLocal is database._SessionLocal
        async with database.create_log_session() as db:
            assert (await db.execute(text("SELECT 1"))).scalar_one() == 1
    finally:
        reset_db()


@pytest.mark.asyncio
async def test_create_log_session_falls_back_to_the_request_pool(tmp_path: Path) -> None:
    from gateway.core import database

    reset_db()
    init_db(
        GatewayConfig(
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'otari.db'}",
            auto_migrate=False,
        )
    )
    try:
        # What a test that builds a writer directly hits: a process with a
        # request pool and no metering pool still logs rather than raising.
        database._LogSessionLocal = None
        async with database.create_log_session() as db:
            assert (await db.execute(text("SELECT 1"))).scalar_one() == 1
    finally:
        reset_db()


@pytest.mark.asyncio
async def test_create_log_session_without_any_pool_is_an_error() -> None:
    from gateway.core import database

    reset_db()
    with pytest.raises(RuntimeError, match="Database not initialized"):
        async with database.create_log_session():
            pass
