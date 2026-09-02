"""The usage-log writer draws from the metering pool, not the request pool.

A saturated request pool used to time the writer out as well, and a dropped
usage row cannot be recovered: spend, budget reconciliation and the activity
log are all rebuilt from it afterwards. Metering must not be the first thing a
busy gateway loses.
"""

from pathlib import Path

import pytest
from sqlalchemy import text

from gateway.core.config import GatewayConfig
from gateway.core.database import init_db, reset_db
from gateway.services import log_writer as log_writer_module
from gateway.services.log_writer import BatchLogWriter, SingleLogWriter


def test_writers_use_the_metering_session_factory() -> None:
    source = Path(log_writer_module.__file__).read_text()
    # Both writers, and no path left on the request pool.
    assert source.count("create_log_session()") == 2
    assert "create_session()" not in source


@pytest.mark.asyncio
async def test_metering_pool_is_a_separate_engine(tmp_path: Path) -> None:
    from gateway.core import database

    reset_db()
    init_db(GatewayConfig(database_url=f"sqlite+aiosqlite:///{tmp_path/'otari.db'}", auto_migrate=False))
    try:
        assert database._log_engine is not None
        assert database._log_engine is not database._engine
        # Same database, so a row the writer commits is one the request path reads.
        assert database._log_engine.url == database._engine.url  # type: ignore[union-attr]
        async with database.create_log_session() as db:
            assert (await db.execute(text("SELECT 1"))).scalar_one() == 1
    finally:
        reset_db()


@pytest.mark.asyncio
async def test_writers_construct_without_init_db() -> None:
    # The fallback in create_log_session: tests build these directly.
    assert isinstance(SingleLogWriter(), SingleLogWriter)
    assert isinstance(BatchLogWriter(), BatchLogWriter)
