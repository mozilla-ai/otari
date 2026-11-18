"""Unit tests for SQLite connection pragmas configured in ``init_db``.

Regression coverage for issue #106: without WAL journal mode and a
``busy_timeout``, concurrent SQLite access (e.g. a key-creation write
colliding with an auth-lookup read) raised "database is locked" and surfaced
as intermittent HTTP 500s. ``init_db`` must apply these pragmas to every
SQLite connection.
"""

from pathlib import Path

import pytest
from sqlalchemy import text

from gateway.core.config import GatewayConfig
from gateway.core.database import (
    _SQLITE_BUSY_TIMEOUT_MS,
    create_session,
    init_db,
    reset_db,
)


@pytest.mark.asyncio
async def test_sqlite_connection_pragmas_applied(tmp_path: Path) -> None:
    db_path = tmp_path / "pragmas.db"
    config = GatewayConfig(database_url=f"sqlite:///{db_path}", auto_migrate=False)

    init_db(config)
    try:
        async with create_session() as db:
            journal_mode = (await db.execute(text("PRAGMA journal_mode"))).scalar_one()
            busy_timeout = (await db.execute(text("PRAGMA busy_timeout"))).scalar_one()
            foreign_keys = (await db.execute(text("PRAGMA foreign_keys"))).scalar_one()

        assert str(journal_mode).lower() == "wal"
        assert int(busy_timeout) == _SQLITE_BUSY_TIMEOUT_MS
        assert int(foreign_keys) == 1
    finally:
        reset_db()


@pytest.mark.asyncio
@pytest.mark.parametrize(("end", "kept"), [("rollback", 0), ("commit", 1)])
async def test_a_savepoint_opened_before_any_write_ends_with_its_transaction(
    tmp_path: Path, end: str, kept: int
) -> None:
    """#2535: such a savepoint was the outermost transaction, so releasing it committed and a rollback undid nothing."""
    init_db(GatewayConfig(database_url=f"sqlite:///{tmp_path / 'savepoint.db'}", auto_migrate=False))
    try:
        async with create_session() as db:
            await db.execute(text("CREATE TABLE things (name TEXT)"))
            await db.commit()
        async with create_session() as db:
            async with db.begin_nested():
                await db.execute(text("INSERT INTO things VALUES ('written in the savepoint')"))
            await getattr(db, end)()
        async with create_session() as db:
            assert (await db.execute(text("SELECT COUNT(*) FROM things"))).scalar_one() == kept
    finally:
        reset_db()
