"""Unit tests for accepting an async database URL.

README tells SQLite users to set ``sqlite+aiosqlite:///./otari.db``. The app
engine has always accepted that, but migrations run on a *sync* engine, so with
``auto_migrate`` on (the documented default path) startup died in
``_run_migrations`` with MissingGreenlet before serving anything. Both URL forms
must reach a working schema.
"""

from pathlib import Path

import pytest
from sqlalchemy import text

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session, init_db, reset_db, to_sync_url


@pytest.mark.parametrize(
    ("database_url", "expected"),
    [
        ("sqlite+aiosqlite:///./otari.db", "sqlite:///./otari.db"),
        ("postgresql+asyncpg://u:p@host:5432/db", "postgresql://u:p@host:5432/db"),
        # Already sync, or a driver with no async counterpart: left alone.
        ("sqlite:///./otari.db", "sqlite:///./otari.db"),
        ("postgresql://u:p@host:5432/db", "postgresql://u:p@host:5432/db"),
        ("postgresql+psycopg2://u:p@host:5432/db", "postgresql+psycopg2://u:p@host:5432/db"),
    ],
)
def test_to_sync_url(database_url: str, expected: str) -> None:
    assert to_sync_url(database_url) == expected


def test_to_sync_url_keeps_password_and_query() -> None:
    # render_as_string masks the password by default, which would turn a real
    # URL into one that cannot connect.
    url = to_sync_url("postgresql+asyncpg://user:s3cret@host:5432/db?ssl=require")
    assert "s3cret" in url
    assert "ssl=require" in url


@pytest.mark.asyncio
async def test_auto_migrate_accepts_async_sqlite_url(tmp_path: Path) -> None:
    """The URL README documents must survive startup with auto_migrate on."""
    db_path = tmp_path / "async-url.db"
    config = GatewayConfig(database_url=f"sqlite+aiosqlite:///{db_path}", auto_migrate=True)

    init_db(config)
    try:
        # Migrations ran against the same database the app engine is pointed at,
        # so a table the schema defines is queryable rather than missing.
        async with create_session() as db:
            assert (await db.execute(text("SELECT COUNT(*) FROM users"))).scalar_one() == 0
    finally:
        reset_db()
    assert db_path.exists()
