"""The ``country_database`` revision's Alembic chain, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half: the
table holds the file's raw bytes, holds one row at most, and goes with the
downgrade.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "e3a1c5d7f9b2"
_BEFORE = "db1bdb2e9a93"
_INSERT = text(
    "INSERT INTO country_database (id, content, database_type, built_at, size_bytes, sha256, uploaded_at)"
    " VALUES (:id, :content, 'Test-Country', CURRENT_TIMESTAMP, 3, 'ab', CURRENT_TIMESTAMP)"
)


@pytest.fixture
def sqlite_at_revision(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'country_database.db'}"
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, _REVISION)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def test_the_table_holds_the_raw_bytes_of_one_file(sqlite_at_revision: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_revision
    with engine.begin() as connection:
        connection.execute(_INSERT, {"id": 1, "content": b"\x00\xab\xff"})
    with engine.connect() as connection:
        assert connection.execute(text("SELECT content FROM country_database")).scalar_one() == b"\x00\xab\xff"

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(_INSERT, {"id": 2, "content": b"\x00"})


def test_the_downgrade_drops_the_table(sqlite_at_revision: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_revision

    command.downgrade(config, _BEFORE)
    assert "country_database" not in inspect(engine).get_table_names()

    command.upgrade(config, _REVISION)
    assert "country_database" in inspect(engine).get_table_names()
