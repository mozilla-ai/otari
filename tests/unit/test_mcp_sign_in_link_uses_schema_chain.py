"""The ``mcp_sign_in_link_uses`` revision's Alembic chain, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half: the table and its
expiry index come and go with the revision, and the chain upgrades again after the downgrade.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "e5b9d2f7a4c1"
_BEFORE = "a9d3f5b7c1e2"
_TABLE = "mcp_sign_in_link_uses"
_INDEX = "ix_mcp_sign_in_link_uses_expires_at"


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'sign_in_link_uses.db'}"
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, _BEFORE)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def test_the_table_comes_and_goes_with_its_revision(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    assert _TABLE not in inspect(engine).get_table_names()

    command.upgrade(config, _REVISION)
    assert {column["name"] for column in inspect(engine).get_columns(_TABLE)} == {
        "ticket_hash",
        "expires_at",
        "created_at",
    }
    assert inspect(engine).get_pk_constraint(_TABLE)["constrained_columns"] == ["ticket_hash"]
    assert _INDEX in {index["name"] for index in inspect(engine).get_indexes(_TABLE)}
    with engine.begin() as connection:
        insert = text(
            f"INSERT INTO {_TABLE} (ticket_hash, expires_at, created_at)"  # noqa: S608 - constant table name
            " VALUES (:hash, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
        connection.execute(insert, {"hash": "a" * 64})

    command.downgrade(config, _BEFORE)
    assert _TABLE not in inspect(engine).get_table_names()

    command.upgrade(config, _REVISION)
    assert _TABLE in inspect(engine).get_table_names()
