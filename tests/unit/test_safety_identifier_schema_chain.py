"""The ``usage_logs.safety_identifier`` revision's Alembic chain, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half: a
row written before the column existed reads as having no identifier, the column
is indexed, and the downgrade drops both without taking the row with it.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "7e02300efc98"
_BEFORE = "e8b2d4f6a0c3"
_INDEX = "ix_usage_logs_safety_identifier"


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'safety_identifier.db'}"
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


def _columns(engine: Engine) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns("usage_logs")}


def _indexes(engine: Engine) -> set[str | None]:
    return {index["name"] for index in inspect(engine).get_indexes("usage_logs")}


def test_an_existing_row_has_no_identifier_and_survives_the_downgrade(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO usage_logs (id, workspace_id, timestamp, model, endpoint, source, counts_toward_budget,"
                " status) VALUES ('row-1', :workspace, CURRENT_TIMESTAMP, 'gpt-4o', '/v1/chat/completions',"
                " 'gateway', 1, 'success')"
            ),
            {"workspace": uuid.uuid4().hex},
        )

    command.upgrade(config, _REVISION)
    assert _INDEX in _indexes(engine)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT safety_identifier FROM usage_logs")).scalar_one() is None

    command.downgrade(config, _BEFORE)
    assert "safety_identifier" not in _columns(engine)
    assert _INDEX not in _indexes(engine)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT id FROM usage_logs")).scalar_one() == "row-1"

    command.upgrade(config, _REVISION)
    assert "safety_identifier" in _columns(engine)
