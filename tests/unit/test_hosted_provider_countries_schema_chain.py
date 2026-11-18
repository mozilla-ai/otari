"""The ``hosted_providers.supported_countries`` revision's Alembic chain, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half: a
provider stored before the column existed serves every country, and the
downgrade drops the column without taking the row with it.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "db1bdb2e9a93"
_BEFORE = "c3e5a7b9d1f2"


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'hosted_provider_countries.db'}"
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
    return {column["name"] for column in inspect(engine).get_columns("hosted_providers")}


def test_an_existing_provider_serves_every_country_and_survives_the_downgrade(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO hosted_providers (id, name, provider, enabled, created_at)"
                " VALUES (:id, 'openai', 'openai', 1, CURRENT_TIMESTAMP)"
            ),
            {"id": uuid.uuid4().hex},
        )

    command.upgrade(config, _REVISION)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT supported_countries FROM hosted_providers")).scalar_one() is None

    command.downgrade(config, _BEFORE)
    assert "supported_countries" not in _columns(engine)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT name FROM hosted_providers")).scalar_one() == "openai"

    command.upgrade(config, _REVISION)
    assert "supported_countries" in _columns(engine)
