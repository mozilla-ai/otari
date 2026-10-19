"""The revision that names hosted providers, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half: an
existing provider keeps its implementation as its name, two names may share an
implementation, a keyless row stores no key, and the downgrade refuses what the
old table could not hold.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "c7e1a3f5b9d2"
_BEFORE = "557fc0f5bfda"

_INSERT_BEFORE = text(
    "INSERT INTO hosted_providers (id, provider, encrypted_api_key, enabled, created_at)"
    " VALUES (:id, :provider, 'ciphertext', 1, CURRENT_TIMESTAMP)"
)
_INSERT_NAMED = text(
    "INSERT INTO hosted_providers (id, name, provider, encrypted_api_key, enabled, created_at)"
    " VALUES (:id, :name, :provider, :key, 1, CURRENT_TIMESTAMP)"
)


@pytest.fixture
def chain(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'hosted_names.db'}"
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


def _names(engine: Engine) -> list[tuple[str, str]]:
    with engine.begin() as connection:
        return [
            tuple(row) for row in connection.execute(text("SELECT name, provider FROM hosted_providers ORDER BY name"))
        ]


def test_an_existing_provider_is_named_for_its_implementation(chain: tuple[Config, Engine]) -> None:
    config, engine = chain
    with engine.begin() as connection:
        connection.execute(_INSERT_BEFORE, {"id": "00000000000000000000000000000001", "provider": "openai"})

    command.upgrade(config, _REVISION)

    assert _names(engine) == [("openai", "openai")]


def test_names_are_unique_and_an_implementation_is_not(chain: tuple[Config, Engine]) -> None:
    config, engine = chain
    command.upgrade(config, _REVISION)
    with engine.begin() as connection:
        connection.execute(_INSERT_NAMED, {"id": "1", "name": "openai", "provider": "openai", "key": "c"})
        connection.execute(_INSERT_NAMED, {"id": "2", "name": "openai-eu", "provider": "openai", "key": None})

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(_INSERT_NAMED, {"id": "3", "name": "openai-eu", "provider": "openai", "key": "c"})

    assert _names(engine) == [("openai", "openai"), ("openai-eu", "openai")]


def test_the_downgrade_refuses_rows_the_old_table_cannot_hold(chain: tuple[Config, Engine]) -> None:
    config, engine = chain
    command.upgrade(config, _REVISION)
    with engine.begin() as connection:
        connection.execute(_INSERT_NAMED, {"id": "1", "name": "openai", "provider": "openai", "key": "c"})
        connection.execute(_INSERT_NAMED, {"id": "2", "name": "openai-eu", "provider": "openai", "key": "c"})

    with pytest.raises(RuntimeError, match="named otherwise than their implementation"):
        command.downgrade(config, _BEFORE)


def test_the_downgrade_refuses_a_lone_renamed_provider(chain: tuple[Config, Engine]) -> None:
    """Its roster and prices are keyed on the name, which the old code never looks up."""
    config, engine = chain
    command.upgrade(config, _REVISION)
    with engine.begin() as connection:
        connection.execute(_INSERT_NAMED, {"id": "1", "name": "openai-eu", "provider": "openai", "key": "c"})

    with pytest.raises(RuntimeError, match="named otherwise than their implementation"):
        command.downgrade(config, _BEFORE)


def test_the_downgrade_restores_one_row_per_implementation(chain: tuple[Config, Engine]) -> None:
    config, engine = chain
    command.upgrade(config, _REVISION)
    with engine.begin() as connection:
        connection.execute(_INSERT_NAMED, {"id": "1", "name": "openai", "provider": "openai", "key": "c"})

    command.downgrade(config, _BEFORE)

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(_INSERT_BEFORE, {"id": "2", "provider": "openai"})
