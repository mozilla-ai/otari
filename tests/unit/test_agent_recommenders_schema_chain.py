"""The ``agent_recommenders`` revision's Alembic chain, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half:
the partial unique index lets any number of routers be stored and one be
switched on, and the downgrade takes the table away cleanly.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "7c4e2a9b5d18"
_BEFORE = "d9f1a3c5e7b2"

_INSERT = text(
    "INSERT INTO agent_recommenders (id, provider, model, enabled) VALUES (:id, 'typesafe', :model, :enabled)"
)


@pytest.fixture
def sqlite_at_revision(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'agent_recommenders.db'}"
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


def _tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def test_many_routers_fit_switched_off_and_one_switched_on(sqlite_at_revision: tuple[Config, Engine]) -> None:
    """The index is partial on both dialects: a second switched-off row is not a duplicate."""
    _, engine = sqlite_at_revision
    with engine.begin() as connection:
        for model, enabled in (("a", 0), ("b", 0), ("c", 1)):
            connection.execute(_INSERT, {"id": uuid.uuid4().hex, "model": model, "enabled": enabled})

    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM agent_recommenders")).scalar_one() == 3


def test_a_second_router_switched_on_is_refused(sqlite_at_revision: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_revision
    with engine.begin() as connection:
        connection.execute(_INSERT, {"id": uuid.uuid4().hex, "model": "a", "enabled": 1})

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(_INSERT, {"id": uuid.uuid4().hex, "model": "b", "enabled": 1})


def test_a_router_starts_switched_off_with_free_rates(sqlite_at_revision: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_revision
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO agent_recommenders (id, provider, model) VALUES (:id, 'typesafe', 'a')"),
            {"id": uuid.uuid4().hex},
        )

    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT enabled, input_price_per_million, output_price_per_million FROM agent_recommenders")
        ).one()
    assert tuple(row) == (0, 0, 0)


def test_the_downgrade_drops_the_table_and_the_upgrade_brings_it_back(
    sqlite_at_revision: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_at_revision

    command.downgrade(config, _BEFORE)
    assert "agent_recommenders" not in _tables(engine)

    command.upgrade(config, _REVISION)
    assert "agent_recommenders" in _tables(engine)
