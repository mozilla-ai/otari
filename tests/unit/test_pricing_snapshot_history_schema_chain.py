"""The snapshot-history revision's Alembic chain, exercised on SQLite."""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.orm import Session
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.models.entities import PricingSnapshotHistory

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "b2d4f6a8c0e2"
_BEFORE = "c7e9a1b3d5f7"
_TABLE = "pricing_snapshot_history"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_at_head(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'history.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def test_the_migrated_table_matches_the_model(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    columns = {column["name"] for column in inspect(engine).get_columns(_TABLE)}
    assert columns == set(SQLModel.metadata.tables[_TABLE].columns.keys())


def test_an_accepted_snapshot_is_kept_and_the_revision_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head
    with Session(engine) as session:
        session.add(
            PricingSnapshotHistory(
                id=uuid.uuid4(),
                source="genai-prices",
                accepted_at=datetime(2026, 9, 6, tzinfo=UTC),
                accepted_by="schedule",
                model_count=1450,
                snapshot="{}",
            )
        )
        session.commit()
        assert session.query(PricingSnapshotHistory).count() == 1

    command.downgrade(config, _BEFORE)
    assert _TABLE not in inspect(engine).get_table_names()

    command.upgrade(config, _REVISION)
    assert _TABLE in inspect(engine).get_table_names()
