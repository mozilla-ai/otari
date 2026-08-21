"""The setup-guide revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default, and nothing else in the suite migrates it,
so this is the only coverage of that path for this revision. Two things are
worth pinning beyond "the table appears": the revision is hand-written, so
nothing else would notice it and the model drifting apart, and the index it adds
to ``usage_logs`` is not decoration. The guide asks a question of that table on
every dashboard load, and the index is what keeps it a seek rather than a scan of
the workspace's traffic, so an index silently absent is a performance regression
with no failing test anywhere else.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_ACTIVATION_REVISION = "c8e2a4f6b0d3"
_BEFORE_ACTIVATION = "a7c3e5d9b1f4"

_TABLE = "workspace_activation_state"
_USAGE_INDEX = "ix_usage_logs_workspace_status_timestamp"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_at_head(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'activation.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def test_the_migrated_table_matches_the_model(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head

    declared = SQLModel.metadata.tables[_TABLE]
    migrated = {column["name"] for column in inspect(engine).get_columns(_TABLE)}

    assert migrated == set(declared.columns.keys())


def test_the_foreign_keys_carry_the_delete_policy_each_column_needs(
    sqlite_at_head: tuple[Config, Engine],
) -> None:
    """A workspace takes its row with it; a deleted key must not.

    Deleting the guide's key from the Keys page is a legitimate thing to do, and
    a cascade there would take the dismissal on this row with it, so the guide
    would return to a workspace that had turned it down.
    """
    _, engine = sqlite_at_head

    policies = {
        tuple(key["constrained_columns"]): key["options"].get("ondelete")
        for key in inspect(engine).get_foreign_keys(_TABLE)
    }

    assert policies[("workspace_id",)] == "CASCADE"
    assert policies[("api_key_id",)] == "SET NULL"


def test_the_usage_index_the_guide_reads_through_exists(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head

    indexes = {index["name"]: list(index["column_names"]) for index in inspect(engine).get_indexes("usage_logs")}

    assert indexes[_USAGE_INDEX] == ["workspace_id", "status", "timestamp"]


def test_the_revision_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head

    command.downgrade(config, _BEFORE_ACTIVATION)
    inspector = inspect(engine)
    assert _TABLE not in inspector.get_table_names()
    assert _USAGE_INDEX not in {index["name"] for index in inspector.get_indexes("usage_logs")}

    command.upgrade(config, _ACTIVATION_REVISION)
    inspector = inspect(engine)
    assert _TABLE in inspector.get_table_names()
    assert _USAGE_INDEX in {index["name"] for index in inspector.get_indexes("usage_logs")}
