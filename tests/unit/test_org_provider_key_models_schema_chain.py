"""The offered-models revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default and nothing else in the suite migrates it,
so this is the only coverage of that path for this revision. Three things are
worth pinning beyond "the table appears". The revision is hand-written, so
nothing else would notice it and the model drifting apart. The foreign key is
composite, which is the whole mechanism stopping one organization's row from
naming another organization's key, and it is the part of a hand-written
revision easiest to reduce to a plain key on ``org_provider_key_id`` without
anything failing. And ``enabled`` carries a server default, because the service
writes rows in bulk and a row that arrived NULL would be neither served nor
withheld.
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
_OFFERED_MODELS_REVISION = "b4d7f1a9c2e6"
_BEFORE_OFFERED_MODELS = "f2a6c81d9b47"

_TABLE = "org_provider_key_models"
_INDEX = "ix_org_provider_key_models_org_provider_key_id"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_at_head(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'offered_models.db'}"
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


def test_the_key_reference_is_composite_and_cascades(sqlite_at_head: tuple[Config, Engine]) -> None:
    """A row is pinned to its own organization, and dies with the key.

    Composite rather than a plain key on ``org_provider_key_id``: a row offering
    a model is a statement about one organization's key, and a plain reference
    would let a row name a key belonging to somebody else without the database
    minding. CASCADE because an offered model has no meaning once the credential
    it was offered on is gone.
    """
    _, engine = sqlite_at_head

    keys = inspect(engine).get_foreign_keys(_TABLE)
    composite = [key for key in keys if len(key["constrained_columns"]) == 2]

    assert len(composite) == 1
    assert composite[0]["constrained_columns"] == ["organization_id", "org_provider_key_id"]
    assert composite[0]["referred_table"] == "org_provider_keys"
    assert composite[0]["referred_columns"] == ["organization_id", "id"]
    assert composite[0]["options"].get("ondelete") == "CASCADE"


def test_one_model_is_offered_once_per_key(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head

    constraints = {
        constraint["name"]: list(constraint["column_names"])
        for constraint in inspect(engine).get_unique_constraints(_TABLE)
    }

    assert constraints["uq_org_provider_key_models_key_model"] == ["org_provider_key_id", "model"]


def test_enabled_carries_a_server_default(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head

    enabled = next(column for column in inspect(engine).get_columns(_TABLE) if column["name"] == "enabled")

    assert enabled["nullable"] is False
    assert enabled["default"] is not None


def test_the_key_lookup_is_indexed(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Every read of this table leads with the key, so the index is not decoration."""
    _, engine = sqlite_at_head

    indexes = {index["name"]: list(index["column_names"]) for index in inspect(engine).get_indexes(_TABLE)}

    assert indexes[_INDEX] == ["org_provider_key_id"]


def test_the_revision_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head

    command.downgrade(config, _BEFORE_OFFERED_MODELS)
    assert _TABLE not in inspect(engine).get_table_names()

    command.upgrade(config, _OFFERED_MODELS_REVISION)
    assert _TABLE in inspect(engine).get_table_names()
