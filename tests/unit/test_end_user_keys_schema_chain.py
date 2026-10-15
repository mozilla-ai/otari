"""The end-user keys revision's Alembic chain, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half: the
three columns come and go without rebuilding ``api_keys``, whose partial index a
rebuild would lose, and a downgrade takes the minted keys with the column that
marks them.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "c9e1a3f5b7d2"
_BEFORE = "b7d9f1a3c5e8"
_COLUMNS = {"parent_key_id", "max_child_ttl_sec", "tags"}


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'end_user_keys.db'}"
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
    return {column["name"] for column in inspect(engine).get_columns("api_keys")}


def _indexes(engine: Engine) -> set[str]:
    return {str(index["name"]) for index in inspect(engine).get_indexes("api_keys")}


def _insert_key(engine: Engine, key_id: str, parent_key_id: str | None = None) -> None:
    columns = "id, key_hash, workspace_id, created_at, is_active, exclude_from_budget, is_service_key, metadata"
    values = ":id, :hash, :workspace, CURRENT_TIMESTAMP, 1, 0, 0, '{}'"
    if parent_key_id is not None:
        columns += ", parent_key_id"
        values += ", :parent"
    with engine.begin() as connection:
        connection.execute(
            text(f"INSERT INTO api_keys ({columns}) VALUES ({values})"),  # noqa: S608
            {"id": key_id, "hash": f"hash-{key_id}", "workspace": uuid.uuid4().hex, "parent": parent_key_id},
        )


def test_the_columns_come_and_go_and_the_minted_keys_with_them(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    _insert_key(engine, "existing")

    command.upgrade(config, _REVISION)
    assert _COLUMNS <= _columns(engine)
    assert {"ix_api_keys_parent_key_id", "ix_api_keys_internal_dispatch"} <= _indexes(engine)
    _insert_key(engine, "minted", parent_key_id="existing")
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT max_child_ttl_sec FROM api_keys WHERE id = 'existing'")).scalar_one()
            is None
        )

    command.downgrade(config, _BEFORE)
    assert not _COLUMNS & _columns(engine)
    assert "ix_api_keys_parent_key_id" not in _indexes(engine)
    assert "ix_api_keys_internal_dispatch" in _indexes(engine)
    with engine.connect() as connection:
        index_sql = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE name = 'ix_api_keys_internal_dispatch'")
        ).scalar_one()
    assert "WHERE" in index_sql.upper()
    with engine.connect() as connection:
        assert connection.execute(text("SELECT id FROM api_keys")).scalars().all() == ["existing"]

    command.upgrade(config, _REVISION)
    assert _COLUMNS <= _columns(engine)
