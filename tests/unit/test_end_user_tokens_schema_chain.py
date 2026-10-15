"""The signed end-user tokens revision's Alembic chain, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half: both
tables come and go, ``api_keys`` keeps its partial index, and a downgrade takes
the keys tokens were materialized as with the table that says what they are.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "d4f8b2c6e1a9"
_BEFORE = "c9e1a3f5b7d2"
_TABLES = {"api_key_signing_keys", "api_key_token_redemptions"}


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'end_user_tokens.db'}"
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


def test_the_tables_come_and_go_and_the_materialized_keys_with_them(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    _insert_key(engine, "service")
    _insert_key(engine, "minted", parent_key_id="service")

    command.upgrade(config, _REVISION)
    assert _TABLES <= set(inspect(engine).get_table_names())
    _insert_key(engine, "materialized", parent_key_id="service")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO api_key_signing_keys (id, api_key_id, public_key, created_at) "
                "VALUES ('kid-1', 'service', 'pem', CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO api_key_token_redemptions "
                "(parent_key_id, jti, child_key_id, signing_key_id, claims_digest, expires_at, created_at) "
                "VALUES ('service', 'run-1', 'materialized', 'kid-1', 'digest', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO api_key_token_redemptions "
                "(parent_key_id, jti, child_key_id, signing_key_id, claims_digest, expires_at, created_at) "
                "VALUES ('service', 'run-2', NULL, 'kid-1', 'digest', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )

    command.downgrade(config, _BEFORE)
    assert not _TABLES & set(inspect(engine).get_table_names())
    assert "ix_api_keys_internal_dispatch" in {str(index["name"]) for index in inspect(engine).get_indexes("api_keys")}
    with engine.connect() as connection:
        assert sorted(connection.execute(text("SELECT id FROM api_keys")).scalars()) == ["minted", "service"]

    command.upgrade(config, _REVISION)
    assert _TABLES <= set(inspect(engine).get_table_names())
