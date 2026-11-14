"""The revision that scopes API keys keeps every existing key reaching what it reached, on SQLite.

Each key is seeded at the revision before and read back after the upgrade: a
service key gets every scope, any other key every scope but the end-user ones,
and a key inserted afterwards without naming any gets none.
"""

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

from gateway.types.key_scope import KeyScopeSet

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "c3f8a1d6e2b9"
_BEFORE = "d8a491bc02fa"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before this one."""
    url = f"sqlite:///{tmp_path / 'scopes.db'}"
    config = _alembic_config(url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _key(engine: Engine, *, is_service_key: bool) -> str:
    key_id = str(uuid.uuid4())
    with engine.begin() as connection:
        workspace_id = connection.execute(text("SELECT id FROM workspace LIMIT 1")).scalar_one()
        connection.execute(
            text(
                "INSERT INTO api_keys (id, key_hash, workspace_id, created_at, is_active, is_service_key, metadata) "
                "VALUES (:id, :hash, :workspace_id, '2026-11-01 00:00:00', 1, :service, '{}')"
            ),
            {"id": key_id, "hash": key_id, "workspace_id": workspace_id, "service": is_service_key},
        )
    return key_id


def _scopes(engine: Engine, key_id: str) -> list[str]:
    with engine.connect() as connection:
        stored = connection.execute(text("SELECT scopes FROM api_keys WHERE id = :id"), {"id": key_id}).scalar_one()
    scopes: list[str] = json.loads(stored)
    return scopes


def test_each_existing_key_keeps_what_it_could_reach(before: tuple[Config, Engine]) -> None:
    config, engine = before
    plain = _key(engine, is_service_key=False)
    service = _key(engine, is_service_key=True)

    command.upgrade(config, _REVISION)

    assert _scopes(engine, plain) == KeyScopeSet.unscoped(is_service_key=False).stored
    assert _scopes(engine, service) == KeyScopeSet.unscoped(is_service_key=True).stored


def test_a_key_inserted_without_scopes_reaches_nothing(before: tuple[Config, Engine]) -> None:
    config, engine = before
    command.upgrade(config, _REVISION)

    assert _scopes(engine, _key(engine, is_service_key=True)) == []


def test_the_revision_downgrades_and_upgrades_again(before: tuple[Config, Engine]) -> None:
    config, engine = before
    plain = _key(engine, is_service_key=False)
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)
    command.upgrade(config, _REVISION)

    assert _scopes(engine, plain) == KeyScopeSet.unscoped(is_service_key=False).stored
