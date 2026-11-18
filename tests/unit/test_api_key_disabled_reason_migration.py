"""The revision that replaces the API key operator switch-off flag with a reason, on SQLite.

Each case seeds keys at the revision before, then upgrades or round trips: a
flagged key becomes ``operator``, an unflagged one stays NULL, and a downgrade
maps either reason back to the flag.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "c4b8e2a19d73"
_BEFORE = "a7f2c91d4e60"
_CREATED = "2026-01-01 00:00:00.000000"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before this one."""
    url = f"sqlite:///{tmp_path / 'keys.db'}"
    config = _alembic_config(url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _workspace_id(engine: Engine) -> str:
    """The id of the workspace the migration chain seeds, which every key needs."""
    with engine.connect() as connection:
        workspace_id = connection.execute(text("SELECT id FROM workspace LIMIT 1")).scalar_one()
    return str(workspace_id)


def _key(engine: Engine, *, is_active: bool, operator_disabled: bool) -> str:
    key_id = uuid.uuid4().hex
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO api_keys (id, key_hash, workspace_id, is_active, operator_disabled, "
                "exclude_from_budget, is_service_key, created_at, metadata) "
                "VALUES (:id, :hash, :workspace, :is_active, :operator_disabled, 0, 0, :created, '{}')"
            ),
            {
                "id": key_id,
                "hash": f"hash-{key_id}",
                "workspace": _workspace_id(engine),
                "is_active": is_active,
                "operator_disabled": operator_disabled,
                "created": _CREATED,
            },
        )
    return key_id


def _reasons(engine: Engine) -> dict[str, str | None]:
    with engine.connect() as connection:
        return dict(connection.execute(text("SELECT id, disabled_reason FROM api_keys")).all())


def _flags(engine: Engine) -> dict[str, bool]:
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT id, operator_disabled FROM api_keys")).all()
    return {key_id: bool(flag) for key_id, flag in rows}


def test_a_flagged_key_carries_over_as_operator(before: tuple[Config, Engine]) -> None:
    config, engine = before
    locked = _key(engine, is_active=False, operator_disabled=True)
    owner_switched_off = _key(engine, is_active=False, operator_disabled=False)
    live = _key(engine, is_active=True, operator_disabled=False)

    command.upgrade(config, _REVISION)

    reasons = _reasons(engine)
    assert reasons[locked] == "operator"
    assert reasons[owner_switched_off] is None
    assert reasons[live] is None


def test_the_flag_is_gone_after_the_upgrade(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _key(engine, is_active=False, operator_disabled=True)

    command.upgrade(config, _REVISION)

    with engine.connect() as connection:
        columns = {row[1] for row in connection.execute(text("PRAGMA table_info(api_keys)")).all()}
    assert "operator_disabled" not in columns
    assert "disabled_reason" in columns


def test_the_downgrade_maps_either_reason_back_to_the_flag(before: tuple[Config, Engine]) -> None:
    """Both reasons are locks the owner may not clear, so both come back as true."""
    config, engine = before
    operator_locked = _key(engine, is_active=False, operator_disabled=True)
    revoked = _key(engine, is_active=False, operator_disabled=False)
    owner_switched_off = _key(engine, is_active=False, operator_disabled=False)
    command.upgrade(config, _REVISION)
    with engine.begin() as connection:
        connection.execute(text("UPDATE api_keys SET disabled_reason = 'revoked' WHERE id = :id"), {"id": revoked})

    command.downgrade(config, _BEFORE)

    flags = _flags(engine)
    assert flags[operator_locked] is True
    assert flags[revoked] is True
    assert flags[owner_switched_off] is False


def test_the_unique_index_survives_the_round_trip(before: tuple[Config, Engine]) -> None:
    """SQLite rebuilds the table to drop a column, which must leave its indexes in place."""
    config, engine = before

    command.upgrade(config, _REVISION)
    command.downgrade(config, _BEFORE)
    command.upgrade(config, _REVISION)

    with engine.connect() as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'api_keys'")
            ).all()
        }
    assert "uq_api_keys_config_name" in indexes
    assert "ix_api_keys_key_hash" in indexes
