"""Tests that the organization invitation migrations upgrade and downgrade on SQLite."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

import gateway.models  # noqa: F401  (registers every table on the shared metadata)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "3ec751593322"
_DELIVERY_REVISION = "62256ffefe75"
_SEND_REQUEST_REVISION = "a6c9fa1da92a"
_DELIVERY_COLUMNS = {"next_send_at", "delivery_attempts"}
_NEXT_SEND_AT_INDEX = "ix_organization_invitation_next_send_at"
_TABLES = {"organization_invitation", "organization_membership_origin"}


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def _sqlite_upgraded_to(tmp_path: Path, target: str) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'invitation.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, target)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


@pytest.fixture
def sqlite_at_head(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """Yield a SQLite database at the latest migration, so these tables are checked as they survive to it."""
    yield from _sqlite_upgraded_to(tmp_path, "head")


@pytest.fixture
def sqlite_at_revision(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """Yield a SQLite database at this revision, so a relative downgrade target names this migration."""
    yield from _sqlite_upgraded_to(tmp_path, _REVISION)


def test_upgrade_creates_both_tables(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    assert _TABLES <= set(inspect(engine).get_table_names())


def test_an_organization_has_one_invitation_per_address(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    constraints = {
        constraint["name"]: constraint["column_names"]
        for constraint in inspect(engine).get_unique_constraints("organization_invitation")
    }
    assert constraints["uq_organization_invitation_organization_email"] == ["organization_id", "email"]


def test_the_invitation_table_carries_every_index(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    indexes = {
        index["name"]: (index["column_names"], index["unique"])
        for index in inspect(engine).get_indexes("organization_invitation")
    }
    assert indexes["ix_organization_invitation_expires_at"] == (["expires_at"], False)
    assert indexes["ix_organization_invitation_invited_by_user_id"] == (["invited_by_user_id"], False)
    assert indexes[_NEXT_SEND_AT_INDEX] == (["next_send_at"], False)
    assert indexes["ix_organization_invitation_token_hash"] == (["token_hash"], True)


def test_the_membership_origins_inviter_column_is_indexed(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    indexes = {
        index["name"]: index["column_names"] for index in inspect(engine).get_indexes("organization_membership_origin")
    }
    assert indexes["ix_organization_membership_origin_invited_by_user_id"] == ["invited_by_user_id"]


def test_upgrade_downgrade_upgrade_round_trips(sqlite_at_revision: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_revision

    command.downgrade(config, f"{_REVISION}-1")
    assert not (_TABLES & set(inspect(engine).get_table_names()))

    command.upgrade(config, _REVISION)
    assert _TABLES <= set(inspect(engine).get_table_names())


def _invitation_columns(engine: Engine) -> dict[str, bool]:
    return {column["name"]: column["nullable"] for column in inspect(engine).get_columns("organization_invitation")}


def _invitation_indexes(engine: Engine) -> set[str | None]:
    return {index["name"] for index in inspect(engine).get_indexes("organization_invitation")}


def test_the_invitation_table_carries_the_delivery_columns(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    columns = _invitation_columns(engine)
    assert (columns["next_send_at"], columns["delivery_attempts"]) == (True, False)


def test_an_invitation_saved_before_the_delivery_columns_has_no_attempts(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'existing.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _REVISION)
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO organization_invitation (id, organization_id, email, role, token_hash,"
                    " workspace_assignments, expires_at, send_count, created_at)"
                    " VALUES ('i1', 'o1', 'ada@example.com', 'member', 't1', '[]',"
                    " '2030-01-01 00:00:00', 0, '2026-01-01 00:00:00')"
                )
            )
        command.upgrade(config, "head")
        with engine.connect() as connection:
            row = connection.execute(text("SELECT next_send_at, delivery_attempts FROM organization_invitation")).one()
    finally:
        engine.dispose()

    assert tuple(row) == (None, 0)


def test_the_delivery_columns_round_trip(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head

    command.downgrade(config, _REVISION)
    assert not (_DELIVERY_COLUMNS & set(_invitation_columns(engine)))
    assert _NEXT_SEND_AT_INDEX not in _invitation_indexes(engine)

    command.upgrade(config, _DELIVERY_REVISION)
    assert _DELIVERY_COLUMNS <= set(_invitation_columns(engine))
    assert _NEXT_SEND_AT_INDEX in _invitation_indexes(engine)


def test_a_null_last_send_requested_at_is_backfilled_from_created_at(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'backfill-send-request.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _DELIVERY_REVISION)
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO organization_invitation (id, organization_id, email, role, token_hash,"
                    " workspace_assignments, expires_at, send_count, created_at)"
                    " VALUES ('i1', 'o1', 'ada@example.com', 'member', 't1', '[]',"
                    " '2030-01-01 00:00:00', 0, '2026-01-01 00:00:00')"
                )
            )
        command.upgrade(config, "head")
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT last_send_requested_at, created_at FROM organization_invitation")
            ).one()
        nullable = _invitation_columns(engine)["last_send_requested_at"]
    finally:
        engine.dispose()

    assert row.last_send_requested_at == row.created_at
    assert nullable is False


def test_last_send_requested_at_nullability_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head
    assert _invitation_columns(engine)["last_send_requested_at"] is False

    command.downgrade(config, _DELIVERY_REVISION)
    assert _invitation_columns(engine)["last_send_requested_at"] is True

    command.upgrade(config, _SEND_REQUEST_REVISION)
    assert _invitation_columns(engine)["last_send_requested_at"] is False


def test_the_invitation_table_keeps_its_foreign_keys_through_the_send_request_revision(
    sqlite_at_head: tuple[Config, Engine],
) -> None:
    _, engine = sqlite_at_head
    foreign_keys = {
        tuple(key["constrained_columns"]): key["referred_table"]
        for key in inspect(engine).get_foreign_keys("organization_invitation")
    }
    assert foreign_keys[("organization_id",)] == "organization"
    assert foreign_keys[("invited_by_user_id",)] == "user"
