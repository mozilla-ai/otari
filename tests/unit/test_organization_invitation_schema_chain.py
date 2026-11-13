"""Tests that the organization invitation migration upgrades and downgrades on SQLite."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect

import gateway.models  # noqa: F401  (registers every table on the shared metadata)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "3ec751593322"
_TABLES = {"organization_invitation", "organization_membership_origin"}


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_at_head(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """Yield a SQLite database at the latest migration, with the Alembic config that built it."""
    database_url = f"sqlite:///{tmp_path / 'invitation.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


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
    assert indexes["ix_organization_invitation_token_hash"] == (["token_hash"], True)


def test_the_membership_origins_inviter_column_is_indexed(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    indexes = {
        index["name"]: index["column_names"] for index in inspect(engine).get_indexes("organization_membership_origin")
    }
    assert indexes["ix_organization_membership_origin_invited_by_user_id"] == ["invited_by_user_id"]


def test_upgrade_downgrade_upgrade_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head

    command.downgrade(config, "-1")
    assert not (_TABLES & set(inspect(engine).get_table_names()))

    command.upgrade(config, _REVISION)
    assert _TABLES <= set(inspect(engine).get_table_names())
