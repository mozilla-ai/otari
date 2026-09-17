"""The test fixture that starts a new SQLite database from a migrated template."""

import sqlite3
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from conftest import ROOT, _new_sqlite_file
from gateway.core import database


def test_a_missing_file_in_an_existing_directory_is_new(tmp_path: Path) -> None:
    assert _new_sqlite_file(f"sqlite:///{tmp_path / 'gateway.db'}") == tmp_path / "gateway.db"
    assert _new_sqlite_file(f"sqlite+aiosqlite:///{tmp_path / 'gateway.db'}") == tmp_path / "gateway.db"


def test_an_existing_file_is_not_new(tmp_path: Path) -> None:
    existing = tmp_path / "gateway.db"
    existing.touch()

    assert _new_sqlite_file(f"sqlite:///{existing}") is None


@pytest.mark.parametrize(
    "database_url",
    [
        "sqlite://",
        "sqlite:///:memory:",
        "sqlite:///file:gateway?mode=memory&uri=true",
        "postgresql://user@localhost/gateway",
        "sqlite:////no/such/directory/gateway.db",
    ],
)
def test_a_database_that_is_not_a_new_file_is_left_to_the_real_migrations(database_url: str) -> None:
    assert _new_sqlite_file(database_url) is None


def test_a_new_database_starts_at_the_head_revision(tmp_path: Path) -> None:
    path = tmp_path / "gateway.db"

    database._run_migrations(f"sqlite:///{path}")

    config = Config()
    config.set_main_option("script_location", str(ROOT / "alembic"))
    with sqlite3.connect(path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    assert revision == ScriptDirectory.from_config(config).get_current_head()
