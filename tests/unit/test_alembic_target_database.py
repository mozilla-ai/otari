"""Which database otari's migration chain is willing to run against.

Otari stamps the default ``alembic_version`` table, the same one every other
Alembic application stamps, so nothing about a URL says whose database it names.
Two ways a deployment ends up pointed at someone else's, both seen in practice
when otari's control plane runs beside an older application: the process
inherits that application's ``DATABASE_URL``, or it is handed one outright.
Alembic's own report of that is "Can't locate revision identified by ...", which
reads as a corrupt database rather than as a URL naming the wrong one.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.util import CommandError
from sqlalchemy import create_engine, inspect

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_DIR = _REPO_ROOT / "alembic"

# A revision from otari-ai's platform chain, which is exactly the history a
# control plane sharing that deployment's environment can be aimed at.
_FOREIGN_REVISION = "a5f83c1d6e07"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def _stamp(db_path: Path, revision: str) -> None:
    """Give a database an ``alembic_version`` row from another chain."""
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        connection.execute("INSERT INTO alembic_version (version_num) VALUES (?)", (revision,))
        connection.commit()
    finally:
        connection.close()


def _version_rows(db_path: Path) -> list[tuple[str]]:
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute("SELECT version_num FROM alembic_version").fetchall()
    finally:
        connection.close()


def _run_alembic(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Drive the CLI, which is the entry point an operator has."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "alembic", "-c", str(_REPO_ROOT / "alembic.ini"), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_upgrade_refuses_a_database_stamped_by_another_chain(tmp_path: Path) -> None:
    """The error names the target and the revision, not just the revision.

    This is the path ``auto_migrate`` and ``otari migrate`` both take, so the
    refusal is what a control plane boots into rather than a schema written over
    another application's tables.
    """
    db_path = tmp_path / "platform.db"
    _stamp(db_path, _FOREIGN_REVISION)

    with pytest.raises(CommandError) as excinfo:
        command.upgrade(_alembic_config(f"sqlite:///{db_path}"), "head")

    message = str(excinfo.value)
    assert _FOREIGN_REVISION in message
    assert "platform.db" in message
    assert "OTARI_DATABASE_URL" in message
    # Refused before writing: the foreign database still holds only its own table.
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        assert inspect(engine).get_table_names() == ["alembic_version"]
    finally:
        engine.dispose()


def test_stamp_still_rewrites_a_foreign_version_table(tmp_path: Path) -> None:
    """The refusal must not take the repair for it away.

    ``alembic stamp --purge`` is what an operator runs on a version table that
    holds the wrong row, so a check that refused to run it would leave a
    database that can only be fixed by hand.
    """
    db_path = tmp_path / "platform.db"
    _stamp(db_path, _FOREIGN_REVISION)
    env = {key: value for key, value in os.environ.items() if key != "DATABASE_URL"}
    env["OTARI_DATABASE_URL"] = f"sqlite:///{db_path}"

    result = _run_alembic(["stamp", "head", "--purge"], cwd=tmp_path, env=env)

    assert result.returncode == 0, result.stderr
    head = ScriptDirectory.from_config(_alembic_config("sqlite://")).get_current_head()
    assert _version_rows(db_path) == [(head,)]


def test_a_bare_alembic_run_refuses_a_foreign_database_url(tmp_path: Path) -> None:
    """``DATABASE_URL`` alone selects nothing, and selecting nothing is an error.

    ``otari serve`` and ``otari migrate`` read that name themselves and pass the
    URL on explicitly, so a bare ``alembic`` invocation reading it too only ever
    picked up the value some other application left in the environment. Falling
    back to the SQLite default instead would be the same wrong-database run with
    the failure taken out of it: a throwaway file migrated in the working
    directory, exit 0, and the operator's database untouched.
    """
    foreign = tmp_path / "platform.db"
    env = {key: value for key, value in os.environ.items() if key != "OTARI_DATABASE_URL"}
    env["DATABASE_URL"] = f"sqlite:///{foreign}"

    result = _run_alembic(["current"], cwd=tmp_path, env=env)

    assert result.returncode != 0
    assert "OTARI_DATABASE_URL" in result.stderr
    assert not foreign.exists(), "DATABASE_URL selected the database this chain connected to"
    assert list(tmp_path.iterdir()) == [], "a default URL migrated a database nobody named"
