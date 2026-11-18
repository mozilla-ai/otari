"""The revision recording a sign-in's workspace, on SQLite: the column comes and goes, and the chain upgrades again.

The integration suite migrates PostgreSQL only, so this is the SQLite half.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "da9ab6d90202"
_BEFORE = "124a7328c1b3"
_TABLE = "mcp_oauth_flows"


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'sign_in_workspace.db'}"
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


def _shape(engine: Engine) -> tuple[set[str], set[str | None], set[str]]:
    schema = inspect(engine)
    return (
        {column["name"] for column in schema.get_columns(_TABLE)},
        {index["name"] for index in schema.get_indexes(_TABLE)},
        {key["referred_table"] for key in schema.get_foreign_keys(_TABLE)},
    )


def test_a_sign_in_records_its_workspace_after_the_revision_and_not_before(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before
    columns, indexes, keys = _shape(engine)
    assert "workspace_id" not in columns

    command.upgrade(config, _REVISION)
    upgraded = _shape(engine)
    assert "workspace_id" in upgraded[0]
    assert upgraded[1] == indexes | {"ix_mcp_oauth_flows_workspace_id"}
    assert upgraded[2] == keys | {"workspace"}

    command.downgrade(config, _BEFORE)
    assert _shape(engine) == (columns, indexes, keys)

    command.upgrade(config, _REVISION)
    assert "workspace_id" in _shape(engine)[0]
