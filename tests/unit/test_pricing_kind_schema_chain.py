"""The pricing-kind revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default and the integration suite migrates
PostgreSQL only, so this is that engine's coverage for the revision. It rebuilds
``model_pricing`` to put ``kind`` in its primary key, which is the step SQLite's
batch mode can get wrong. Pinned: every existing row becomes a ``model`` row,
one key can then hold a row of each kind, and the downgrade refuses to drop a
tool's rate.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "d8b2f6a4c0e9"
_BEFORE = "c0e83c6fe801"
_AT = "2026-05-20 00:00:00.000000"


def _alembic_config(database_url: str) -> Config:
    """An Alembic configuration on this repository's revisions and ``database_url``."""
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database migrated to the revision just before ``kind`` arrives."""
    database_url = f"sqlite:///{tmp_path / 'kind.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _insert(engine: Engine, model_key: str, kind: str | None = None) -> None:
    """Store a per-request price row with raw SQL, with ``kind`` only when the schema has it."""
    columns = "kind, " if kind else ""
    values = ":kind, " if kind else ""
    with engine.begin() as connection:
        connection.execute(
            text(
                f"INSERT INTO model_pricing ({columns}model_key, effective_at, input_price_per_million, "  # noqa: S608
                f"output_price_per_million, pricing_tiers, unit, created_at, updated_at) "
                f"VALUES ({values}:key, :at, 5000, 0, '[]', 'requests', :at, :at)"
            ),
            {"kind": kind, "key": model_key, "at": _AT},
        )


def test_existing_rows_become_model_rows_and_a_key_holds_one_row_per_kind(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before
    _insert(engine, "exa:exa")

    command.upgrade(config, _REVISION)
    engine.dispose()

    assert inspect(engine).get_pk_constraint("model_pricing")["constrained_columns"] == [
        "kind",
        "model_key",
        "effective_at",
    ]
    _insert(engine, "exa:exa", kind="search")
    with engine.connect() as connection:
        kinds = connection.execute(text("SELECT kind FROM model_pricing ORDER BY kind")).scalars().all()
    assert kinds == ["model", "search"]
    # The override table's checks survive: SQLite cannot add one back without a rebuild.
    assert len(inspect(engine).get_check_constraints("organization_model_pricing")) == 5


def test_the_downgrade_keeps_model_rows_and_refuses_to_drop_a_tool_rate(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before
    _insert(engine, "exa:exa")
    command.upgrade(config, _REVISION)
    engine.dispose()

    command.downgrade(config, _BEFORE)
    engine.dispose()
    with engine.connect() as connection:
        assert connection.execute(text("SELECT model_key FROM model_pricing")).scalars().all() == ["exa:exa"]

    command.upgrade(config, _REVISION)
    engine.dispose()
    _insert(engine, "exa", kind="search")
    with pytest.raises(RuntimeError, match="search or fetch tool prices"):
        command.downgrade(config, _BEFORE)
