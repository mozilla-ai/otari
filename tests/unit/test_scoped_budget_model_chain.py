"""The model-narrowing revision's Alembic chain, exercised on SQLite.

The revision swaps two partial unique indexes for one expression index and adds
a CHECK through a batch rebuild, which is where SQLite drops whatever
``copy_from`` does not declare. Integration runs migrate PostgreSQL only, so
this is the coverage of that path. Modeled on ``test_scoped_budget_alignment_chain.py``.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "a7c3e9f1b5d2"
_PREVIOUS_REVISION = "d4b8e2f6a917"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_at_revision(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'model.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _REVISION)
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO budgets (budget_id, created_at, updated_at) VALUES ('b1', '2026-10-07', '2026-10-07')")
        )
    try:
        yield config, engine
    finally:
        engine.dispose()


def _insert(engine: Engine, ceiling_id: str, provider: str | None, model: str | None) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO scoped_budgets (id, scope_type, scope_id, provider_key_id, model, budget_id,"
                " created_at, updated_at)"
                " VALUES (:id, 'workspace', 'w1', :provider, :model, 'b1', '2026-10-07', '2026-10-07')"
            ),
            {"id": ceiling_id, "provider": provider, "model": model},
        )


def test_the_rebuild_keeps_every_index(sqlite_at_revision: tuple[Config, Engine]) -> None:
    _config, engine = sqlite_at_revision
    # sqlite_master rather than the inspector, which skips expression indexes.
    with engine.connect() as connection:
        names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'scoped_budgets'")
            ).scalars()
        )
    assert {"uq_scoped_budgets_entity", "ix_scoped_budgets_scope", "ix_scoped_budgets_budget_id"} <= names
    assert not names & {"uq_scoped_budgets_scope_with_key", "uq_scoped_budgets_scope_no_key"}


@pytest.mark.parametrize(("provider", "model"), [(None, None), ("openai", None), ("openai", "gpt-4o")])
def test_one_ceiling_per_entity(
    sqlite_at_revision: tuple[Config, Engine], provider: str | None, model: str | None
) -> None:
    _config, engine = sqlite_at_revision
    _insert(engine, "first", provider, model)
    with pytest.raises(IntegrityError):
        _insert(engine, "second", provider, model)


def test_every_shape_coexists_on_one_scope(sqlite_at_revision: tuple[Config, Engine]) -> None:
    _config, engine = sqlite_at_revision
    _insert(engine, "all", None, None)
    _insert(engine, "openai", "openai", None)
    _insert(engine, "gpt-4o", "openai", "gpt-4o")
    _insert(engine, "gpt-4o-mini", "openai", "gpt-4o-mini")
    _insert(engine, "anthropic-gpt-4o", "anthropic", "gpt-4o")


def test_a_model_without_a_provider_is_refused(sqlite_at_revision: tuple[Config, Engine]) -> None:
    _config, engine = sqlite_at_revision
    with pytest.raises(IntegrityError):
        _insert(engine, "orphan", None, "gpt-4o")


def test_downgrade_drops_model_ceilings_and_restores_the_partial_indexes(
    sqlite_at_revision: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_at_revision
    _insert(engine, "openai", "openai", None)
    _insert(engine, "gpt-4o", "openai", "gpt-4o")

    command.downgrade(config, _PREVIOUS_REVISION)

    inspector = inspect(engine)
    assert "model" not in {column["name"] for column in inspector.get_columns("scoped_budgets")}
    names = {index["name"] for index in inspector.get_indexes("scoped_budgets")}
    assert {"uq_scoped_budgets_scope_with_key", "uq_scoped_budgets_scope_no_key"} <= names
    with engine.connect() as connection:
        assert connection.execute(text("SELECT id FROM scoped_budgets")).scalars().all() == ["openai"]
