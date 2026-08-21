"""The ``reset_alignment`` revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default, and this revision adds a CHECK constraint
to an existing table, which SQLite has no ``ALTER TABLE ... ADD CONSTRAINT`` for:
Alembic's batch mode rebuilds the table instead, and a rebuild is what silently
drops what reflection could not see. ``scoped_budgets`` carries two *partial*
unique indexes, which is exactly that, so ``copy_from`` in the revision is what
has to carry them across.

Every integration run migrates PostgreSQL and nothing migrates SQLite, so this is
the only coverage of that path. Driven against a real file database rather than
in-memory because the rebuild is what is under test. Modeled on
``test_tenancy_schema_chain.py``.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_ALIGNMENT_REVISION = "b6e8c2a4d7f1"
_PREVIOUS_REVISION = "a3c7e1b9d5f2"
_CHECK = "ck_scoped_budgets_single_period_source"
_INDEXES = {
    "uq_scoped_budgets_scope_with_key",
    "uq_scoped_budgets_scope_no_key",
    "ix_scoped_budgets_scope",
}

_INSERT = text(
    "INSERT INTO scoped_budgets"
    " (id, scope_type, scope_id, current_spend, reserved_spend, budget_duration_sec, reset_alignment,"
    "  created_at, updated_at)"
    " VALUES (:id, 'workspace', :scope_id, 0, 0, :duration, :alignment, '2026-08-19', '2026-08-19')"
)


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_at_head(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database migrated to the alignment revision, with its config."""
    database_url = f"sqlite:///{tmp_path / 'alignment.db'}"
    config = _alembic_config(database_url)
    # This revision, not ``head``: a later revision moves the cadence onto
    # ``budgets`` and drops these columns, and what is under test here is this
    # revision's own batch rebuild.
    command.upgrade(config, _ALIGNMENT_REVISION)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _insert(engine: Engine, budget_id: str, *, duration: int | None, alignment: str | None) -> None:
    with engine.begin() as connection:
        connection.execute(
            _INSERT,
            {
                "id": budget_id,
                "scope_id": budget_id,
                "duration": duration,
                "alignment": alignment,
            },
        )


def test_upgrade_adds_the_column_and_its_check(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    inspector = inspect(engine)

    alignment = next(
        column for column in inspector.get_columns("scoped_budgets") if column["name"] == "reset_alignment"
    )
    assert alignment["nullable"] is True
    assert _CHECK in {check["name"] for check in inspector.get_check_constraints("scoped_budgets")}


def test_the_partial_unique_indexes_survive_the_batch_rebuild(sqlite_at_head: tuple[Config, Engine]) -> None:
    """``copy_from`` is what guarantees this: SQLite reflection cannot recover a
    partial index's WHERE clause, so a rebuild without it would silently widen
    both uniqueness rules."""
    _, engine = sqlite_at_head
    assert _INDEXES <= {index["name"] for index in inspect(engine).get_indexes("scoped_budgets")}

    with engine.begin() as connection:
        definitions = connection.execute(
            text("SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'scoped_budgets'")
        ).all()
    partial = {name: sql for name, sql in definitions if name.startswith("uq_")}
    assert "WHERE provider_key_id IS NOT NULL" in partial["uq_scoped_budgets_scope_with_key"]
    assert "WHERE provider_key_id IS NULL" in partial["uq_scoped_budgets_scope_no_key"]


def test_each_of_the_three_legal_states_is_storable(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Never resets, rolling, and calendar aligned. Existing rows are the first
    two, and the CHECK must not have made either unwritable."""
    _, engine = sqlite_at_head

    _insert(engine, "never", duration=None, alignment=None)
    _insert(engine, "rolling", duration=86400, alignment=None)
    _insert(engine, "aligned", duration=None, alignment="calendar_month")

    with engine.begin() as connection:
        stored = connection.execute(text("SELECT id FROM scoped_budgets ORDER BY id")).scalars().all()
    assert list(stored) == ["aligned", "never", "rolling"]


def test_the_check_rejects_a_row_carrying_both_periods(sqlite_at_head: tuple[Config, Engine]) -> None:
    """The fourth state has no meaning, so it must not be storable at all."""
    _, engine = sqlite_at_head
    with pytest.raises(IntegrityError):
        _insert(engine, "both", duration=86400, alignment="calendar_month")


def test_downgrade_drops_the_column_and_keeps_the_rest(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head
    _insert(engine, "rolling", duration=86400, alignment=None)

    command.downgrade(config, _PREVIOUS_REVISION)

    inspector = inspect(engine)
    assert "reset_alignment" not in {column["name"] for column in inspector.get_columns("scoped_budgets")}
    assert _CHECK not in {check["name"] for check in inspector.get_check_constraints("scoped_budgets")}
    assert _INDEXES <= {index["name"] for index in inspector.get_indexes("scoped_budgets")}
    with engine.begin() as connection:
        assert connection.execute(text("SELECT budget_duration_sec FROM scoped_budgets")).scalar() == 86400


def test_upgrade_downgrade_upgrade_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head

    command.downgrade(config, _PREVIOUS_REVISION)
    command.upgrade(config, _ALIGNMENT_REVISION)

    inspector = inspect(engine)
    assert "reset_alignment" in {column["name"] for column in inspector.get_columns("scoped_budgets")}
    assert _CHECK in {check["name"] for check in inspector.get_check_constraints("scoped_budgets")}
    assert _INDEXES <= {index["name"] for index in inspector.get_indexes("scoped_budgets")}
