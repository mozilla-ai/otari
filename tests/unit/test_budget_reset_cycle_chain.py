"""The reset-cycle revision's Alembic chain, exercised on SQLite.

The revision converts every budget's period and rebuilds ``budgets`` to swap one
CHECK for ten, which SQLite can only do through a batch rebuild. A rebuild drops
whatever ``copy_from`` does not declare, and the conversion is a product decision
per row, so both are pinned here. Every integration run migrates PostgreSQL and
nothing migrates SQLite, so this is the only coverage of that path. Modeled on
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
_REVISION = "d4b8e2f6a917"
_PREVIOUS_REVISION = "a9d3e5f7b1c2"

# One row per conversion arm: (budget_id, budget_duration_sec, reset_alignment).
_OLD_ROWS = [
    ("day", None, "calendar_day"),
    ("week", None, "calendar_week"),
    ("month", None, "calendar_month"),
    ("two-days", 2 * 86400, None),
    ("six-hours", 6 * 3600, None),
    ("ninety-minutes", 90 * 60, None),
    ("five-minutes", 300, None),
    ("never", None, None),
]


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def migrated(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database holding one budget per old period, migrated through the revision."""
    database_url = f"sqlite:///{tmp_path / 'reset_cycles.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _PREVIOUS_REVISION)
    engine = create_engine(database_url)
    with engine.begin() as connection:
        for budget_id, duration, alignment in _OLD_ROWS:
            connection.execute(
                text(
                    "INSERT INTO budgets (budget_id, budget_duration_sec, reset_alignment, created_at, updated_at)"
                    " VALUES (:id, :duration, :alignment, '2026-10-01', '2026-10-01')"
                ),
                {"id": budget_id, "duration": duration, "alignment": alignment},
            )
    command.upgrade(config, _REVISION)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _cycles(engine: Engine) -> dict[str, tuple[object, ...]]:
    with engine.begin() as connection:
        rows = connection.execute(
            text(
                "SELECT budget_id, reset_cycle, reset_every_n, reset_anchor_at IS NOT NULL,"
                " reset_weekdays, reset_month_day, reset_month FROM budgets"
            )
        ).all()
    return {row[0]: tuple(row[1:]) for row in rows}


def test_every_old_period_converts_to_its_cycle(migrated: tuple[Config, Engine]) -> None:
    """Exact where the new vocabulary has the spelling, and rounded up to the hour floor where it does not."""
    _, engine = migrated
    assert _cycles(engine) == {
        "day": ("daily", None, 0, None, None, None),
        "week": ("weekly", None, 0, 1, None, None),
        "month": ("monthly", None, 0, None, 1, None),
        "two-days": ("every_n_days", 2, 1, None, None, None),
        "six-hours": ("every_n_hours", 6, 1, None, None, None),
        "ninety-minutes": ("every_n_hours", 2, 1, None, None, None),
        "five-minutes": ("every_n_hours", 1, 1, None, None, None),
        "never": (None, None, 0, None, None, None),
    }


def test_the_rebuild_keeps_the_foreign_key_and_its_index(migrated: tuple[Config, Engine]) -> None:
    _, engine = migrated
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("budgets")}
    assert {"budget_duration_sec", "reset_alignment"}.isdisjoint(columns)
    assert {"rpm_limit", "tpm_limit", "organization_id"} <= columns
    assert "fk_budgets_organization_id" in {fk["name"] for fk in inspector.get_foreign_keys("budgets")}
    assert "ix_budgets_organization_id" in {index["name"] for index in inspector.get_indexes("budgets")}


def test_a_cycle_missing_its_setting_is_refused(migrated: tuple[Config, Engine]) -> None:
    _, engine = migrated
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO budgets (budget_id, reset_cycle, created_at, updated_at)"
                " VALUES ('weekly-no-days', 'weekly', '2026-10-01', '2026-10-01')"
            )
        )


def test_downgrade_never_admits_more_than_the_cycle_did(migrated: tuple[Config, Engine]) -> None:
    """A yearly budget has no old spelling and comes back as never resetting, not monthly."""
    config, engine = migrated
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO budgets (budget_id, reset_cycle, reset_month_day, reset_month, created_at, updated_at)"
                " VALUES ('year', 'yearly', 1, 1, '2026-10-01', '2026-10-01')"
            )
        )

    command.downgrade(config, _PREVIOUS_REVISION)

    with engine.begin() as connection:
        rows = connection.execute(text("SELECT budget_id, budget_duration_sec, reset_alignment FROM budgets")).all()
    assert {row[0]: (row[1], row[2]) for row in rows} == {
        "day": (None, "calendar_day"),
        "week": (None, "calendar_week"),
        "month": (None, "calendar_month"),
        "two-days": (2 * 86400, None),
        "six-hours": (6 * 3600, None),
        "ninety-minutes": (2 * 3600, None),
        "five-minutes": (3600, None),
        "never": (None, None),
        "year": (None, None),
    }


def test_upgrade_downgrade_upgrade_round_trips(migrated: tuple[Config, Engine]) -> None:
    config, engine = migrated
    before = _cycles(engine)

    command.downgrade(config, _PREVIOUS_REVISION)
    command.upgrade(config, _REVISION)

    assert _cycles(engine) == before
    assert "fk_budgets_organization_id" in {fk["name"] for fk in inspect(engine).get_foreign_keys("budgets")}
