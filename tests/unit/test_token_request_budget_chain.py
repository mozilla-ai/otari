"""The token and request ceiling revision's Alembic chain, exercised on SQLite.

Sibling of ``test_exact_budget_schema_chain.py``. Two things are worth a test
rather than a reading of the migration:

**A live deployment upgrades into zeroed counters, not into a refusal.** The new
columns are NOT NULL, so every existing row needs a value the moment the column
appears, and every INSERT written before they existed needs one too. A
``server_default`` is what supplies both, and the failure it prevents is not
subtle: without it the upgrade itself fails on any non-empty table, and with the
default declared only in the ORM the first insert from an older code path does.

**The rule that survives a rebuild.** SQLite has no ALTER for much of this, so
the whole point of checking after the upgrade is that ``budgets`` still refuses
a row naming two period sources, which is the constraint a table rebuild is
most likely to drop.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

import gateway.models  # noqa: F401  (registers every table on the shared metadata)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "c9f2a6b4e8d7"
_BEFORE = "b7e1c4a9d2f5"

# Every counter the revision adds, as ``table -> columns``, mirroring the
# migration's own map so a column added to one and not the other is a failure.
_COUNTERS = {
    "users": ("current_tokens", "reserved_tokens", "current_requests", "reserved_requests"),
    "scoped_budgets": ("current_tokens", "reserved_tokens", "current_requests", "reserved_requests"),
    "budget_reservations": ("token_estimate", "request_estimate"),
    "budget_reservation_scopes": ("token_amount", "request_amount"),
}

_NOW = "2026-09-02 00:00:00"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_before_limits(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database migrated to the revision before the two new ceilings."""
    database_url = f"sqlite:///{tmp_path / 'limits.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _columns(engine: Engine, table: str) -> dict[str, dict[str, object]]:
    return {column["name"]: dict(column) for column in inspect(engine).get_columns(table)}


def _seed_pre_upgrade_rows(engine: Engine) -> None:
    """A budget, a user holding it, and a ceiling naming it, as they exist today."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO budgets (budget_id, max_budget, budget_duration_sec, created_at, updated_at) "
                "VALUES ('b1', 25.0, 86400, :n, :n)"
            ),
            {"n": _NOW},
        )
        connection.execute(
            text(
                "INSERT INTO users (user_id, spend, reserved, budget_id, blocked, created_at, updated_at, metadata) "
                "VALUES ('u1', 1.5, 0.25, 'b1', 0, :n, :n, '{}')"
            ),
            {"n": _NOW},
        )
        connection.execute(
            text(
                "INSERT INTO scoped_budgets (id, scope_type, scope_id, budget_id, current_spend, reserved_spend, "
                "created_at, updated_at) VALUES ('s1', 'workspace', 'ws-1', 'b1', 2.0, 0.0, :n, :n)"
            ),
            {"n": _NOW},
        )


def test_the_new_limits_are_nullable_and_the_counters_are_not(
    sqlite_before_limits: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before_limits
    assert "token_limit" not in _columns(engine, "budgets")

    command.upgrade(config, _REVISION)

    limits = _columns(engine, "budgets")
    for column in ("token_limit", "request_limit"):
        # Nullable is the contract, not an oversight: NULL is how a budget says
        # it caps nothing on that axis, which is what every existing row means.
        assert limits[column]["nullable"] is True, column
    for table, columns in _COUNTERS.items():
        present = _columns(engine, table)
        for column in columns:
            assert present[column]["nullable"] is False, f"{table}.{column}"


def test_rows_that_predate_the_columns_upgrade_to_zero(sqlite_before_limits: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before_limits
    _seed_pre_upgrade_rows(engine)

    command.upgrade(config, _REVISION)

    with engine.begin() as connection:
        assert connection.execute(
            text("SELECT current_tokens, reserved_tokens, current_requests, reserved_requests FROM users")
        ).one() == (0, 0, 0, 0)
        assert connection.execute(
            text("SELECT current_tokens, reserved_tokens, current_requests, reserved_requests FROM scoped_budgets")
        ).one() == (0, 0, 0, 0)
        # The dollar counters are untouched: this revision adds axes, it does not
        # restate the one that was already being enforced.
        assert connection.execute(text("SELECT spend, reserved FROM users")).one() == (1.5, 0.25)
        # And the limits read as "capped nothing", which is what they were.
        assert connection.execute(text("SELECT token_limit, request_limit FROM budgets")).one() == (None, None)


def test_an_insert_written_before_these_columns_still_works(
    sqlite_before_limits: tuple[Config, Engine],
) -> None:
    """The server-side default, not the ORM one: this INSERT names no counter."""
    config, engine = sqlite_before_limits
    command.upgrade(config, _REVISION)

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO budgets (budget_id, max_budget, created_at, updated_at) "
                "VALUES ('b2', 5.0, :n, :n)"
            ),
            {"n": _NOW},
        )
        connection.execute(
            text(
                "INSERT INTO scoped_budgets (id, scope_type, scope_id, budget_id, current_spend, reserved_spend, "
                "created_at, updated_at) VALUES ('s2', 'workspace', 'ws-2', 'b2', 0.0, 0.0, :n, :n)"
            ),
            {"n": _NOW},
        )
        assert connection.execute(
            text("SELECT current_tokens, current_requests FROM scoped_budgets WHERE id = 's2'")
        ).one() == (0, 0)


def test_the_single_period_source_rule_survives_the_upgrade(
    sqlite_before_limits: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before_limits
    command.upgrade(config, _REVISION)

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO budgets (budget_id, budget_duration_sec, reset_alignment, created_at, updated_at) "
                "VALUES ('b3', 86400, 'calendar_month', :n, :n)"
            ),
            {"n": _NOW},
        )


def test_the_downgrade_takes_every_column_back_off(sqlite_before_limits: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before_limits
    _seed_pre_upgrade_rows(engine)
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)

    assert "token_limit" not in _columns(engine, "budgets")
    for table, columns in _COUNTERS.items():
        present = _columns(engine, table)
        for column in columns:
            assert column not in present, f"{table}.{column}"
