"""The budget-counter revision's Alembic chain, exercised on SQLite.

Sibling of ``test_exact_money_schema_chain.py``, and for the same reason: the
OSS base ships SQLite by default, and a revision that retypes a column can only
do that on SQLite by rebuilding the table. A rebuild is where a constraint goes
missing, and since ``f3a5c7e9d1b4`` made ``budgets`` the one table holding a
limit, it is also the one carrying the rule worth losing sleep over: a CHECK
that a budget names one period source rather than two.

The other half of the file is the question the change has to answer before it
can be trusted: **what happens to a counter that is already drifted when it
converts?** The answer is that it lands on the amount it should have held all
along, which is the drift being discarded rather than money.
"""

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

import gateway.models  # noqa: F401  (registers every table on the shared metadata)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_COUNTER_REVISION = "b3f8d1c6a4e7"
_BEFORE_COUNTERS = "f3a5c7e9d1b4"

# Every counter the revision retypes, as ``table -> columns``.
_COUNTERS = {
    "budgets": ("max_budget",),
    "users": ("spend", "reserved"),
    "budget_reset_logs": ("previous_spend",),
    "scoped_budgets": ("current_spend", "reserved_spend"),
}

_SCOPED_INDEXES = {
    "uq_scoped_budgets_scope_with_key",
    "uq_scoped_budgets_scope_no_key",
    "ix_scoped_budgets_scope",
}

_NOW = "2026-08-21 00:00:00"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_before_counters(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database migrated to the revision before the budget counters."""
    database_url = f"sqlite:///{tmp_path / 'counters.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _BEFORE_COUNTERS)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _column_types(engine: Engine, table: str) -> dict[str, str]:
    return {column["name"]: str(column["type"]) for column in inspect(engine).get_columns(table)}


def _seed_float_rows(engine: Engine) -> None:
    """Write the drifted counters a float-era deployment holds."""
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
                "VALUES ('u1', 0.6619999999999999, 0.30000000000000004, 'b1', 0, :n, :n, '{}')"
            ),
            {"n": _NOW},
        )
        connection.execute(
            text(
                "INSERT INTO scoped_budgets (id, scope_type, scope_id, budget_id, current_spend, reserved_spend, "
                "created_at, updated_at) VALUES ('s1', 'workspace', 'ws-1', 'b1', 0.1 + 0.2, 0.0, :n, :n)"
            ),
            {"n": _NOW},
        )


def test_every_budget_counter_becomes_numeric(sqlite_before_counters: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before_counters
    assert _column_types(engine, "users")["spend"] == "FLOAT"

    command.upgrade(config, _COUNTER_REVISION)

    for table, columns in _COUNTERS.items():
        types = _column_types(engine, table)
        for column in columns:
            assert types[column] == "NUMERIC(18, 6)", f"{table}.{column}"


def test_a_drifted_counter_converges_on_the_amount_it_should_have_held(
    sqlite_before_counters: tuple[Config, Engine],
) -> None:
    """``0.6619999999999999`` is not a smaller number than ``0.662``, it is the
    same number written by an implementation that could not write the right one.
    Six decimals is where the difference between them stops existing.
    """
    config, engine = sqlite_before_counters
    _seed_float_rows(engine)

    command.upgrade(config, _COUNTER_REVISION)

    with engine.connect() as connection:
        # Read the raw storage rather than through the ORM: on SQLite the value
        # is still a REAL in the file, and the point is that it now formats to
        # the column's scale, so what the gateway compares against is exact.
        spend, reserved = connection.execute(text("SELECT spend, reserved FROM users")).one()
        current = connection.execute(text("SELECT current_spend FROM scoped_budgets")).scalar_one()
    assert Decimal(str(spend)).quantize(Decimal("0.000001")) == Decimal("0.662000")
    assert Decimal(str(reserved)).quantize(Decimal("0.000001")) == Decimal("0.300000")
    assert Decimal(str(current)).quantize(Decimal("0.000001")) == Decimal("0.300000")


def test_the_budget_period_check_survives_the_rebuild(sqlite_before_counters: tuple[Config, Engine]) -> None:
    """The CHECK is why the revision hands batch mode an explicit ``copy_from``.

    A reflected SQLite ``Table`` carries no CHECK constraints, so a
    reflection-driven rebuild would drop the rule keeping ``(86400,
    calendar_month)`` out of the table. ``scoped_budgets`` is rebuilt by
    reflection instead, so its partial unique indexes are checked here too:
    those *are* reflected faithfully, and this is what says so.
    """
    config, engine = sqlite_before_counters
    _seed_float_rows(engine)

    command.upgrade(config, _COUNTER_REVISION)

    checks = {check["name"] for check in inspect(engine).get_check_constraints("budgets")}
    assert "ck_budgets_single_period_source" in checks
    assert _SCOPED_INDEXES <= {index["name"] for index in inspect(engine).get_indexes("scoped_budgets")}

    with engine.begin() as connection, pytest.raises(Exception, match="CHECK constraint failed"):
        connection.execute(
            text(
                "INSERT INTO budgets (budget_id, budget_duration_sec, reset_alignment, created_at, updated_at) "
                "VALUES ('b2', 86400, 'calendar_month', :n, :n)"
            ),
            {"n": _NOW},
        )
    with engine.begin() as connection, pytest.raises(Exception, match="UNIQUE constraint failed"):
        # The seeded ceiling is the aggregate one for its scope, so a second is
        # refused by the partial unique index the rebuild had to carry across.
        connection.execute(
            text(
                "INSERT INTO scoped_budgets (id, scope_type, scope_id, budget_id, current_spend, reserved_spend, "
                "created_at, updated_at) VALUES ('s2', 'workspace', 'ws-1', 'b1', 0, 0, :n, :n)"
            ),
            {"n": _NOW},
        )


def test_the_conversion_is_reversible(sqlite_before_counters: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before_counters
    _seed_float_rows(engine)

    command.upgrade(config, _COUNTER_REVISION)
    command.downgrade(config, _BEFORE_COUNTERS)

    for table, columns in _COUNTERS.items():
        types = _column_types(engine, table)
        for column in columns:
            assert types[column] == "FLOAT", f"{table}.{column}"
    checks = {check["name"] for check in inspect(engine).get_check_constraints("budgets")}
    assert "ck_budgets_single_period_source" in checks
    with engine.connect() as connection:
        assert connection.execute(text("SELECT max_budget FROM budgets")).scalar_one() == 25.0
