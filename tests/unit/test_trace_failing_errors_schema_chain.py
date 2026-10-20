"""The revision that counts the errors that fail a trace's session apart, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half: an
existing trace's failing errors are backfilled from its spans, and the downgrade
drops the column again.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "a8c2e4f6b0d1"
_BEFORE = "f0f9d05dafc6"
_WORKSPACE = "00000000000000000000000000000001"

_INSERT_TRACE = text(
    "INSERT INTO traces (workspace_id, trace_id, session_source, started_at, last_activity_at, error_count)"
    " VALUES (:workspace, :trace, 'client', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, :errors)"
)
_INSERT_SPAN = text(
    "INSERT INTO trace_spans (workspace_id, trace_id, span_id, kind, origin, name, outcome, recovered)"
    " VALUES (:workspace, :trace, :span, :kind, 'gateway', 'span', :outcome, :recovered)"
)


@pytest.fixture
def chain(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'trace_failing_errors.db'}"
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


def test_an_existing_trace_has_its_failing_errors_backfilled(chain: tuple[Config, Engine]) -> None:
    config, engine = chain
    spans = [
        # trace, span, kind, outcome, recovered
        ("recovered-run", "step-1", "step", "ok", False),
        ("recovered-run", "tool-1", "tool", "error", False),
        ("recovered-run", "tool-2", "tool", "error", False),
        ("recovered-run", "tool-3", "tool", "ok", False),
        ("failed-run", "step-1", "step", "error", False),
        ("failed-run", "llm-1", "llm", "error", True),
        ("failed-run", "tool-1", "tool", "error", False),
        ("clean-run", "step-1", "step", "ok", False),
    ]
    with engine.begin() as connection:
        for trace, errors in (("recovered-run", 2), ("failed-run", 2), ("clean-run", 0)):
            connection.execute(_INSERT_TRACE, {"workspace": _WORKSPACE, "trace": trace, "errors": errors})
        for trace, span, kind, outcome, recovered in spans:
            connection.execute(
                _INSERT_SPAN,
                {
                    "workspace": _WORKSPACE,
                    "trace": trace,
                    "span": span,
                    "kind": kind,
                    "outcome": outcome,
                    "recovered": recovered,
                },
            )

    command.upgrade(config, _REVISION)

    with engine.begin() as connection:
        rows = connection.execute(
            text("SELECT trace_id, error_count, failing_error_count FROM traces ORDER BY trace_id")
        )
        assert [tuple(row) for row in rows] == [("clean-run", 0, 0), ("failed-run", 2, 1), ("recovered-run", 2, 0)]


def test_the_downgrade_drops_the_column(chain: tuple[Config, Engine]) -> None:
    config, engine = chain
    command.upgrade(config, _REVISION)
    command.downgrade(config, _BEFORE)

    assert "failing_error_count" not in {column["name"] for column in inspect(engine).get_columns("traces")}
    command.upgrade(config, _REVISION)
    assert "failing_error_count" in {column["name"] for column in inspect(engine).get_columns("traces")}
