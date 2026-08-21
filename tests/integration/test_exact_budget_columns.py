"""What the exact budget columns refuse, at the API and at the migration.

``NUMERIC(18, 6)`` holds just under $1T where the float columns it replaced had
no ceiling (mozilla-ai/otari#691). A cap is the one amount here an operator types
by hand, and reaching for a very large number to mean "no limit" is a habit, so
both ends of that are covered: a request body naming a bigger cap is refused with
a 422 rather than reaching PostgreSQL and coming back as a 500, and a deployment
that already stored one, in either direction, is told which table and column
before the migration starts rather than after an ALTER fails.
"""

import uuid
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.models.money import MAX_USD_LIMIT

_BEFORE_COUNTERS = "f3a5c7e9d1b4"
_COUNTERS_REVISION = "b3f8d1c6a4e7"
# Above the column's ceiling, and the shape of a "no limit" sentinel.
_TOO_LARGE = 1e15


def test_a_budget_cap_above_the_column_ceiling_is_refused(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """422 from the schema, not a 500 from a numeric overflow two layers down."""
    assert client.post("/v1/budgets", json={"max_budget": _TOO_LARGE}, headers=master_key_header).status_code == 422
    created = client.post("/v1/budgets", json={"max_budget": MAX_USD_LIMIT}, headers=master_key_header)
    assert created.status_code == 200, created.text
    assert created.json()["max_budget"] == MAX_USD_LIMIT

    budget_id = created.json()["budget_id"]
    assert (
        client.patch(f"/v1/budgets/{budget_id}", json={"max_budget": _TOO_LARGE}, headers=master_key_header).status_code
        == 422
    )


def test_the_only_cap_surface_is_the_one_that_is_bounded(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A scoped ceiling names a budget rather than carrying its own limit, so
    the bound on ``/v1/budgets`` is the whole of the guard. Asserted rather than
    assumed: a future surface that reintroduced a cap field would need bounding
    too, and this is what would notice.
    """
    body: dict[str, Any] = {
        "scope_type": "workspace",
        "scope_id": str(uuid.uuid4()),
        "budget_id": "no-such-budget",
        "max_budget": _TOO_LARGE,
    }
    response = client.post("/v1/scoped-budgets", json=body, headers=master_key_header)
    # 404 for the unknown budget, never 200: the cap field is not a field here.
    assert response.status_code == 404, response.text


def test_the_migration_names_the_row_it_cannot_convert(postgres_url: str) -> None:
    """A stored cap too large for the column stops the upgrade legibly.

    Run on a database of its own: the guard only has anything to say about a
    schema that is still float, and the suite's shared database is at head.
    """
    admin = create_engine(postgres_url, isolation_level="AUTOCOMMIT")
    scratch = f"otari_preflight_{uuid.uuid4().hex[:12]}"
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{scratch}"'))
        scratch_url = create_engine(postgres_url).url.set(database=scratch).render_as_string(hide_password=False)
        config = Config()
        config.set_main_option("script_location", str(Path(__file__).parents[2] / "alembic"))
        config.set_main_option("sqlalchemy.url", scratch_url)
        config.attributes["configure_logger"] = False
        command.upgrade(config, _BEFORE_COUNTERS)

        engine = create_engine(scratch_url)
        try:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO budgets (budget_id, max_budget, created_at, updated_at) "
                        "VALUES ('too-big', :cap, now(), now())"
                    ),
                    {"cap": _TOO_LARGE},
                )
            with pytest.raises(RuntimeError, match=r"budgets\.max_budget"):
                command.upgrade(config, _COUNTERS_REVISION)

            # The range is symmetric, so the guard has to be. A one-sided check
            # would pass this row straight into the ALTER it exists to precede.
            with engine.begin() as connection:
                connection.execute(
                    text("UPDATE budgets SET max_budget = :cap WHERE budget_id = 'too-big'"),
                    {"cap": -_TOO_LARGE},
                )
            with pytest.raises(RuntimeError, match=r"budgets\.max_budget"):
                command.upgrade(config, _COUNTERS_REVISION)

            # The guard ran before any DDL, so the schema is untouched and the
            # operator can lower the cap and migrate again.
            with engine.connect() as connection:
                assert (
                    connection.execute(
                        text(
                            "SELECT data_type FROM information_schema.columns "
                            "WHERE table_name = 'budgets' AND column_name = 'max_budget'"
                        )
                    ).scalar_one()
                    == "double precision"
                )
                connection.execute(text("UPDATE budgets SET max_budget = 25 WHERE budget_id = 'too-big'"))
                connection.commit()
            command.upgrade(config, _COUNTERS_REVISION)
        finally:
            engine.dispose()
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{scratch}" WITH (FORCE)'))
        admin.dispose()
