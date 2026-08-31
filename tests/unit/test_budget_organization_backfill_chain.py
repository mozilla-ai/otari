"""``b7e1c4a9d2f5``'s ownership backfill, on SQLite, over the three populations.

The revision adds ``budgets.organization_id`` and then assigns it, but only in
the one case where assigning grants nobody anything they did not already hold: a
deployment with exactly one organization, whose operator is that organization's
owner. Getting the guard wrong in the permissive direction would hand one
tenant's admins authority over a cap set above them, so the guard is what is
under test here rather than the ADD COLUMN.

SQLite because the OSS base ships it by default and because the revision goes
through ``batch_alter_table`` to add its foreign key, which is the SQLite-only
path and the one where a table rebuild can drop a constraint.
"""

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

import gateway.models  # noqa: F401  (registers every table on the shared metadata)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "b7e1c4a9d2f5"
_BEFORE = "a4d7f1c9e2b6"
_NOW = "2026-08-31 00:00:00"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'budget-owner.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _sole_organization(engine: Engine) -> str:
    """The organization the migration chain itself seeds.

    A real deployment is never at this revision without one: an earlier data
    migration provisions "Default organization", which is exactly why the
    backfill fires on an ordinary upgrade instead of being dead code.
    """
    with engine.begin() as connection:
        # `str(...)` rather than indexing straight into the result: a raw textual
        # query is untyped, so mypy sees `Any` and `--disallow-any-return` refuses
        # it. On SQLite the column is CHAR(32), so this is a narrowing, not a
        # conversion.
        return str(list(connection.execute(text("SELECT id FROM organization")).scalars())[0])


def _add_organization(engine: Engine, *, name: str, slug: str) -> str:
    # `uuid4().hex`, not `str(uuid4())`: SQLAlchemy's `Uuid` stores as CHAR(32)
    # on SQLite, so a dashed string would not compare equal to the seeded row's.
    organization_id = uuid4().hex
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO organization (id, name, slug, created_at, updated_at) "
                "VALUES (:id, :name, :slug, :n, :n)"
            ),
            {"id": organization_id, "name": name, "slug": slug, "n": _NOW},
        )
    return organization_id


def _add_budget(engine: Engine, budget_id: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO budgets (budget_id, name, max_budget, budget_duration_sec, created_at, updated_at) "
                "VALUES (:id, :id, 25.0, 86400, :n, :n)"
            ),
            {"id": budget_id, "n": _NOW},
        )


def _add_gateway_user(engine: Engine, *, user_id: str, budget_id: str | None) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (user_id, spend, reserved, budget_id, blocked, created_at, updated_at, metadata) "
                "VALUES (:user_id, 0, 0, :budget_id, 0, :n, :n, '{}')"
            ),
            {"user_id": user_id, "budget_id": budget_id, "n": _NOW},
        )


def _owners(engine: Engine) -> dict[str, str | None]:
    with engine.begin() as connection:
        return {
            row[0]: row[1] for row in connection.execute(text("SELECT budget_id, organization_id FROM budgets"))
        }


def test_the_column_and_its_index_arrive(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    assert "organization_id" not in {column["name"] for column in inspect(engine).get_columns("budgets")}

    command.upgrade(config, _REVISION)

    assert "organization_id" in {column["name"] for column in inspect(engine).get_columns("budgets")}
    assert "ix_budgets_organization_id" in {index["name"] for index in inspect(engine).get_indexes("budgets")}


def test_the_period_check_survives_the_batch_rebuild(sqlite_before: tuple[Config, Engine]) -> None:
    """The rule worth losing sleep over, for the reason ``test_exact_budget_schema_chain`` gives.

    ``batch_alter_table`` recreates the table on SQLite, and a CHECK is the kind
    of thing a rebuild silently drops. A budget resetting on both a duration and
    a calendar boundary would then be storable and meaningless.
    """
    config, engine = sqlite_before
    command.upgrade(config, _REVISION)

    with pytest.raises(Exception, match="ck_budgets_single_period_source|CHECK constraint"):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO budgets (budget_id, budget_duration_sec, reset_alignment, created_at, updated_at) "
                    "VALUES ('both', 86400, 'calendar_month', :n, :n)"
                ),
                {"n": _NOW},
            )


def test_a_single_organization_deployment_gets_its_budgets(sqlite_before: tuple[Config, Engine]) -> None:
    """The ordinary upgrade, and the one where the operator is already that org's owner."""
    config, engine = sqlite_before
    organization_id = _sole_organization(engine)
    _add_budget(engine, "b1")
    _add_budget(engine, "b2")

    command.upgrade(config, _REVISION)

    assert _owners(engine) == {"b1": organization_id, "b2": organization_id}


def test_a_multi_organization_deployment_is_left_alone(sqlite_before: tuple[Config, Engine]) -> None:
    """The case the guard exists for.

    Assigning here would hand one tenant's admins a cap set above them, and for a
    budget the otari-ai cutover shared across two tenants' ceilings there is no
    single right answer at all. Left NULL, which reads as the deployment's own and
    is invisible to every tenant.
    """
    config, engine = sqlite_before
    # Beside the one the chain seeds, which makes two.
    _add_organization(engine, name="Globex", slug="globex")
    _add_budget(engine, "b1")

    command.upgrade(config, _REVISION)

    assert _owners(engine) == {"b1": None}


def test_a_deployment_with_no_organization_is_left_alone(sqlite_before: tuple[Config, Engine]) -> None:
    """The defensive arm of the same guard, which no chain reaches on its own.

    Asserted anyway because ``len(organizations) != 1`` is one branch covering
    both "too many" and "none", and a version of it that indexed first would
    crash the upgrade on a database whose tenancy seed had been removed.
    """
    config, engine = sqlite_before
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM organization"))
    _add_budget(engine, "b1")

    command.upgrade(config, _REVISION)

    assert _owners(engine) == {"b1": None}


def test_a_budget_handed_to_a_gateway_user_is_skipped(sqlite_before: tuple[Config, Engine]) -> None:
    """``users`` is the deployment's own table, with no tenancy column at all.

    A budget an operator handed to a gateway user is not the organization's to
    redefine, even on a single-organization deployment: moving its figure would
    move what that user may spend, and the page that assigns it is one an admin
    cannot see.
    """
    config, engine = sqlite_before
    organization_id = _sole_organization(engine)
    _add_budget(engine, "shared")
    _add_budget(engine, "tenant-only")
    _add_gateway_user(engine, user_id="u1", budget_id="shared")

    command.upgrade(config, _REVISION)

    assert _owners(engine) == {"shared": None, "tenant-only": organization_id}


def test_a_budget_with_a_reset_record_is_skipped(sqlite_before: tuple[Config, Engine]) -> None:
    """A reset record outlives the assignment that produced it.

    So a budget carrying one was handed to a gateway user even where no live
    ``users`` row still points at it, and the reference goes on refusing a delete
    either way. Skipped for the same reason the live assignment is.
    """
    config, engine = sqlite_before
    organization_id = _sole_organization(engine)
    _add_budget(engine, "had-a-user")
    _add_budget(engine, "never-had-one")
    # The exact shape that makes this distinct from the live-assignment case: the
    # user has since detached, so `users.budget_id` is null and the `users` clause
    # does not exclude the budget, while the reset record still names it.
    # `budget_reset_logs.user_id` is NOT NULL at this revision, so the row needs a
    # user to hang off.
    _add_gateway_user(engine, user_id="detached", budget_id=None)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO budget_reset_logs (user_id, budget_id, previous_spend, reset_at) "
                "VALUES ('detached', 'had-a-user', 1.0, :n)"
            ),
            {"n": _NOW},
        )

    command.upgrade(config, _REVISION)

    assert _owners(engine) == {"had-a-user": None, "never-had-one": organization_id}


def test_the_downgrade_removes_the_column(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    _add_budget(engine, "b1")
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)

    assert "organization_id" not in {column["name"] for column in inspect(engine).get_columns("budgets")}
