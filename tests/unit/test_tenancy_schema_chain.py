"""The tenancy revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default, so the reconciled schema has to migrate
there as well as on PostgreSQL, and this revision has a step that behaves
differently on each: ``user.active_organization_id`` and
``organization.created_by_user_id`` reference each other, so one of the two
foreign keys can only be added after both tables exist. PostgreSQL takes that as
an ``ALTER TABLE``; SQLite has no ``ADD CONSTRAINT``, so Alembic's batch mode
rebuilds the table, which is the step that can silently lose a constraint.

Every integration run migrates PostgreSQL and nothing migrates SQLite, so this
is the only coverage of that path. Driven against a real file database rather
than in-memory because batch mode's rebuild is what is under test.
"""

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_TENANCY_REVISION = "c4b6d8e0f2a3"
_PREVIOUS_REVISION = "b2d4f6a8c0e1"
_TENANCY_TABLES = {"user", "organization", "organization_member", "workspace", "workspace_member"}


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_at_head(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database migrated to head, with its config for further steps."""
    database_url = f"sqlite:///{tmp_path / 'tenancy.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def test_upgrade_creates_every_tenancy_table(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    assert _TENANCY_TABLES <= set(inspect(engine).get_table_names())


def test_circular_foreign_keys_survive_the_batch_rebuild(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Both halves of the cycle are present, pointing at each other."""
    _, engine = sqlite_at_head
    inspector = inspect(engine)

    user_targets = {
        (fk["referred_table"], tuple(fk["constrained_columns"])) for fk in inspector.get_foreign_keys("user")
    }
    organization_targets = {
        (fk["referred_table"], tuple(fk["constrained_columns"])) for fk in inspector.get_foreign_keys("organization")
    }

    assert ("organization", ("active_organization_id",)) in user_targets
    assert ("user", ("created_by_user_id",)) in organization_targets


def test_rebuilt_table_keeps_its_named_unique_constraint(sqlite_at_head: tuple[Config, Engine]) -> None:
    """The rebuild must not degrade ``uq_organization_slug`` into an unnamed index.

    ``copy_from`` in the revision is what guarantees this; without it the
    constraint's identity depends on SQLite reflection.
    """
    _, engine = sqlite_at_head
    constraint_names = {constraint["name"] for constraint in inspect(engine).get_unique_constraints("organization")}
    assert "uq_organization_slug" in constraint_names


def test_email_is_uniquely_indexed_and_nullable(sqlite_at_head: tuple[Config, Engine]) -> None:
    """A local identity has no email, so the column is nullable but still unique."""
    _, engine = sqlite_at_head
    inspector = inspect(engine)

    email_indexes = [index for index in inspector.get_indexes("user") if index["column_names"] == ["email"]]
    assert [index["unique"] for index in email_indexes] == [True]

    email_column = next(column for column in inspector.get_columns("user") if column["name"] == "email")
    assert email_column["nullable"] is True


def test_workspace_activation_classification_is_constrained(sqlite_at_head: tuple[Config, Engine]) -> None:
    """The check constraint travels with the column it guards."""
    _, engine = sqlite_at_head
    checks = inspect(engine).get_check_constraints("workspace")
    assert "check_workspace_activation_classification" in {check["name"] for check in checks}


def test_downgrade_removes_the_tenancy_tables_and_leaves_the_rest(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Rolling back this revision alone must not disturb the gateway's own tables."""
    config, engine = sqlite_at_head

    command.downgrade(config, _PREVIOUS_REVISION)

    remaining = set(inspect(engine).get_table_names())
    assert not (_TENANCY_TABLES & remaining)
    # The gateway's own tables predate this revision and must be untouched.
    assert {"users", "api_keys", "usage_logs"} <= remaining


def test_upgrade_downgrade_upgrade_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    """A downgrade leaves nothing behind for the next upgrade to collide with."""
    config, engine = sqlite_at_head

    command.downgrade(config, _PREVIOUS_REVISION)
    command.upgrade(config, _TENANCY_REVISION)

    assert _TENANCY_TABLES <= set(inspect(engine).get_table_names())


def test_naming_one_model_module_registers_them_all() -> None:
    """``Base.metadata`` is whole however few model modules the caller imported.

    ``alembic/env.py`` names only ``gateway.models.entities`` and relies on the
    package ``__init__`` to pull in the rest. If that import chain breaks, the
    metadata silently loses the tenancy tables and autogenerate proposes
    ``DROP TABLE`` for them, which is data-loss-class and invisible until
    someone runs it. Asserting it needs a fresh interpreter, because by the time
    a test runs in this one every model module is already imported.
    """
    source = (
        "from gateway.models.entities import Base;"
        "import json,sys;"
        "sys.stdout.write(json.dumps(sorted(Base.metadata.tables)))"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=True,
    )

    assert _TENANCY_TABLES <= set(json.loads(result.stdout))


def test_workspace_scope_seeds_a_default_and_backfills_existing_rows(tmp_path: Path) -> None:
    """The workspace column is NOT NULL, so the migration has to supply a value.

    Provisioning is lazy, so a gateway that has only ever served completions has
    no organization and no workspace to backfill onto. The migration seeds them
    under the slug and name provisioning looks up, so a later first boot adopts
    these rather than creating a second default.
    """
    url = f"sqlite:///{tmp_path / 'backfill.db'}"
    config = _alembic_config(url)

    command.upgrade(config, "c4b6d8e0f2a3")
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (user_id, spend, reserved, blocked, created_at, updated_at, metadata) "
                "VALUES ('alice', 0, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '{}')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO api_keys (id, key_hash, key_name, user_id, created_at, is_active, "
                "exclude_from_budget, metadata) "
                "VALUES ('k1', 'h1', 'ada', 'alice', CURRENT_TIMESTAMP, 1, 0, '{}')"
            )
        )
        assert connection.execute(text("SELECT COUNT(*) FROM workspace")).scalar_one() == 0

    command.upgrade(config, "head")

    with engine.begin() as connection:
        seeded = connection.execute(
            text(
                "SELECT w.id FROM workspace w "
                "JOIN organization o ON o.id = w.organization_id WHERE o.slug = 'default'"
            )
        ).scalar_one()
        assert connection.execute(text("SELECT workspace_id FROM api_keys")).scalar_one() == seeded
    engine.dispose()


def test_workspace_scope_keeps_the_partial_indexes_it_rebuilds_around(
    sqlite_at_head: tuple[Config, Engine],
) -> None:
    """Adding the column must not cost the alias and policy partial indexes.

    Both tables carry a ``user_id IS NULL`` partial unique index that SQLite's
    batch mode reflects poorly, which is why the column is added with a default
    in one statement rather than through a table rebuild.
    """
    _, engine = sqlite_at_head
    with engine.begin() as connection:
        partial = {
            name
            for (name,) in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='index' AND sql LIKE '%WHERE%'")
            )
        }
    assert {"uq_model_aliases_global_name", "uq_routing_policies_global_name"} <= partial
