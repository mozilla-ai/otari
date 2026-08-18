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
from sqlalchemy import Engine, create_engine, inspect

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
