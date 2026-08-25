"""The migration chain and the models must describe the same schema.

Nothing else in the suite compares the two, which is how a model and its
migration can disagree and still pass CI. That is not hypothetical: within one
week this repo shipped an `index=True` the migration never created (so a
`create_all` schema and a migrated one differed, and `alembic revision
--autogenerate` would have proposed the missing index) and, in the same edit,
dropped `index=True` from an unrelated CASCADE foreign key whose index the
migration *does* create (so the model silently under-described a live index).
Both directions are drift, both were invisible to every existing test, and both
were caught by a human reading the diff.

The check builds the schema twice against the same PostgreSQL server, once by
running the migration chain and once through ``metadata.create_all``, and
compares what the database itself reports. Comparing the two rendered schemas
rather than diffing source is what makes it catch the second direction too: a
model that forgets a flag looks fine on its own, and only the comparison shows
the index it failed to mention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

import gateway.models  # noqa: F401  # populates the metadata with every model module
from gateway.models.entities import Base

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy.engine import Engine

# Alembic's own bookkeeping table has no model and never will.
_IGNORED_TABLES = frozenset({"alembic_version"})

# The one difference that is not a defect in either direction, kept explicit so
# it stays visible rather than being absorbed by a loose comparison.
#
# ``APIKey.key_hash`` is ``mapped_column(unique=True, index=True)``, which
# SQLAlchemy renders as a single unique index. The initial migration
# (``28d153c22616``) emitted a ``UniqueConstraint`` *and* a unique index for it,
# so a migrated database carries both and enforces the rule twice. Pre-existing
# and harmless, but it is real drift: dropping the redundant constraint needs its
# own migration against an auth-adjacent table, so it is deliberately not done
# here. Remove this entry with that migration.
_KNOWN_MIGRATED_ONLY_INDEXES = frozenset({("api_keys", "api_keys_key_hash_key")})
_KNOWN_MIGRATED_ONLY_UNIQUES = frozenset({("api_keys", "api_keys_key_hash_key")})


def _snapshot(engine: Engine) -> dict[str, dict[str, Any]]:
    """What the database says it holds, as plain comparable values."""
    with engine.connect() as conn:
        inspector = inspect(conn)
        return {
            table: {
                "columns": {
                    column["name"]: (str(column["type"]).upper(), bool(column["nullable"]))
                    for column in inspector.get_columns(table)
                },
                "indexes": {
                    index["name"]: (tuple(index["column_names"]), bool(index["unique"]))
                    for index in inspector.get_indexes(table)
                },
                "uniques": {
                    constraint["name"]: tuple(constraint["column_names"])
                    for constraint in inspector.get_unique_constraints(table)
                },
            }
            for table in inspector.get_table_names()
            if table not in _IGNORED_TABLES
        }


@pytest.fixture
def metadata_built_url(postgres_url: str) -> Generator[str]:
    """A sibling database holding the schema ``create_all`` produces.

    Built beside the worker's migrated database rather than inside it, so the two
    can be inspected independently without either disturbing the other.
    """
    url = make_url(postgres_url)
    parity_database = f"{url.database}_parity"
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{parity_database}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{parity_database}"'))
        parity_url = url.set(database=parity_database).render_as_string(hide_password=False)
        engine = create_engine(parity_url)
        try:
            Base.metadata.create_all(engine)
        finally:
            engine.dispose()
        yield parity_url
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{parity_database}" WITH (FORCE)'))
        admin.dispose()


def test_the_migration_chain_and_the_models_agree(postgres_url: str, metadata_built_url: str) -> None:
    """Every table, column and index is the same whichever way the schema is built.

    A failure here names the drift directly. "migrated only" means the models
    stopped describing something the chain creates; "models only" means a
    migration is missing, and ``alembic revision --autogenerate`` would propose it.
    """
    migrated_engine = create_engine(postgres_url)
    metadata_engine = create_engine(metadata_built_url)
    try:
        migrated = _snapshot(migrated_engine)
        from_models = _snapshot(metadata_engine)
    finally:
        migrated_engine.dispose()
        metadata_engine.dispose()

    assert sorted(migrated) == sorted(from_models), (
        f"tables only in the migrated schema: {sorted(set(migrated) - set(from_models))}; "
        f"tables only in the models: {sorted(set(from_models) - set(migrated))}"
    )

    for table in sorted(migrated):
        assert migrated[table]["columns"] == from_models[table]["columns"], (
            f"{table}: columns disagree between the migration chain and the models"
        )

        migrated_indexes = dict(migrated[table]["indexes"])
        for ignored_table, ignored_index in _KNOWN_MIGRATED_ONLY_INDEXES:
            if ignored_table == table:
                migrated_indexes.pop(ignored_index, None)
        assert migrated_indexes == from_models[table]["indexes"], (
            f"{table}: indexes disagree. Migrated only: "
            f"{sorted(set(migrated_indexes) - set(from_models[table]['indexes']))}; models only: "
            f"{sorted(set(from_models[table]['indexes']) - set(migrated_indexes))}"
        )

        migrated_uniques = dict(migrated[table]["uniques"])
        for ignored_table, ignored_unique in _KNOWN_MIGRATED_ONLY_UNIQUES:
            if ignored_table == table:
                migrated_uniques.pop(ignored_unique, None)
        assert migrated_uniques == from_models[table]["uniques"], (
            f"{table}: unique constraints disagree between the migration chain and the models"
        )
