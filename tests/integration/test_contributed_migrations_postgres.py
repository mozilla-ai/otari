"""A contributed chain runs on PostgreSQL, including the async URL form.

The unit coverage in ``tests/unit/test_contributed_migrations.py`` is SQLite
only, which is how an async-form URL reaching a contributed ``env.py`` as a
driver Alembic cannot run got through CI: the failure needs a real driver to
show up as itself. This runs the same fixture chain against the worker's
PostgreSQL database, on the ``postgresql+asyncpg://`` form an operator may
configure, since Alembic builds a synchronous engine either way.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from gateway.container import MigrationContribution
from gateway.core.database import run_migrations, to_sync_url

_PLUGIN_ALEMBIC = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "plugin_alembic"

PLUGIN = MigrationContribution(
    name="plugin_demo",
    script_location=str(_PLUGIN_ALEMBIC),
    version_table="plugin_demo_alembic_version",
)


@pytest.fixture
def drop_contributed_tables(postgres_url: str) -> Iterator[None]:
    """Leave the worker's schema as the test found it.

    The chain's tables are not in the conftest's reset plan, which snapshots the
    schema once per worker, so nothing else would remove them and every later
    test on this worker would run against a schema that has them.
    """
    yield
    engine = create_engine(to_sync_url(postgres_url))
    try:
        with engine.begin() as connection:
            connection.execute(text("DROP TABLE IF EXISTS plugin_demo"))
            connection.execute(text(f"DROP TABLE IF EXISTS {PLUGIN.version_table}"))
    finally:
        engine.dispose()


def test_a_contributed_chain_runs_on_postgres_from_an_async_url(
    postgres_url: str, clean_database: None, drop_contributed_tables: None
) -> None:
    """Core's chain is already at head here, so what this exercises is the contributed one."""
    async_url = to_sync_url(postgres_url).replace("postgresql://", "postgresql+asyncpg://", 1)

    run_migrations(async_url, (PLUGIN,))

    engine = create_engine(to_sync_url(postgres_url))
    try:
        tables = set(inspect(engine).get_table_names())
        with engine.connect() as connection:
            stamped = [row[0] for row in connection.execute(text(f"SELECT version_num FROM {PLUGIN.version_table}"))]
    finally:
        engine.dispose()

    assert "plugin_demo" in tables
    assert {"alembic_version", PLUGIN.version_table} <= tables
    assert len(stamped) == 1
