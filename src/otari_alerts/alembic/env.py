"""The plugin's Alembic environment, honoring the contract ``init_db`` documents.

Otari offers the database URL on two channels and this script takes the
attribute one, which is what a contributed chain should do:
``sqlalchemy.url`` is read back through configparser, whose interpolation
treats a percent sign as a token, so a password containing one breaks it.

The version table is the plugin's own, so this chain's history never touches
Otari's ``alembic_version`` row. It is read from
``config.attributes["version_table"]`` when Otari sets it and falls back to the
same constant the :class:`gateway.container.MigrationContribution` declares, so
a bare ``alembic`` run outside the gateway stamps the same table.

``include_object`` is the other half of keeping the two chains apart: an
autogenerate run here connects to a database that also holds Otari's fifty-odd
tables, and without a filter it would emit a revision dropping every one of
them.
"""

from typing import Any

from alembic import context
from sqlalchemy import create_engine, pool

from otari_alerts.models import Base

VERSION_TABLE = "alerts_alembic_version"

config = context.config

_url: object = config.attributes.get("database_url") or config.get_main_option("sqlalchemy.url")
if not isinstance(_url, str) or not _url:
    msg = "No database URL: Otari passes one as config.attributes['database_url']"
    raise RuntimeError(msg)
database_url: str = _url

_table: object = config.attributes.get("version_table")
version_table: str = _table if isinstance(_table, str) and _table else VERSION_TABLE

target_metadata = Base.metadata


def include_object(obj: Any, name: str | None, type_: str, _reflected: bool, _compare_to: Any) -> bool:
    """Let autogenerate see this plugin's own two tables and nothing else.

    Otari's fifty-odd tables share the database, so an unfiltered autogenerate
    would propose dropping every one of them. Alembic excludes its own version
    table on its own, so this filter need not name it.
    """
    if type_ == "table":
        return name in target_metadata.tables
    return True


def run_migrations_online() -> None:
    connectable = create_engine(database_url, poolclass=pool.NullPool)
    try:
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                version_table=version_table,
                include_object=include_object,
                # The two engines the chain runs on render the same DDL, but
                # SQLite cannot ALTER a constraint, so a later revision that
                # changes one needs batch mode to rebuild the table.
                render_as_batch=connection.dialect.name == "sqlite",
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        connectable.dispose()


run_migrations_online()
