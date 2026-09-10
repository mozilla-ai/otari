"""A contributed chain's ``env.py``, honoring the contract ``init_db`` documents.

Otari hands the URL over as ``sqlalchemy.url`` and the version table as
``config.attributes["version_table"]``; this script passes the table to
``context.configure`` so the chain's history never touches Otari's own row.
"""

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

# Absent means a bare ``alembic`` run outside Otari, where Alembic's default
# stands. Through ``init_db`` the attribute is always set.
version_table = config.attributes.get("version_table")


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None, version_table=version_table)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
