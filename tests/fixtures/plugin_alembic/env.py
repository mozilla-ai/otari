"""A contributed chain's ``env.py``, honoring the contract ``init_db`` documents.

Otari offers the URL on two channels and this script takes the attribute one,
which is what a real contributed chain does: ``sqlalchemy.url`` is read back
through configparser, whose interpolation treats a percent sign as a token, so
a password containing one breaks it. The version table comes from
``config.attributes["version_table"]`` and is passed to ``context.configure``,
so the chain's history never touches Otari's own row.
"""

from alembic import context
from sqlalchemy import create_engine, pool

config = context.config

database_url = config.attributes["database_url"]
# Absent means a bare ``alembic`` run outside Otari, where Alembic's default
# stands. Through ``init_db`` the attribute is always set.
version_table = config.attributes.get("version_table")


def run_migrations_online() -> None:
    connectable = create_engine(database_url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None, version_table=version_table)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
