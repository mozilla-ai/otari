"""A contributed chain's ``env.py``, honoring the contract ``init_db`` documents.

Otari offers the URL on two channels and this script takes the attribute one,
which is what a real contributed chain does: ``sqlalchemy.url`` is stored in a
configparser and comes back only through interpolation, while the attribute
holds the URL verbatim. The version table comes from
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
    # Passing ``version_table=None`` would override Alembic's default rather
    # than select it, so the keyword is omitted when the attribute is absent.
    configured = {} if version_table is None else {"version_table": version_table}
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None, **configured)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
