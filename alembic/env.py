import logging
import os
from logging.config import fileConfig

from alembic import context
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util import CommandError
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Engine, make_url

from gateway.core.database import to_sync_url

# Importing anything from gateway.models registers every model module on this
# metadata (see gateway/models/__init__.py), which is what makes the comparison
# below cover the whole schema rather than the half this file names.
from gateway.models.entities import Base

logger = logging.getLogger("alembic")


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Get database URL from config (if already set programmatically) or the environment.
# Priority: Programmatically set URL -> OTARI_DATABASE_URL -> default
#
# A bare `DATABASE_URL` is deliberately not read. `otari serve` and `otari
# migrate` both take that name as `--database-url` and pass the resolved URL on
# explicitly, so the only invocation this fallback ever served was a bare
# `alembic upgrade head` in a process holding another application's
# `DATABASE_URL`, which aims this chain at that application's database.
database_url = config.get_main_option("sqlalchemy.url") or os.getenv("OTARI_DATABASE_URL") or "sqlite:///./otari.db"
# Normalized here rather than at each call site so every entry point is covered:
# `otari migrate`, auto_migrate on startup, and a bare `alembic upgrade head`
# reading OTARI_DATABASE_URL. Migrations run on a sync engine, so an async URL
# (the form README documents for SQLite) would otherwise fail with MissingGreenlet.
config.set_main_option("sqlalchemy.url", to_sync_url(database_url))

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def _reject_foreign_history(engine: Engine) -> None:
    """Refuse a database whose stamped history is not this chain's.

    Otari stamps the default `alembic_version` table, and so does every other
    Alembic application, so a URL naming someone else's database is accepted as
    readily as otari's own. Alembic then reports only that it cannot locate the
    revision it read, which reads as a corrupt database rather than as the URL
    being wrong, and the next `upgrade` would write otari's schema into that
    database. Naming the target and the foreign revision is what turns it back
    into a configuration error.

    On a connection of its own, so that reading the version table leaves no
    transaction open on the one the migrations run in: Alembic commits its work
    inside `begin_transaction`, which is a no-op when a transaction is already
    under way, and the run would then be rolled back when the connection closes.
    """
    with engine.connect() as connection:
        heads = MigrationContext.configure(connection).get_current_heads()
    if not heads:
        return

    known = {script.revision for script in ScriptDirectory.from_config(config).walk_revisions()}
    foreign = sorted(set(heads) - known)
    if not foreign:
        return

    target = make_url(config.get_main_option("sqlalchemy.url") or "").render_as_string(hide_password=True)
    msg = (
        f"{target} is stamped with alembic revision(s) {', '.join(foreign)}, which are not otari's. "
        "The database holds another application's migration history, or one written by a newer otari "
        "than this one. Point OTARI_DATABASE_URL at otari's own database."
    )
    raise CommandError(msg)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    _reject_foreign_history(connectable)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
