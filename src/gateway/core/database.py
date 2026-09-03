"""Async database initialization helpers."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from gateway.core.config import GatewayConfig
from gateway.log_config import logger

_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None
_log_engine: AsyncEngine | None = None
_LogSessionLocal: async_sessionmaker[AsyncSession] | None = None

# How long a SQLite connection waits for a held lock before raising
# "database is locked", in milliseconds.
_SQLITE_BUSY_TIMEOUT_MS = 5000

# Async drivers this gateway can be pointed at, mapped to the sync driver
# Alembic needs for the same database. Both are declared dependencies.
_ASYNC_TO_SYNC_DRIVER = {
    "sqlite+aiosqlite": "sqlite",
    "postgresql+asyncpg": "postgresql",
}


def _to_async_url(database_url: str) -> tuple[str, dict[str, Any]]:
    """Convert a sync SQLAlchemy URL into its async equivalent."""

    url: URL = make_url(database_url)
    connect_args: dict[str, Any] = {}
    drivername = url.drivername

    if drivername.startswith("sqlite"):
        async_url = url.set(drivername="sqlite+aiosqlite")
        connect_args["check_same_thread"] = False
        return async_url.render_as_string(hide_password=False), connect_args

    if drivername in {"postgresql", "postgresql+psycopg2"}:
        query = dict(url.query)
        sslmode = query.pop("sslmode", None)
        async_url = url.set(drivername="postgresql+asyncpg", query=query)
        if sslmode:
            connect_args["ssl"] = sslmode
        return async_url.render_as_string(hide_password=False), connect_args

    if drivername == "postgresql+asyncpg":
        return database_url, connect_args

    return database_url, connect_args


def to_sync_url(database_url: str) -> str:
    """Convert an async SQLAlchemy URL into its sync equivalent.

    Alembic builds a synchronous engine, so an async URL reaches it as a driver
    it cannot run and fails with ``MissingGreenlet``. The app engine accepts
    either form (see :func:`_to_async_url`), so the documented
    ``sqlite+aiosqlite://`` URL must not be the one thing that breaks startup.

    A URL that is already synchronous, or whose driver has no async counterpart
    here, is returned unchanged.
    """

    url: URL = make_url(database_url)
    sync_drivername = _ASYNC_TO_SYNC_DRIVER.get(url.drivername)
    if sync_drivername is None:
        return database_url
    return url.set(drivername=sync_drivername).render_as_string(hide_password=False)


def _run_migrations(database_url: str) -> None:
    alembic_cfg = Config()
    alembic_dir = Path(__file__).resolve().parents[3] / "alembic"
    alembic_cfg.set_main_option("script_location", str(alembic_dir))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    alembic_cfg.attributes["configure_logger"] = False
    command.upgrade(alembic_cfg, "head")


def _configure_sqlite_pragmas(engine: AsyncEngine) -> None:
    """Apply per-connection SQLite pragmas.

    ``foreign_keys`` enforces referential integrity. ``journal_mode=WAL`` lets
    readers and the single writer proceed concurrently instead of blocking each
    other, and ``busy_timeout`` makes a connection wait for a held lock rather
    than immediately raising "database is locked". Together the latter two
    remove the lock-contention races that otherwise surface as intermittent HTTP
    500s (e.g. an auth lookup colliding with a key-creation write).
    """
    sync_engine = engine.sync_engine

    @event.listens_for(sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, _: Any) -> None:  # noqa: ANN001, ANN202
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        cursor.close()


# What a database call can raise once a client-side ``command_timeout`` is set.
#
# asyncpg raises a bare ``asyncio.TimeoutError`` (``builtins.TimeoutError`` on
# 3.11+) when a command exceeds it, and SQLAlchemy's asyncpg dialect translates
# only asyncpg's own exception classes, so it would propagate untranslated past
# every ``except SQLAlchemyError`` arm in the codebase. Those arms are what turn
# a database failure into a 503 rather than a 500, release a budget hold rather
# than leaking it, and roll a usage-log write back rather than dropping it.
#
# ``_install_timeout_translation`` closes that for statements. It cannot close
# it for *connect*: SQLAlchemy only wraps DBAPI errors raised while opening a
# connection, so a ``db_connect_timeout`` still surfaces as a plain
# ``TimeoutError``. Handlers on the request path therefore catch both.
DATABASE_ERRORS: tuple[type[BaseException], ...] = (SQLAlchemyError, TimeoutError)


def translate_timeout_error(context: Any) -> None:
    """Re-raise a bare command timeout as a ``SQLAlchemyError``.

    A SQLAlchemy ``handle_error`` listener. Anything that is already a
    ``SQLAlchemyError``, or is not a timeout at all, is left alone: re-wrapping
    would erase the distinction between an ``IntegrityError`` and an
    ``OperationalError`` that callers switch on.
    """
    original = context.original_exception
    if isinstance(original, TimeoutError) and not isinstance(original, SQLAlchemyError):
        raise OperationalError(
            "database statement timed out (db_command_timeout)", None, original
        ) from original


def _install_timeout_translation(engine: AsyncEngine) -> None:
    """Normalize what a statement on *engine* can raise.

    One listener per engine rather than a widened ``except`` at each of the
    ~80 call sites: the exception type a statement can raise is a property of
    how the engine is configured, so the engine is where it is normalized.
    """
    event.listen(engine.sync_engine, "handle_error", translate_timeout_error)


async def release_session(session: AsyncSession | None) -> bool:
    """End *session*'s transaction so its connection goes back to the pool.

    The gateway's request path reads the API key, the billed user and its
    budget, the organization's guardrails and the workspace's tool rows on the
    request-scoped session from :func:`get_db`. That session opens a
    transaction on its first statement and holds its connection until something
    ends it, and ``get_db`` only closes it when the request is over. Called
    before dispatching upstream, this keeps one pooled connection from being
    pinned for the whole provider call, which would make
    ``db_pool_size + db_max_overflow`` a ceiling on concurrent *provider calls*
    rather than on concurrent database work. Past that ceiling a request waits
    ``db_pool_timeout`` and is then refused by the authentication dependency,
    whose database-error arm reports a pool timeout as an authentication
    outage.

    Commits rather than closes, so the session stays usable: settlement and the
    failure-row write that follow dispatch simply check a connection back out.
    Safe to commit because the session factory sets ``expire_on_commit=False``
    and no ORM ``relationship()`` is declared anywhere, so nothing the caller
    already loaded is reloaded afterwards.

    Returns whether the transaction was committed. A failure is not raised: the
    caller is about to reach the provider, and losing a request that passed
    every gate because the connection could not be handed back early is the
    worse outcome. The session is rolled back in that case, because a failed
    commit leaves it unusable and the settlement writes after dispatch would
    otherwise raise ``PendingRollbackError`` on it.

    ``None`` is accepted for hybrid mode, which is handed no session at all.
    """
    if session is None:
        return False
    try:
        await session.commit()
    except DATABASE_ERRORS:
        logger.warning("Could not release the database connection before dispatch", exc_info=True)
        with contextlib.suppress(*DATABASE_ERRORS):
            await session.rollback()
        return False
    return True


def engine_kwargs(
    config: GatewayConfig,
    *,
    connect_args: dict[str, Any],
    is_sqlite: bool,
    pool_size: int | None = None,
    max_overflow: int | None = None,
) -> dict[str, Any]:
    """Build ``create_async_engine`` kwargs for this deployment's database.

    ``connect_args`` is copied, not adopted, so two engines built from the same
    parsed URL cannot rewrite each other's arguments. ``pool_size`` /
    ``max_overflow`` override the configured request-pool sizes for a secondary
    pool; PostgreSQL only, since SQLite runs on ``NullPool``.

    The PostgreSQL arm adds the timeouts that otherwise do not exist anywhere on
    the database path. ``pool_pre_ping`` is what keeps a connection the server
    has closed from being handed to a request, but the ping is itself a
    statement: on a socket that went away without a FIN, which managed
    PostgreSQL and the NAT in front of it do to idle connections routinely, it
    blocks on TCP retransmission rather than failing. With no ``command_timeout``
    that wait is the operating system's to end, minutes later, and every request
    behind it is waiting on a pool slot the whole time. ``pool_recycle`` retires
    connections before they get old enough to be in that state at all.
    """
    # Copied, never mutated in place: two engines are built from one set of
    # parsed URL arguments, and the first has already captured the reference by
    # the time the second is built.
    args = dict(connect_args)
    kwargs: dict[str, Any] = {"pool_pre_ping": True, "connect_args": args}
    if is_sqlite:
        kwargs["poolclass"] = NullPool
        return kwargs

    kwargs["pool_size"] = config.db_pool_size if pool_size is None else pool_size
    kwargs["max_overflow"] = config.db_max_overflow if max_overflow is None else max_overflow
    kwargs["pool_timeout"] = config.db_pool_timeout
    if config.db_pool_recycle >= 0:
        kwargs["pool_recycle"] = config.db_pool_recycle

    args["timeout"] = config.db_connect_timeout
    if config.db_command_timeout > 0:
        args["command_timeout"] = config.db_command_timeout
    if config.db_statement_timeout_ms > 0:
        # Server-side backstop for the client-side ``command_timeout`` above:
        # this one still ends the statement when the client is the wedged half.
        # Configured to fire after it, so the translated client-side error is
        # the one callers normally see.
        server_settings = dict(args.get("server_settings") or {})
        server_settings.setdefault("statement_timeout", str(config.db_statement_timeout_ms))
        args["server_settings"] = server_settings
    return kwargs


def init_db(config: GatewayConfig) -> None:
    """Initialize async database engine and optionally run migrations."""

    global _engine, _SessionLocal, _log_engine, _LogSessionLocal  # noqa: PLW0603

    database_url = config.database_url
    async_url, connect_args = _to_async_url(database_url)

    is_sqlite = async_url.startswith("sqlite+aiosqlite")

    _engine = create_async_engine(
        async_url,
        **engine_kwargs(config, connect_args=connect_args, is_sqlite=is_sqlite),
    )
    _install_timeout_translation(_engine)
    _SessionLocal = async_sessionmaker(_engine, expire_on_commit=False)

    if is_sqlite:
        _configure_sqlite_pragmas(_engine)
        # No second engine on SQLite. It would gain nothing (``NullPool``
        # ignores ``db_log_pool_size``) and would cost something: a second set
        # of connections to the same file, contending for its single writer
        # without the busy timeout and foreign-key pragmas configured above.
        # SQLite is the default for a bare run and for the OSS edition smoke
        # test, so that would be an intermittent "database is locked" on the
        # most common path.
        _log_engine = None
        _LogSessionLocal = _SessionLocal
    else:
        # Usage logging gets a pool of its own. It shares the request pool's
        # database, but not its contention: a saturated request pool used to
        # time the writer out too, and a dropped usage row is unrecoverable.
        # Metering is how spend, budgets and the activity log are reconstructed
        # afterwards, so it must not be the first thing a busy gateway loses.
        # Small and with no overflow, because the writer batches and is never a
        # source of bursts.
        _log_engine = create_async_engine(
            async_url,
            **engine_kwargs(
                config,
                connect_args=connect_args,
                is_sqlite=is_sqlite,
                pool_size=config.db_log_pool_size,
                max_overflow=0,
            ),
        )
        _install_timeout_translation(_log_engine)
        _LogSessionLocal = async_sessionmaker(_log_engine, expire_on_commit=False)

    if config.auto_migrate:
        _run_migrations(database_url)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an AsyncSession."""

    if _SessionLocal is None:
        msg = "Database not initialized. Call init_db() first."
        raise RuntimeError(msg)

    async with _SessionLocal() as session:
        yield session


@asynccontextmanager
async def create_session() -> AsyncIterator[AsyncSession]:
    """Async context manager for creating sessions outside request scope."""

    if _SessionLocal is None:
        msg = "Database not initialized. Call init_db() first."
        raise RuntimeError(msg)

    async with _SessionLocal() as session:
        yield session


@asynccontextmanager
async def create_log_session() -> AsyncIterator[AsyncSession]:
    """Session for the usage-log writer, on the metering pool.

    Falls back to the request pool when :func:`init_db` has not run, which is
    only the case in tests that construct a writer directly.
    """
    factory = _LogSessionLocal or _SessionLocal
    if factory is None:
        msg = "Database not initialized. Call init_db() first."
        raise RuntimeError(msg)

    async with factory() as session:
        yield session


def reset_db() -> None:
    """Dispose the active engine so it can be re-initialized (testing helper)."""

    global _engine, _SessionLocal, _log_engine, _LogSessionLocal  # noqa: PLW0603

    engines = [engine for engine in (_engine, _log_engine) if engine is not None]
    _engine = None
    _SessionLocal = None
    _log_engine = None
    _LogSessionLocal = None

    if not engines:
        return

    async def _dispose_all() -> None:
        for engine in engines:
            await engine.dispose()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_dispose_all())
    else:
        loop.create_task(_dispose_all())


__all__ = [
    "DATABASE_ERRORS",
    "create_log_session",
    "create_session",
    "engine_kwargs",
    "get_db",
    "init_db",
    "release_session",
    "reset_db",
    "translate_timeout_error",
]
