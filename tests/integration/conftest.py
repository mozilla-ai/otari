import asyncio
import os
import socket
import sys
import threading
import time
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import uvicorn
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if "gateway" in sys.modules:
    del sys.modules["gateway"]

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.db import get_db
from gateway.main import create_app

MODEL_NAME = "gemini:gemini-2.5-flash"


def alembic_config(database_url: str) -> Config:
    """An Alembic config pointed at a test database."""
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent.parent.parent / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@dataclass(frozen=True)
class _ResetPlan:
    """What it takes to put a migrated database back the way the migrations left it."""

    truncate: str
    seeds: tuple[tuple[str, tuple[dict[str, Any], ...]], ...]


_SCHEMA_READY: set[str] = set()
_RESET_PLANS: dict[str, _ResetPlan] = {}
_RESET_ENGINES: dict[str, Engine] = {}


def _reset_engine(database_url: str) -> Engine:
    """One sync engine per database, kept for the whole session.

    Every test resets its database, so the alternative is building and disposing
    an engine (and its connection) once per test for statements that take
    milliseconds. Sync only: an async engine belongs to the event loop that
    opened its connections, and each test runs in a new one.
    """
    engine = _RESET_ENGINES.get(database_url)
    if engine is None:
        engine = create_engine(database_url, pool_pre_ping=True)
        _RESET_ENGINES[database_url] = engine
    return engine


def _build_reset_plan(database_url: str) -> _ResetPlan:
    """Record the schema's tables, and whatever the migrations seeded into them."""
    with _reset_engine(database_url).connect() as conn:
        tables = [name for name in inspect(conn).get_table_names() if name != "alembic_version"]
        seeds = tuple(
            (table, rows)
            for table in tables
            if (rows := tuple(dict(row) for row in conn.execute(text(f'SELECT * FROM "{table}"')).mappings()))
        )
    return _ResetPlan(truncate=", ".join(f'"{table}"' for table in tables), seeds=seeds)


def _run_alembic_migrations(database_url: str) -> None:
    """Build the test schema, once per database per session.

    Every test wants the same schema, and building it is what this suite mostly
    spends its time on. Each xdist worker owns its own database (see
    ``postgres_url``), so "already migrated" is a fact this process can hold on
    to; ``reset_database`` is what hands every test an empty one.

    Still called by the modules that build a client on a config of their own, so
    that a module run by itself gets its schema; the calls after the first are
    the short-circuit.
    """
    if database_url in _SCHEMA_READY:
        return
    command.upgrade(alembic_config(database_url), "head")
    _SCHEMA_READY.add(database_url)
    _RESET_PLANS[database_url] = _build_reset_plan(database_url)


def reset_database(database_url: str) -> None:
    """Empty every table and put the migration-seeded rows back.

    ``CASCADE`` frees TRUNCATE from foreign-key order and ``RESTART IDENTITY``
    puts the sequences back where a freshly built schema would have them, so a
    test cannot tell this from a database created a moment ago.
    """
    plan = _RESET_PLANS[database_url]
    with _reset_engine(database_url).begin() as conn:
        conn.execute(text(f"TRUNCATE {plan.truncate} RESTART IDENTITY CASCADE"))
        for table, rows in plan.seeds:
            columns = list(rows[0])
            column_list = ", ".join(f'"{column}"' for column in columns)
            placeholders = ", ".join(f":{column}" for column in columns)
            conn.execute(
                text(f'INSERT INTO "{table}" ({column_list}) VALUES ({placeholders})'),
                [dict(row) for row in rows],
            )


def _to_async_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg2://"):
        return database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if database_url.startswith("sqlite:///") and not database_url.startswith("sqlite+aiosqlite"):
        return database_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    return database_url


def build_async_session_override(
    database_url: str,
) -> tuple[Callable[[], AsyncGenerator[AsyncSession, None]], Callable[[], None]]:
    """Return an async get_db override for ad-hoc FastAPI apps."""

    async_engine = create_async_engine(_to_async_url(database_url), pool_pre_ping=True)
    async_session_factory = async_sessionmaker(async_engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with async_session_factory() as session:
            yield session

    def dispose() -> None:
        pass

    return override_get_db, dispose


def _worker_database_url(server_url: str) -> Generator[str]:
    """Give this xdist worker a database of its own on ``server_url``.

    The schema outlives each test, so two workers sharing one database would be
    resetting it out from under each other. Under ``-n auto`` without
    ``TEST_DATABASE_URL`` each worker gets its own Postgres container anyway; a
    worker-suffixed database extends the same isolation to a shared server, which
    is the arrangement AGENTS.md points at when Docker is unavailable.

    One suite per server, though: a second one starting up picks the same names
    and drops these databases mid-run.
    """
    worker = os.getenv("PYTEST_XDIST_WORKER", "master")
    url = make_url(server_url)
    database = f"{url.database or 'postgres'}_{worker}"
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{database}"'))
        yield url.set(database=database).render_as_string(hide_password=False)
    finally:
        for engine in _RESET_ENGINES.values():
            engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="session")
def postgres_url() -> Generator[str]:
    """Get PostgreSQL URL from environment or start temporary container."""
    if url := os.getenv("TEST_DATABASE_URL"):
        yield from _worker_database_url(url)
    else:
        postgres = PostgresContainer("postgres:17", username="test", password="test", dbname="test_db")  # noqa: S106
        postgres.start()
        try:
            yield from _worker_database_url(postgres.get_connection_url())
        finally:
            postgres.stop()


@pytest.fixture(autouse=True)
def clean_database(postgres_url: str) -> None:
    """Hand every test the empty, migrated database the suite promises.

    Autouse and at setup rather than teardown, so it also covers the modules that
    build a client on a config of their own, and so a test that dies mid-way
    cannot leave its rows for the next one.
    """
    _run_alembic_migrations(postgres_url)
    reset_database(postgres_url)


@pytest.fixture
def test_db(postgres_url: str, clean_database: None) -> Generator[Session]:
    """Create a test database session."""
    engine = create_engine(postgres_url, pool_pre_ping=True)
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = testing_session_local()

    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest_asyncio.fixture
async def async_db(postgres_url: str, clean_database: None) -> AsyncGenerator[AsyncSession, None]:
    """Create an async test database session."""
    async_engine = create_async_engine(_to_async_url(postgres_url), pool_pre_ping=True)
    async_session_factory = async_sessionmaker(async_engine, expire_on_commit=False)

    try:
        async with async_session_factory() as session:
            yield session
    finally:
        await async_engine.dispose()


@pytest.fixture
def db_session(test_config: GatewayConfig) -> Generator[Session]:
    """Create a standalone DB session for verifying state outside the test client."""
    engine = create_engine(test_config.database_url, pool_pre_ping=True)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def db_session_factory(test_config: GatewayConfig) -> Generator[Callable[[], Session]]:
    """Hand out fresh standalone DB sessions for verifying state outside the client.

    Some tests read DB state at several points around a request, or poll while a
    background writer commits usage logs; each read needs its own session so it
    observes the latest committed rows rather than a cached identity map. This
    owns one engine for the test and centralizes the boilerplate those tests used
    to hand-roll (create_engine / sessionmaker / dispose) at each call site.
    """
    engine = create_engine(test_config.database_url, pool_pre_ping=True)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def make_session() -> Session:
        return factory()

    try:
        yield make_session
    finally:
        engine.dispose()


@pytest.fixture(scope="session")
def test_config(postgres_url: str) -> GatewayConfig:
    """Create a test configuration."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        # The bulk of the suite predates require_pricing and exercises models
        # without configuring pricing. Keep the permissive baseline here; the
        # fail-closed require_pricing=True behavior is covered by dedicated tests
        # that build their own config (see test_require_pricing.py).
        require_pricing=False,
    )


def dispose_async_engine(async_engine: AsyncEngine) -> None:
    """Close an async engine's connections from synchronous teardown."""
    try:
        asyncio.run(async_engine.dispose())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(async_engine.dispose())
        loop.close()


def build_test_client(config: GatewayConfig) -> Generator[TestClient]:
    """Boot an app on the worker's database and hand back a client for it.

    Where a test client that owns its database lifecycle is assembled: the schema
    is already built and ``clean_database`` has already emptied it, which is why
    nothing here migrates or drops. Modules that need a client on a config of
    their own ``yield from`` this rather than restating it. A handful still build
    an app inline through ``build_async_session_override``; those never touched
    the schema, so they were left alone.
    """
    _run_alembic_migrations(config.database_url)
    async_engine = create_async_engine(_to_async_url(config.database_url), pool_pre_ping=True)
    async_session_factory = async_sessionmaker(async_engine, expire_on_commit=False)
    app = create_app(config)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with async_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        dispose_async_engine(async_engine)


@pytest.fixture
def client(test_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    """Create a test client for the FastAPI app."""
    yield from build_test_client(test_config)


@pytest.fixture
def master_key_header(test_config: GatewayConfig) -> dict[str, str]:
    """Return authentication header with master key."""
    header_name = API_KEY_HEADER
    return {header_name: f"Bearer {test_config.master_key}"}


@pytest.fixture
def api_key_obj(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    """Create a test API key and return its details."""
    response = client.post(
        "/v1/keys",
        json={"key_name": "test-key"},
        headers=master_key_header,
    )
    assert response.status_code == 200
    result: dict[str, Any] = response.json()
    return result


@pytest.fixture
def api_key_header(test_config: GatewayConfig, api_key_obj: dict[str, Any]) -> dict[str, str]:
    """Return authentication header with API key."""
    header_name = API_KEY_HEADER
    return {header_name: f"Bearer {api_key_obj['key']}"}


@pytest.fixture
def test_user(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    """Create a test user."""
    response = client.post(
        "/v1/users",
        json={"user_id": "test-user", "alias": "Test User"},
        headers=master_key_header,
    )
    assert response.status_code == 200
    result: dict[str, Any] = response.json()
    return result


@pytest.fixture
def responses_request_body(test_user: dict[str, Any]) -> dict[str, Any]:
    """A minimal /v1/responses request body. Shared so endpoint and
    provider-error-classification tests can both use it."""
    return {
        "model": "openai:gpt-4o-mini",
        "input": {"type": "text", "text": "Hello"},
        "user": test_user["user_id"],
    }


@pytest.fixture
def messages_request_body() -> dict[str, Any]:
    """A minimal /v1/messages request body. Shared so endpoint and
    provider-error-classification tests can both use it."""
    return {
        "model": "anthropic:claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024,
    }


@pytest.fixture
def test_messages() -> list[dict[str, str]]:
    """Return test messages for completion requests."""
    return [{"role": "user", "content": "Say 'hello' and nothing else"}]


@pytest.fixture
def test_messages_with_longer_response() -> list[dict[str, str]]:
    """Return test messages for completion requests with usage."""
    return [{"role": "user", "content": "Tell me a brief story"}]


@pytest.fixture
def model_pricing(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    """Create model pricing for gemini-2.5-flash."""
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": MODEL_NAME,
            "input_price_per_million": 0.075,
            "output_price_per_million": 0.30,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    result: dict[str, Any] = response.json()
    return result


@dataclass
class LiveServer:
    """Holds information about a running test server."""

    url: str
    api_key: str


@pytest.fixture
def live_server(test_config: GatewayConfig, api_key_obj: dict[str, Any]) -> Generator[LiveServer]:
    """Start a live uvicorn server and yield its URL and API key."""
    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    app = create_app(test_config)

    server_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(server_config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)

    try:
        yield LiveServer(url=f"http://127.0.0.1:{port}", api_key=api_key_obj["key"])
    finally:
        server.should_exit = True
        thread.join(timeout=5)
