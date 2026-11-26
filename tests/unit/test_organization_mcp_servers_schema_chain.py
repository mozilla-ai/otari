"""The organization MCP servers revision's Alembic chain, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half, where every column
change is a batch rebuild: existing servers and grants take their organization and workspace,
the new rules hold, and a downgrade drops what the previous schema cannot hold and nothing else.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "124a7328c1b3"
_BEFORE = "49e3ccf30619"


def _config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'organization_mcp_servers.db'}"
    config = _config(database_url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _insert(connection: Connection, table: str, **values: Any) -> None:  # noqa: ANN401
    columns = ", ".join(values)
    placeholders = ", ".join(f":{name}" for name in values)
    connection.execute(text(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"), values)  # noqa: S608


def _organization(connection: Connection, slug: str) -> str:
    organization_id = uuid.uuid4().hex
    _insert(connection, "organization", id=organization_id, name=slug.title(), slug=slug)
    return organization_id


def _workspace(connection: Connection, organization_id: str, name: str) -> str:
    workspace_id = uuid.uuid4().hex
    _insert(
        connection,
        "workspace",
        id=workspace_id,
        organization_id=organization_id,
        name=name,
        activation_classification="eligible",
    )
    return workspace_id


def _server(connection: Connection, workspace_id: str | None, name: str, organization_id: str | None = None) -> str:
    server_id = uuid.uuid4().hex
    values: dict[str, Any] = {
        "id": server_id,
        "workspace_id": workspace_id,
        "name": name,
        "url": f"https://{name}.example/mcp",
        "enabled": True,
        "created_at": "2026-11-18 00:00:00",
        "updated_at": "2026-11-18 00:00:00",
    }
    if organization_id is not None:
        values["organization_id"] = organization_id
    _insert(connection, "workspace_mcp_servers", **values)
    return server_id


def _user(connection: Connection, user_id: str) -> None:
    """The request-plane user a grant names, which its key requires where SQLite enforces keys."""
    _insert(
        connection,
        "users",
        user_id=user_id,
        spend=0,
        blocked=False,
        metadata="{}",
        created_at="2026-11-18 00:00:00",
        updated_at="2026-11-18 00:00:00",
    )


def _grant(connection: Connection, server_id: str, user_id: str, workspace_id: str | None = None) -> None:
    values: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "server_id": server_id,
        "user_id": user_id,
        "tools": '["*"]',
        "created_at": "2026-11-18 00:00:00",
        "updated_at": "2026-11-18 00:00:00",
    }
    if workspace_id is not None:
        values["workspace_id"] = workspace_id
    _insert(connection, "workspace_mcp_server_grants", **values)


def _scalar(engine: Engine, sql: str, **values: Any) -> Any:  # noqa: ANN401
    with engine.connect() as connection:
        return connection.execute(text(sql), values).scalar()


def test_existing_servers_and_grants_take_their_organization_and_workspace(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before
    with engine.begin() as connection:
        organization_id = _organization(connection, "acme")
        workspace_id = _workspace(connection, organization_id, "Marcomms")
        server_id = _server(connection, workspace_id, "drive")
        _user(connection, "someone")
        _grant(connection, server_id, "someone")

    command.upgrade(config, _REVISION)

    server_organization = _scalar(engine, "SELECT organization_id FROM workspace_mcp_servers")
    grant_workspace = _scalar(engine, "SELECT workspace_id FROM workspace_mcp_server_grants")
    assert uuid.UUID(server_organization) == uuid.UUID(organization_id)
    assert uuid.UUID(grant_workspace) == uuid.UUID(workspace_id)
    servers = {column["name"]: column for column in inspect(engine).get_columns("workspace_mcp_servers")}
    assert servers["workspace_id"]["nullable"] is True
    assert servers["organization_id"]["nullable"] is False
    assert {"workspace_mcp_server_overrides", "mcp_server_aliases"} <= set(inspect(engine).get_table_names())
    indexes = {index["name"] for index in inspect(engine).get_indexes("workspace_mcp_servers")}
    assert {"ix_workspace_mcp_servers_workspace_id", "uq_workspace_mcp_servers_organization_name"} <= indexes
    grant_indexes = {index["name"] for index in inspect(engine).get_indexes("workspace_mcp_server_grants")}
    assert {
        "ix_workspace_mcp_server_grants_api_key_id",
        "ix_workspace_mcp_server_grants_user_id",
        "ix_workspace_mcp_server_grants_workspace_id",
    } <= grant_indexes


def test_an_organization_server_name_is_unique_in_its_organization_only(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before
    command.upgrade(config, _REVISION)
    with engine.begin() as connection:
        acme = _organization(connection, "acme")
        globex = _organization(connection, "globex")
        workspace_id = _workspace(connection, acme, "Marcomms")
        _server(connection, None, "drive", organization_id=acme)
        # Another organization, and a workspace server of the same organization, may use the name.
        _server(connection, None, "drive", organization_id=globex)
        _server(connection, workspace_id, "drive", organization_id=acme)

    with pytest.raises(IntegrityError), engine.begin() as connection:
        _server(connection, None, "drive", organization_id=acme)


def test_a_user_may_hold_one_grant_per_workspace_on_one_server(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    command.upgrade(config, _REVISION)
    with engine.begin() as connection:
        acme = _organization(connection, "acme")
        marcomms = _workspace(connection, acme, "Marcomms")
        sales = _workspace(connection, acme, "Sales")
        server_id = _server(connection, None, "drive", organization_id=acme)
        _user(connection, "someone")
        _grant(connection, server_id, "someone", workspace_id=marcomms)
        _grant(connection, server_id, "someone", workspace_id=sales)

    with pytest.raises(IntegrityError), engine.begin() as connection:
        _grant(connection, server_id, "someone", workspace_id=sales)


def test_the_downgrade_drops_organization_servers_and_keeps_workspace_servers(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before
    command.upgrade(config, _REVISION)
    with engine.begin() as connection:
        acme = _organization(connection, "acme")
        workspace_id = _workspace(connection, acme, "Marcomms")
        own = _server(connection, workspace_id, "drive", organization_id=acme)
        shared = _server(connection, None, "jira", organization_id=acme)
        _user(connection, "someone")
        _grant(connection, own, "someone", workspace_id=workspace_id)
        _grant(connection, shared, "someone", workspace_id=workspace_id)
        _insert(connection, "workspace_mcp_server_logos", server_id=shared, tile="auto", updated_at="2026-11-18")
        _insert(
            connection,
            "workspace_mcp_server_overrides",
            id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            organization_id=acme,
            server_id=shared,
        )
        _insert(
            connection,
            "mcp_server_aliases",
            old_id=uuid.uuid4().hex,
            organization_id=acme,
            workspace_id=workspace_id,
            server_id=shared,
        )

    command.downgrade(config, _BEFORE)

    assert [uuid.UUID(row) for row in _ids(engine, "SELECT id FROM workspace_mcp_servers")] == [uuid.UUID(own)]
    assert [uuid.UUID(row) for row in _ids(engine, "SELECT server_id FROM workspace_mcp_server_grants")] == [
        uuid.UUID(own)
    ]
    assert _scalar(engine, "SELECT COUNT(*) FROM workspace_mcp_server_logos") == 0
    tables = set(inspect(engine).get_table_names())
    assert not {"workspace_mcp_server_overrides", "mcp_server_aliases"} & tables
    assert "organization_id" not in {column["name"] for column in inspect(engine).get_columns("workspace_mcp_servers")}
    assert "workspace_id" not in {
        column["name"] for column in inspect(engine).get_columns("workspace_mcp_server_grants")
    }
    rules = {rule["name"] for rule in inspect(engine).get_unique_constraints("workspace_mcp_server_grants")}
    assert rules == {"uq_workspace_mcp_server_grants_server_key", "uq_workspace_mcp_server_grants_server_user"}

    command.upgrade(config, _REVISION)
    assert uuid.UUID(_scalar(engine, "SELECT organization_id FROM workspace_mcp_servers")) == uuid.UUID(acme)


def _ids(engine: Engine, sql: str) -> list[str]:
    with engine.connect() as connection:
        return [row[0] for row in connection.execute(text(sql))]


@pytest.fixture
def enforcing_foreign_keys() -> Iterator[None]:
    """Every engine opened while this is active enforces foreign keys, as some SQLite builds do."""

    @event.listens_for(Engine, "connect")
    def _enforce(dbapi_connection: Any, _: Any) -> None:  # noqa: ANN401
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    try:
        yield
    finally:
        event.remove(Engine, "connect", _enforce)


def test_rebuilding_the_servers_table_keeps_the_rows_that_name_a_server(
    enforcing_foreign_keys: None, tmp_path: Path
) -> None:
    url = f"sqlite:///{tmp_path / 'enforcing.db'}"
    config = _config(url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            acme = _organization(connection, "acme")
            workspace_id = _workspace(connection, acme, "Marcomms")
            server_id = _server(connection, workspace_id, "drive")
            _insert(connection, "workspace_mcp_server_logos", server_id=server_id, tile="auto", updated_at="2026-11-18")

        command.upgrade(config, _REVISION)

        assert _scalar(engine, "SELECT COUNT(*) FROM workspace_mcp_server_logos") == 1
    finally:
        engine.dispose()
