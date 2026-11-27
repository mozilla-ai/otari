"""The organization MCP servers revision on PostgreSQL, down and up again with rows in place.

The SQLite half is ``tests/unit/test_organization_mcp_servers_schema_chain.py``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, inspect, text
from sqlalchemy.orm import Session

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "124a7328c1b3"
_BEFORE = "49e3ccf30619"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def _insert(connection: Connection, table: str, **values: Any) -> None:  # noqa: ANN401
    columns = ", ".join(values)
    placeholders = ", ".join(f":{name}" for name in values)
    connection.execute(text(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"), values)  # noqa: S608


def _server(connection: Connection, name: str, **values: Any) -> uuid.UUID:  # noqa: ANN401
    server_id = uuid.uuid4()
    _insert(
        connection,
        "workspace_mcp_servers",
        id=server_id,
        name=name,
        url=f"https://{name}.example/mcp",
        enabled=True,
        created_at="2026-11-18T00:00:00+00:00",
        updated_at="2026-11-18T00:00:00+00:00",
        **values,
    )
    return server_id


def _grant(connection: Connection, server_id: uuid.UUID, user_id: str, **values: Any) -> None:  # noqa: ANN401
    _insert(
        connection,
        "workspace_mcp_server_grants",
        id=uuid.uuid4(),
        server_id=server_id,
        user_id=user_id,
        tools='["*"]',
        created_at="2026-11-18T00:00:00+00:00",
        updated_at="2026-11-18T00:00:00+00:00",
        **values,
    )


def test_the_revision_fills_in_existing_rows_and_its_downgrade_keeps_workspace_servers(
    test_db: Session, postgres_url: str
) -> None:
    engine = test_db.get_bind()
    assert isinstance(engine, Engine)
    # Released before the DDL: an idle transaction would block it.
    test_db.close()
    config = _alembic_config(postgres_url)
    try:
        command.downgrade(config, _BEFORE)
        organization_id, workspace_id = uuid.uuid4(), uuid.uuid4()
        with engine.begin() as connection:
            _insert(connection, "organization", id=organization_id, name="Acme", slug="acme")
            _insert(
                connection,
                "workspace",
                id=workspace_id,
                organization_id=organization_id,
                name="Marcomms",
                activation_classification="eligible",
            )
            _insert(connection, "users", user_id="ada", spend=0, blocked=False, metadata="{}", **_stamps())
            own = _server(connection, "drive", workspace_id=workspace_id)
            _grant(connection, own, "ada")

        command.upgrade(config, _REVISION)

        with engine.begin() as connection:
            assert connection.execute(text("SELECT organization_id FROM workspace_mcp_servers")).scalar_one() == (
                organization_id
            )
            assert connection.execute(text("SELECT workspace_id FROM workspace_mcp_server_grants")).scalar_one() == (
                workspace_id
            )
            shared = _server(connection, "jira", organization_id=organization_id)
            _grant(connection, shared, "ada", workspace_id=workspace_id)
            _insert(
                connection,
                "workspace_mcp_server_overrides",
                id=uuid.uuid4(),
                workspace_id=workspace_id,
                organization_id=organization_id,
                server_id=shared,
            )
            _insert(
                connection,
                "mcp_server_aliases",
                old_id=uuid.uuid4(),
                organization_id=organization_id,
                workspace_id=workspace_id,
                server_id=shared,
            )

        command.downgrade(config, _BEFORE)

        with engine.connect() as connection:
            assert connection.execute(text("SELECT id FROM workspace_mcp_servers")).scalars().all() == [own]
            assert connection.execute(text("SELECT server_id FROM workspace_mcp_server_grants")).scalars().all() == [
                own
            ]
        assert not {"workspace_mcp_server_overrides", "mcp_server_aliases"} & set(inspect(engine).get_table_names())
        assert "organization_id" not in {
            column["name"] for column in inspect(engine).get_columns("workspace_mcp_servers")
        }
    finally:
        # Back at head whatever happened, or every later test on this worker gets a schema its models do not match.
        test_db.close()
        command.upgrade(config, "head")


def _stamps() -> dict[str, str]:
    return {"created_at": "2026-11-18T00:00:00+00:00", "updated_at": "2026-11-18T00:00:00+00:00"}
