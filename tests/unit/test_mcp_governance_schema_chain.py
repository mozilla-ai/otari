"""The MCP server governance revision's Alembic chain on SQLite: which empty read labels it keeps.

An empty ``read_tools`` becomes null, so it follows the tools the server marks read-only, only on
a server no access rule reaches through ``@read``: there an empty list admits nothing today, and
null would admit more. The PostgreSQL half is ``tests/integration/test_mcp_governance_migration.py``.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "c7e2a4f9d1b8"
_BEFORE = "b3d6f8a2c5e1"

_SERVERS = sa.table(
    "workspace_mcp_servers",
    sa.column("id", sa.Uuid()),
    sa.column("workspace_id", sa.Uuid()),
    sa.column("name", sa.String()),
    sa.column("url", sa.String()),
    sa.column("enabled", sa.Boolean()),
    sa.column("everyone_tools", sa.JSON()),
    sa.column("read_tools", sa.JSON()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)
_GRANTS = sa.table(
    "workspace_mcp_server_grants",
    sa.column("id", sa.Uuid()),
    sa.column("server_id", sa.Uuid()),
    sa.column("user_id", sa.String()),
    sa.column("tools", sa.JSON()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)
_USERS = sa.table(
    "users",
    sa.column("user_id", sa.String()),
    sa.column("spend", sa.Numeric()),
    sa.column("blocked", sa.Boolean()),
    sa.column("metadata", sa.JSON()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)

# Each server's name, its everyone_tools and read_tools, and its one grant's tools (None: no grant).
_CASES: list[tuple[str, list[str], list[str], list[str] | None]] = [
    ("everyone-reads", ["@read"], [], None),
    ("granted-reads", ["*"], [], ["@read", "list_issues"]),
    ("granted-other", ["*"], [], ["list_*"]),
    ("no-rules", ["*"], [], None),
    ("labeled", ["@read"], ["list_issues"], None),
]


def seed_servers(connection: sa.Connection, workspace_id: uuid.UUID) -> None:
    """Store ``_CASES`` as the schema before the revision holds them."""
    now = datetime.now(UTC)
    connection.execute(
        sa.insert(_USERS).values(user_id="grantee", spend=0, blocked=False, metadata={}, created_at=now, updated_at=now)
    )
    for name, everyone, read, granted in _CASES:
        server_id = uuid.uuid4()
        connection.execute(
            sa.insert(_SERVERS).values(
                id=server_id,
                workspace_id=workspace_id,
                name=name,
                url=f"https://93.184.216.34/{name}",
                enabled=True,
                everyone_tools=everyone,
                read_tools=read,
                created_at=now,
                updated_at=now,
            )
        )
        if granted is not None:
            connection.execute(
                sa.insert(_GRANTS).values(
                    id=uuid.uuid4(),
                    server_id=server_id,
                    user_id="grantee",
                    tools=granted,
                    created_at=now,
                    updated_at=now,
                )
            )


def read_labels(connection: sa.Connection) -> dict[str, list[str] | None]:
    rows = connection.execute(sa.select(_SERVERS.c.name, _SERVERS.c.read_tools)).all()
    return {name: read for name, read in rows if name in {case[0] for case in _CASES}}


# Only a server no rule reaches through '@read' starts following the server's own read-only marks.
EXPECTED_AFTER_UPGRADE: dict[str, list[str] | None] = {
    "everyone-reads": [],
    "granted-reads": [],
    "granted-other": None,
    "no-rules": None,
    "labeled": ["list_issues"],
}


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'mcp_governance.db'}"
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, _BEFORE)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def test_empty_read_labels_stay_empty_where_a_rule_uses_read_and_the_chain_round_trips(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before
    with engine.begin() as connection:
        workspace_id = connection.execute(sa.text("SELECT id FROM workspace")).scalars().first()
        seed_servers(connection, uuid.UUID(hex=workspace_id))

    command.upgrade(config, _REVISION)
    with engine.connect() as connection:
        assert read_labels(connection) == EXPECTED_AFTER_UPGRADE
    assert "oauth_client_issuer" in {column["name"] for column in inspect(engine).get_columns("workspace_mcp_servers")}

    command.downgrade(config, _BEFORE)
    with engine.connect() as connection:
        assert read_labels(connection) == {
            name: [] if labels is None else labels for name, labels in EXPECTED_AFTER_UPGRADE.items()
        }
    assert "oauth_client_issuer" not in {
        column["name"] for column in inspect(engine).get_columns("workspace_mcp_servers")
    }

    command.upgrade(config, _REVISION)
    with engine.connect() as connection:
        assert read_labels(connection) == EXPECTED_AFTER_UPGRADE
