"""The revision recording a sign-in's workspace, on SQLite: the column comes and goes, and the chain upgrades again.

The integration suite migrates PostgreSQL only, so this is the SQLite half.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, event, inspect, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "da9ab6d90202"
_BEFORE = "124a7328c1b3"
_TABLE = "mcp_oauth_flows"


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'sign_in_workspace.db'}"
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


def _shape(engine: Engine) -> tuple[set[str], set[str | None], set[str]]:
    schema = inspect(engine)
    return (
        {column["name"] for column in schema.get_columns(_TABLE)},
        {index["name"] for index in schema.get_indexes(_TABLE)},
        {key["referred_table"] for key in schema.get_foreign_keys(_TABLE)},
    )


def test_a_sign_in_records_its_workspace_after_the_revision_and_not_before(
    sqlite_before: tuple[Config, Engine],
) -> None:
    config, engine = sqlite_before
    columns, indexes, keys = _shape(engine)
    assert "workspace_id" not in columns

    command.upgrade(config, _REVISION)
    upgraded = _shape(engine)
    assert "workspace_id" in upgraded[0]
    assert upgraded[1] == indexes | {"ix_mcp_oauth_flows_workspace_id"}
    assert upgraded[2] == keys | {"workspace"}

    command.downgrade(config, _BEFORE)
    assert _shape(engine) == (columns, indexes, keys)

    command.upgrade(config, _REVISION)
    assert "workspace_id" in _shape(engine)[0]


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


def _insert(connection: Connection, table: str, **values: Any) -> None:  # noqa: ANN401
    columns = ", ".join(values)
    placeholders = ", ".join(f":{name}" for name in values)
    connection.execute(text(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"), values)  # noqa: S608


def _flow(connection: Connection, server_id: str, **values: Any) -> str:  # noqa: ANN401
    state_hash = uuid.uuid4().hex * 2
    _insert(
        connection,
        _TABLE,
        state_hash=state_hash,
        server_id=server_id,
        user_id="ada",
        encrypted_code_verifier="v",
        client_id="c",
        client_auth_method="none",
        token_endpoint="https://idp.example/token",
        redirect_uri="https://otari.example/callback",
        server_url="https://jira.example/mcp",
        requires_iss=False,
        for_check=False,
        for_shared=False,
        expires_at="2099-01-01 00:00:00",
        created_at="2026-11-18 00:00:00",
        **values,
    )
    return state_hash


def test_a_sign_in_in_flight_survives_and_one_goes_with_its_workspace(
    enforcing_foreign_keys: None, tmp_path: Path
) -> None:
    url = f"sqlite:///{tmp_path / 'flows.db'}"
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    try:
        organization_id, home, sales, server_id = (uuid.uuid4().hex for _ in range(4))
        with engine.begin() as connection:
            _insert(connection, "organization", id=organization_id, name="Acme", slug="acme")
            for workspace_id, name in ((home, "Default"), (sales, "Sales")):
                _insert(
                    connection,
                    "workspace",
                    id=workspace_id,
                    organization_id=organization_id,
                    name=name,
                    activation_classification="eligible",
                )
            _insert(
                connection,
                "users",
                user_id="ada",
                spend=0,
                blocked=False,
                metadata="{}",
                created_at="2026-11-18 00:00:00",
                updated_at="2026-11-18 00:00:00",
            )
            _insert(
                connection,
                "workspace_mcp_servers",
                id=server_id,
                organization_id=organization_id,
                workspace_id=home,
                name="jira",
                url="https://jira.example/mcp",
                enabled=True,
                created_at="2026-11-18 00:00:00",
                updated_at="2026-11-18 00:00:00",
            )
            legacy = _flow(connection, server_id)

        command.upgrade(config, _REVISION)

        with engine.begin() as connection:
            assert connection.execute(
                text(f"SELECT workspace_id FROM {_TABLE} WHERE state_hash = :hash"),
                {"hash": legacy},  # noqa: S608
            ).one() == (None,)
            started_in_sales = _flow(connection, server_id, workspace_id=sales)
            connection.execute(text("DELETE FROM workspace WHERE id = :id"), {"id": sales})
        with engine.connect() as connection:
            remaining = set(connection.execute(text(f"SELECT state_hash FROM {_TABLE}")).scalars())  # noqa: S608
        assert remaining == {legacy}, "the sign-in started from the deleted workspace went with it"
        assert started_in_sales not in remaining
    finally:
        engine.dispose()
