"""The revision that gives organization web search keys a kind, on SQLite.

Each case seeds keys and workspace overrides at the revision before, upgrades,
and reads them back: every key becomes a search key, an organization keeps one
default, and a workspace keeps searching with the key it searched with.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "dbd76ac9d948"
_BEFORE = "ae8d42792fd8"
_CREATED = datetime(2026, 10, 1, tzinfo=UTC)


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before this one."""
    url = f"sqlite:///{tmp_path / 'keys.db'}"
    config = _alembic_config(url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _stamp(value: datetime | None) -> str | None:
    return None if value is None else value.strftime("%Y-%m-%d %H:%M:%S.%f")


def _key(
    engine: Engine,
    organization: uuid.UUID,
    provider: str,
    *,
    default: bool = False,
    archived: bool = False,
    age: int = 0,
) -> uuid.UUID:
    key_id = uuid.uuid4()
    created = _stamp(_CREATED + timedelta(minutes=age))
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO org_web_search_keys (id, organization_id, provider, name, encrypted_api_key, last4, "
                "archived_at, is_org_default, created_at, updated_at) "
                "VALUES (:id, :org, :provider, :name, 'ciphertext', 'abcd', :archived, :default, :at, :at)"
            ),
            {
                "id": key_id.hex,
                "org": organization.hex,
                "provider": provider,
                "name": f"{provider}-{key_id.hex[:6]}",
                "archived": created if archived else None,
                "default": default,
                "at": created,
            },
        )
    return key_id


def _override(
    engine: Engine,
    workspace: uuid.UUID,
    organization: uuid.UUID,
    key_id: uuid.UUID,
    *,
    pinned: bool = False,
    disabled: bool = False,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO workspace_web_search_key_overrides (id, workspace_id, organization_id, "
                "org_web_search_key_id, is_default, disabled, created_at, updated_at) "
                "VALUES (:id, :workspace, :org, :key, :pinned, :disabled, :at, :at)"
            ),
            {
                "id": uuid.uuid4().hex,
                "workspace": workspace.hex,
                "org": organization.hex,
                "key": key_id.hex,
                "pinned": pinned,
                "disabled": disabled,
                "at": _stamp(_CREATED),
            },
        )


def _defaults(engine: Engine, organization: uuid.UUID) -> set[uuid.UUID]:
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT id FROM org_web_search_keys WHERE organization_id = :org AND is_org_default"),
            {"org": organization.hex},
        )
        return {uuid.UUID(row[0]) for row in rows}


def _pins(engine: Engine, workspace: uuid.UUID) -> set[uuid.UUID]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT org_web_search_key_id FROM workspace_web_search_key_overrides "
                "WHERE workspace_id = :ws AND is_default"
            ),
            {"ws": workspace.hex},
        )
        return {uuid.UUID(row[0]) for row in rows}


def test_every_key_becomes_a_search_key_and_keeps_its_overrides(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization, workspace = uuid.uuid4(), uuid.uuid4()
    tavily = _key(engine, organization, "tavily", default=True)
    _override(engine, workspace, organization, tavily, pinned=True)

    command.upgrade(config, _REVISION)

    with engine.connect() as connection:
        kind, api_base, options = connection.execute(
            text("SELECT kind, api_base, options FROM org_web_search_keys WHERE id = :id"), {"id": tavily.hex}
        ).one()
        overrides = connection.execute(text("SELECT count(*) FROM workspace_web_search_key_overrides")).scalar_one()
    assert (kind, api_base, options) == ("search", None, None)
    assert overrides == 1
    assert _defaults(engine, organization) == {tavily}


def test_the_tavily_default_stays_and_the_brave_one_is_pinned_where_tavily_was_turned_off(
    before: tuple[Config, Engine],
) -> None:
    config, engine = before
    organization = uuid.uuid4()
    brave = _key(engine, organization, "brave", default=True)
    tavily = _key(engine, organization, "tavily", default=True, age=1)
    inherits, tavily_off, both_off, pinned, pinned_archived = (uuid.uuid4() for _ in range(5))
    _override(engine, tavily_off, organization, tavily, disabled=True)
    _override(engine, both_off, organization, tavily, disabled=True)
    _override(engine, both_off, organization, brave, disabled=True)
    other = _key(engine, organization, "tavily", age=2)
    _override(engine, pinned, organization, tavily, disabled=True)
    _override(engine, pinned, organization, other, pinned=True)
    archived = _key(engine, organization, "brave", archived=True, age=3)
    _override(engine, pinned_archived, organization, tavily, disabled=True)
    _override(engine, pinned_archived, organization, archived, pinned=True)

    command.upgrade(config, _REVISION)

    assert _defaults(engine, organization) == {tavily}
    assert _pins(engine, inherits) == set()
    assert _pins(engine, tavily_off) == {brave}
    assert _pins(engine, both_off) == set()
    assert _pins(engine, pinned) == {other}
    assert _pins(engine, pinned_archived) == {brave}


def test_an_organization_with_one_default_is_untouched(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization, workspace = uuid.uuid4(), uuid.uuid4()
    brave = _key(engine, organization, "brave", default=True)
    _key(engine, organization, "tavily")
    _override(engine, workspace, organization, brave, disabled=True)

    command.upgrade(config, _REVISION)

    assert _defaults(engine, organization) == {brave}
    assert _pins(engine, workspace) == set()


def test_after_it_one_default_per_kind_and_a_name_per_kind(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization = uuid.uuid4()
    _key(engine, organization, "exa", default=True)
    command.upgrade(config, _REVISION)

    def insert(kind: str, *, name: str, default: bool) -> None:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO org_web_search_keys (id, organization_id, kind, provider, name, is_org_default, "
                    "created_at, updated_at) VALUES (:id, :org, :kind, 'exa', :name, :default, :at, :at)"
                ),
                {
                    "id": uuid.uuid4().hex,
                    "org": organization.hex,
                    "kind": kind,
                    "name": name,
                    "default": default,
                    "at": _stamp(_CREATED),
                },
            )

    insert("fetch", name="exa-shared", default=True)
    insert("search", name="exa-shared", default=False)
    with pytest.raises(IntegrityError):
        insert("search", name="another", default=True)
    with pytest.raises(IntegrityError):
        insert("fetch", name="exa-shared", default=False)
    with pytest.raises(IntegrityError):
        insert("extract", name="third", default=False)


def test_the_downgrade_deletes_what_the_previous_schema_cannot_hold(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization, workspace = uuid.uuid4(), uuid.uuid4()
    search = _key(engine, organization, "exa", default=True)
    command.upgrade(config, _REVISION)
    fetch, keyless = uuid.uuid4(), uuid.uuid4()
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO org_web_search_keys (id, organization_id, kind, provider, name, encrypted_api_key, "
                "is_org_default, created_at, updated_at) "
                "VALUES (:fetch, :org, 'fetch', 'tabstack', 'pages', 'ciphertext', 1, :at, :at), "
                "(:keyless, :org, 'search', 'searxng', 'ours', NULL, 0, :at, :at)"
            ),
            {"fetch": fetch.hex, "keyless": keyless.hex, "org": organization.hex, "at": _stamp(_CREATED)},
        )
    for key_id in (search, fetch, keyless):
        _override(engine, workspace, organization, key_id, disabled=True)

    command.downgrade(config, _BEFORE)

    with engine.connect() as connection:
        keys = {uuid.UUID(row[0]) for row in connection.execute(text("SELECT id FROM org_web_search_keys"))}
        overridden = {
            uuid.UUID(row[0])
            for row in connection.execute(text("SELECT org_web_search_key_id FROM workspace_web_search_key_overrides"))
        }
    assert keys == {search}
    assert overridden == {search}
    assert _defaults(engine, organization) == {search}
