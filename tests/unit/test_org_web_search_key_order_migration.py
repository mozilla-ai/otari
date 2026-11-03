"""The revision that keeps each workspace on its organization web search key under the new order, on SQLite.

Each case seeds keys and workspace overrides at the revision before, reads which key every
workspace used under the previous order, upgrades, and checks the new order picks the same.
"""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "90b7e153c41f"
_BEFORE = "a3d5f7b9c1e4"
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


def _stamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")


def _key(
    engine: Engine,
    organization: uuid.UUID,
    *,
    kind: str = "search",
    default: bool = False,
    archived: bool = False,
    age: int = 0,
) -> uuid.UUID:
    key_id = uuid.uuid4()
    created = _stamp(_CREATED + timedelta(minutes=age))
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO org_web_search_keys (id, organization_id, kind, provider, name, encrypted_api_key, "
                "last4, archived_at, is_org_default, created_at, updated_at) "
                "VALUES (:id, :org, :kind, 'exa', :name, 'ciphertext', 'abcd', :archived, :default, :at, :at)"
            ),
            {
                "id": key_id.hex,
                "org": organization.hex,
                "kind": kind,
                "name": f"key-{key_id.hex[:6]}",
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


@dataclass(frozen=True)
class _Key:
    id: uuid.UUID
    default: bool
    pinned: bool
    disabled: bool


def _candidates(engine: Engine, organization: uuid.UUID, workspace: uuid.UUID, kind: str) -> list[_Key]:
    """The organization's live keys of ``kind``, oldest first, with the workspace's flags on each."""
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT k.id, k.is_org_default, o.is_default, o.disabled FROM org_web_search_keys k "
                "LEFT JOIN workspace_web_search_key_overrides o "
                "ON o.org_web_search_key_id = k.id AND o.workspace_id = :workspace "
                "WHERE k.organization_id = :org AND k.kind = :kind AND k.archived_at IS NULL "
                "ORDER BY k.created_at, k.id"
            ),
            {"org": organization.hex, "workspace": workspace.hex, "kind": kind},
        )
        return [_Key(uuid.UUID(str(row[0])), bool(row[1]), bool(row[2]), bool(row[3])) for row in rows]


def _previous_choice(keys: list[_Key]) -> uuid.UUID | None:
    """The pin, else the default, else the oldest key, passing over keys turned off."""
    kept_on = [key for key in keys if not key.disabled]
    ordered = [key for key in kept_on if key.pinned] + [key for key in kept_on if key.default] + kept_on
    return ordered[0].id if ordered else None


def _new_choice(keys: list[_Key]) -> uuid.UUID | None:
    """The pin, else the default, else none, passing over keys turned off."""
    kept_on = [key for key in keys if not key.disabled]
    ordered = [key for key in kept_on if key.pinned] + [key for key in kept_on if key.default]
    return ordered[0].id if ordered else None


def _defaults(engine: Engine, organization: uuid.UUID, kind: str = "search") -> set[uuid.UUID]:
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT id FROM org_web_search_keys WHERE organization_id = :org AND kind = :kind AND is_org_default"),
            {"org": organization.hex, "kind": kind},
        )
        return {uuid.UUID(str(row[0])) for row in rows}


def _pins(engine: Engine, workspace: uuid.UUID) -> set[uuid.UUID]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT org_web_search_key_id FROM workspace_web_search_key_overrides "
                "WHERE workspace_id = :ws AND is_default"
            ),
            {"ws": workspace.hex},
        )
        return {uuid.UUID(str(row[0])) for row in rows}


def _upgrade_keeping_each_choice(
    config: Config, engine: Engine, organization: uuid.UUID, workspaces: list[uuid.UUID], kind: str = "search"
) -> None:
    """Upgrade, and check that each workspace's key is the one it used before."""
    used = {workspace: _previous_choice(_candidates(engine, organization, workspace, kind)) for workspace in workspaces}
    command.upgrade(config, _REVISION)
    assert {
        workspace: _new_choice(_candidates(engine, organization, workspace, kind)) for workspace in workspaces
    } == used


def test_with_no_default_the_oldest_live_key_becomes_the_default(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization, workspace = uuid.uuid4(), uuid.uuid4()
    _key(engine, organization, archived=True)
    oldest = _key(engine, organization, age=1)
    _key(engine, organization, age=2)

    _upgrade_keeping_each_choice(config, engine, organization, [workspace])

    assert _defaults(engine, organization) == {oldest}
    assert _pins(engine, workspace) == set()


def test_an_organization_with_a_default_keeps_it(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization, workspace = uuid.uuid4(), uuid.uuid4()
    _key(engine, organization)
    default = _key(engine, organization, default=True, age=1)

    _upgrade_keeping_each_choice(config, engine, organization, [workspace])

    assert _defaults(engine, organization) == {default}


def test_a_workspace_that_turned_the_default_off_gets_the_key_it_searched_with_pinned(
    before: tuple[Config, Engine],
) -> None:
    config, engine = before
    organization, workspace, other = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    default = _key(engine, organization, default=True)
    first_off = _key(engine, organization, age=1)
    searched_with = _key(engine, organization, age=2)
    _override(engine, workspace, organization, default, disabled=True)
    _override(engine, workspace, organization, first_off, disabled=True)

    _upgrade_keeping_each_choice(config, engine, organization, [workspace, other])

    assert _pins(engine, workspace) == {searched_with}
    assert _pins(engine, other) == set()


def test_a_workspace_that_turned_the_oldest_key_off_gets_the_next_one_pinned(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization, workspace = uuid.uuid4(), uuid.uuid4()
    oldest = _key(engine, organization)
    nxt = _key(engine, organization, age=1)
    _override(engine, workspace, organization, oldest, disabled=True)

    _upgrade_keeping_each_choice(config, engine, organization, [workspace])

    assert _defaults(engine, organization) == {oldest}
    assert _pins(engine, workspace) == {nxt}


def test_a_live_pin_and_a_workspace_with_every_key_off_are_left_alone(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization, pinning, all_off = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    default = _key(engine, organization, default=True)
    pinned = _key(engine, organization, age=1)
    _override(engine, pinning, organization, default, disabled=True)
    _override(engine, pinning, organization, pinned, pinned=True)
    _override(engine, all_off, organization, default, disabled=True)
    _override(engine, all_off, organization, pinned, disabled=True)

    _upgrade_keeping_each_choice(config, engine, organization, [pinning, all_off])

    assert _pins(engine, pinning) == {pinned}
    assert _pins(engine, all_off) == set()


def test_a_pin_left_on_an_archived_key_is_cleared_for_the_new_one(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization, workspace = uuid.uuid4(), uuid.uuid4()
    archived = _key(engine, organization, archived=True)
    default = _key(engine, organization, default=True, age=1)
    searched_with = _key(engine, organization, age=2)
    _override(engine, workspace, organization, archived, pinned=True)
    _override(engine, workspace, organization, default, disabled=True)

    _upgrade_keeping_each_choice(config, engine, organization, [workspace])

    assert _pins(engine, workspace) == {searched_with}


def test_each_kind_keeps_its_own_key(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization, workspace = uuid.uuid4(), uuid.uuid4()
    search_default = _key(engine, organization, default=True)
    fetch_oldest = _key(engine, organization, kind="fetch", age=1)
    _key(engine, organization, kind="fetch", age=2)

    _upgrade_keeping_each_choice(config, engine, organization, [workspace], kind="fetch")

    assert _defaults(engine, organization) == {search_default}
    assert _defaults(engine, organization, kind="fetch") == {fetch_oldest}


def test_a_downgrade_keeps_the_added_defaults_and_pins_and_a_second_upgrade_adds_none(
    before: tuple[Config, Engine],
) -> None:
    config, engine = before
    organization, workspace = uuid.uuid4(), uuid.uuid4()
    oldest = _key(engine, organization)
    nxt = _key(engine, organization, age=1)
    _override(engine, workspace, organization, oldest, disabled=True)
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)
    assert (_defaults(engine, organization), _pins(engine, workspace)) == ({oldest}, {nxt})
    command.upgrade(config, _REVISION)

    assert (_defaults(engine, organization), _pins(engine, workspace)) == ({oldest}, {nxt})
