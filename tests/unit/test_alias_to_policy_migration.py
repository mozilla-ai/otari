"""Unit tests for the alias to routing-policy data move (issue #463).

The migration is plain portable SQL over two tables, so it is driven here against
an in-memory SQLite database with ``op.get_bind`` pointed at that connection.
This keeps the *data* behavior testable without a Postgres fixture; the schema
itself is exercised by the normal ``upgrade head`` every integration run does.

The name clash is the case worth pinning. Alias resolution runs before policy
resolution, so a name existing in both stores is not a duplicate to deduplicate:
it is two different answers to "which model does this serve", and the migration
must not pick one silently.
"""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import Connection, create_engine, text

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "b5d7f9a1c3e6_move_stored_aliases_into_routing_policies.py"
)

_SCHEMA = (
    """
    CREATE TABLE model_aliases (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        target TEXT NOT NULL,
        user_id TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE routing_policies (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        spec TEXT NOT NULL,
        user_id TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """,
)


def _load(connection: Connection, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the migration with ``op.get_bind()`` bound to ``connection``."""
    spec = importlib.util.spec_from_file_location("migration_b5d7f9a1c3e6", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.op, "get_bind", lambda: connection)
    return module


@pytest.fixture
def conn() -> Iterator[Connection]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        for statement in _SCHEMA:
            connection.execute(text(statement))
        yield connection


def _add_alias(conn: Connection, name: str, target: str, user_id: str | None = None) -> None:
    conn.execute(
        text(
            "INSERT INTO model_aliases (id, name, target, user_id, created_at, updated_at) "
            "VALUES (:id, :name, :target, :user_id, '2026-01-01', '2026-01-02')"
        ),
        {"id": f"alias-{name}-{user_id}", "name": name, "target": target, "user_id": user_id},
    )


def _add_policy(conn: Connection, name: str, spec: dict[str, object], user_id: str | None = None) -> None:
    conn.execute(
        text(
            "INSERT INTO routing_policies (id, name, spec, user_id, created_at, updated_at) "
            "VALUES (:id, :name, :spec, :user_id, '2026-01-01', '2026-01-02')"
        ),
        {"id": f"policy-{name}-{user_id}", "name": name, "spec": json.dumps(spec), "user_id": user_id},
    )


def _policies(conn: Connection) -> list[dict[str, object]]:
    rows = conn.execute(text("SELECT name, spec, user_id, created_at FROM routing_policies ORDER BY name")).mappings()
    return [dict(row) for row in rows]


def test_an_alias_becomes_its_one_target_policy(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    _add_alias(conn, "fast", "openai:gpt-5-mini")
    _load(conn, monkeypatch).upgrade()

    moved = _policies(conn)
    assert len(moved) == 1
    assert moved[0]["name"] == "fast"
    assert json.loads(str(moved[0]["spec"]))["select"] == [{"default": "openai:gpt-5-mini"}]
    # Moved, not copied: leaving the alias behind would shadow every later edit
    # made through the policy API, because alias resolution runs first.
    assert conn.execute(text("SELECT count(*) FROM model_aliases")).scalar() == 0


def test_scope_and_timestamps_survive_the_move(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    _add_alias(conn, "mine", "openai:gpt-5-mini", user_id="user-1")
    _load(conn, monkeypatch).upgrade()

    moved = _policies(conn)
    assert moved[0]["user_id"] == "user-1"
    assert moved[0]["created_at"] == "2026-01-01"


def test_a_name_in_both_stores_refuses_the_move(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    """The alias wins at request time today, so completing the move would change
    which model the name serves. Neither answer is safe to assume.
    """
    _add_alias(conn, "fast", "openai:gpt-5-mini")
    _add_policy(conn, "fast", {"spec_version": 1, "select": [{"default": "anthropic:claude-sonnet-4-5"}]})

    with pytest.raises(RuntimeError, match="already exists"):
        _load(conn, monkeypatch).upgrade()

    # The alias is still there to resolve by hand.
    assert conn.execute(text("SELECT count(*) FROM model_aliases")).scalar() == 1


def test_the_same_name_in_a_different_scope_is_not_a_clash(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    _add_alias(conn, "fast", "openai:gpt-5-mini")
    _add_policy(conn, "fast", {"spec_version": 1, "select": [{"default": "anthropic:x"}]}, user_id="user-1")
    _load(conn, monkeypatch).upgrade()

    assert {(row["name"], row["user_id"]) for row in _policies(conn)} == {("fast", None), ("fast", "user-1")}


def test_a_one_target_policy_moves_back_on_downgrade(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    _add_alias(conn, "fast", "openai:gpt-5-mini")
    migration = _load(conn, monkeypatch)
    migration.upgrade()
    migration.downgrade()

    aliases = conn.execute(text("SELECT name, target, created_at FROM model_aliases")).mappings().all()
    assert [(row["name"], row["target"], row["created_at"]) for row in aliases] == [
        ("fast", "openai:gpt-5-mini", "2026-01-01")
    ]
    assert _policies(conn) == []


def test_downgrade_leaves_a_policy_an_alias_cannot_represent(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    """A chain or a guardrail has no alias form, so flattening it to its first
    target would quietly drop the behavior the operator configured. A rolled-back
    binary does not read ``routing_policies``, so leaving it is inert.
    """
    _add_policy(
        conn,
        "chained",
        {
            "spec_version": 1,
            "select": [{"default": "openai:gpt-5-mini"}],
            "on_failure": [{"target": "anthropic:claude-sonnet-4-5"}],
            "guardrails": [],
        },
    )
    _load(conn, monkeypatch).downgrade()

    assert conn.execute(text("SELECT count(*) FROM model_aliases")).scalar() == 0
    assert [row["name"] for row in _policies(conn)] == ["chained"]


def test_downgrade_refuses_when_the_alias_name_is_taken(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting the policy without writing its alias would drop its target, and
    the surviving alias may point somewhere else.
    """
    _add_alias(conn, "fast", "openai:gpt-5-mini")
    _add_policy(conn, "fast", {"spec_version": 1, "select": [{"default": "anthropic:claude-sonnet-4-5"}]})

    with pytest.raises(RuntimeError, match="already exists"):
        _load(conn, monkeypatch).downgrade()

    assert [row["name"] for row in _policies(conn)] == ["fast"]
