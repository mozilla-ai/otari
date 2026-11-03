"""The revision that drops ``knn`` and ``noop`` router entries from stored policies, on SQLite."""

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "a79a97473478"
_BEFORE = "c4c7c29847fa"
_POOL = ["openai:gpt-5-nano", "openai:gpt-5"]


@pytest.fixture
def before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before this one."""
    url = f"sqlite:///{tmp_path / 'policies.db'}"
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _policy(engine: Engine, name: str, spec: dict[str, Any]) -> None:
    with engine.begin() as connection:
        workspace = connection.execute(text("SELECT id FROM workspace LIMIT 1")).scalar_one()
        connection.execute(
            text(
                "INSERT INTO routing_policies (id, name, spec, workspace_id, created_at, updated_at) "
                "VALUES (:id, :name, :spec, :workspace, '2026-01-01', '2026-01-01')"
            ),
            {"id": str(uuid.uuid4()), "name": name, "spec": json.dumps(spec), "workspace": workspace},
        )


def _upgraded(config: Config, engine: Engine) -> dict[str, dict[str, Any]]:
    command.upgrade(config, _REVISION)
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT name, spec FROM routing_policies")).all()
    return {name: json.loads(spec) for name, spec in rows}


@pytest.mark.parametrize("router", ["knn", "noop", " KNN "])
def test_a_retired_router_entry_is_dropped_and_the_rest_kept(before: tuple[Config, Engine], router: str) -> None:
    config, engine = before
    _policy(
        engine,
        "smart",
        {
            "spec_version": 1,
            "select": [{"router": router, "candidates": _POOL}, {"default": "openai:gpt-5"}],
            "on_failure": ["anthropic:claude-haiku-4-5"],
            "guardrails": [],
        },
    )

    assert _upgraded(config, engine)["smart"] == {
        "spec_version": 1,
        "select": [{"default": "openai:gpt-5"}],
        "on_failure": ["anthropic:claude-haiku-4-5"],
        "guardrails": [],
    }


def test_every_retired_policy_in_one_upgrade_is_rewritten(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _policy(engine, "smart", {"select": [{"router": "knn", "candidates": _POOL}, {"default": "openai:gpt-5"}]})
    _policy(engine, "fast", {"select": [{"router": "noop", "candidates": _POOL}, {"default": "openai:gpt-5-nano"}]})
    _policy(engine, "plain", {"select": [{"default": "openai:gpt-5"}]})

    assert _upgraded(config, engine) == {
        "smart": {"select": [{"default": "openai:gpt-5"}]},
        "fast": {"select": [{"default": "openai:gpt-5-nano"}]},
        "plain": {"select": [{"default": "openai:gpt-5"}]},
    }


def test_policies_on_a_live_router_are_untouched(before: tuple[Config, Engine]) -> None:
    config, engine = before
    weighted = {
        "select": [
            {"router": "weighted", "candidates": _POOL, "weights": {"openai:gpt-5-nano": 1, "openai:gpt-5": 1}},
            {"default": "openai:gpt-5"},
        ]
    }
    plain = {"select": [{"default": "openai:gpt-5"}]}
    _policy(engine, "balanced", weighted)
    _policy(engine, "plain", plain)

    assert _upgraded(config, engine) == {"balanced": weighted, "plain": plain}
