"""The revision that sets include_raw_content on organization Tavily search keys, on SQLite."""

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
_REVISION = "8171c59fe077"
_BEFORE = "d4a7b2c9e510"


@pytest.fixture
def before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before this one."""
    url = f"sqlite:///{tmp_path / 'keys.db'}"
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


def _key(engine: Engine, provider: str, *, kind: str = "search", options: dict[str, Any] | None = None) -> str:
    key_id = uuid.uuid4().hex
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO org_web_search_keys (id, organization_id, kind, provider, name, options, "
                "archived_at, is_org_default, created_at, updated_at) "
                "VALUES (:id, :org, :kind, :provider, :name, :options, NULL, 0, :at, :at)"
            ),
            {
                "id": key_id,
                "org": uuid.uuid4().hex,
                "kind": kind,
                "provider": provider,
                "name": f"{provider}-{key_id[:6]}",
                "options": None if options is None else json.dumps(options),
                "at": "2026-11-01 00:00:00.000000",
            },
        )
    return key_id


def _options(engine: Engine, key_id: str) -> Any:
    with engine.connect() as connection:
        value = connection.execute(text("SELECT options FROM org_web_search_keys WHERE id = :id"), {"id": key_id})
        stored = value.scalar_one()
    return None if stored is None else json.loads(stored)


def test_tavily_search_keys_ask_for_page_text_unless_they_set_it(before: tuple[Config, Engine]) -> None:
    config, engine = before
    bare = _key(engine, "tavily")
    with_options = _key(engine, "tavily", options={"topic": "news"})
    turned_off = _key(engine, "tavily", options={"include_raw_content": False})
    brave = _key(engine, "brave")
    fetch = _key(engine, "tavily", kind="fetch")

    command.upgrade(config, _REVISION)

    assert _options(engine, bare) == {"include_raw_content": True}
    assert _options(engine, with_options) == {"topic": "news", "include_raw_content": True}
    assert _options(engine, turned_off) == {"include_raw_content": False}
    assert _options(engine, brave) is None
    assert _options(engine, fetch) is None
