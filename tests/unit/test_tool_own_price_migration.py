"""The revision that gives each stored search and fetch tool with no price of its own the otari: price, on SQLite."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "9964faffc86b"
_BEFORE = "07802ec50d76"
_EARLIER = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before this one."""
    url = f"sqlite:///{tmp_path / 'prices.db'}"
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


def _price(engine: Engine, kind: str, key: str, rate: str, at: datetime = _EARLIER) -> None:
    stamp = at.strftime("%Y-%m-%d %H:%M:%S.%f")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO model_pricing (kind, model_key, effective_at, input_price_per_million, "
                "output_price_per_million, pricing_tiers, unit, origin, created_at, updated_at) "
                "VALUES (:kind, :key, :at, :rate, 0, '[]', 'requests', 'api', :at, :at)"
            ),
            {"kind": kind, "key": key, "rate": rate, "at": stamp},
        )


def _tool(engine: Engine, name: str, provider: str, kind: str = "search") -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO search_tool_credentials (name, kind, provider, options, created_at, updated_at) "
                "VALUES (:name, :kind, :provider, '{}', :at, :at)"
            ),
            {"name": name, "kind": kind, "provider": provider, "at": _EARLIER.strftime("%Y-%m-%d %H:%M:%S.%f")},
        )


def _upgraded(config: Config, engine: Engine) -> dict[tuple[str, str], list[tuple[Decimal, str]]]:
    """Each search and fetch row after the upgrade, by kind and name: its rates and origins, oldest first."""
    command.upgrade(config, _REVISION)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT kind, model_key, input_price_per_million, origin FROM model_pricing "
                "WHERE kind IN ('search', 'fetch') ORDER BY effective_at"
            )
        ).all()
    prices: dict[tuple[str, str], list[tuple[Decimal, str]]] = {}
    for kind, key, rate, origin in rows:
        prices.setdefault((kind, key), []).append((Decimal(str(rate)), origin))
    return prices


def test_a_fresh_database_gets_builtin_fetch_at_zero(before: tuple[Config, Engine]) -> None:
    assert _upgraded(*before) == {("fetch", "builtin_fetch"): [(Decimal(0), "migration")]}


def test_a_tool_with_no_price_takes_the_otari_price_in_force(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _tool(engine, "mojeek", "mojeek")
    _tool(engine, "tabstack", "tabstack", kind="fetch")
    _price(engine, "model", "otari:web_search", "3000")
    _price(engine, "model", "otari:web_search", "9000", at=datetime.now(UTC) + timedelta(days=30))
    _price(engine, "model", "otari/web_fetch", "1000")

    assert _upgraded(config, engine) == {
        ("search", "mojeek"): [(Decimal(3000), "migration")],
        ("fetch", "tabstack"): [(Decimal(1000), "migration")],
        ("fetch", "builtin_fetch"): [(Decimal(1000), "migration")],
    }


def test_a_tool_priced_under_its_own_row_or_its_old_key_is_left_alone(before: tuple[Config, Engine]) -> None:
    config, engine = before
    for name in ("own", "old", "slash"):
        _tool(engine, name, "exa")
    _price(engine, "search", "own", "2000")
    _price(engine, "model", "exa:old", "2000")
    _price(engine, "model", "exa/slash", "2000")
    _price(engine, "model", "otari:web_search", "3000")

    upgraded = _upgraded(config, engine)

    assert upgraded[("search", "own")] == [(Decimal(2000), "api")]
    assert ("search", "old") not in upgraded
    assert ("search", "slash") not in upgraded


def test_a_tool_priced_only_from_a_later_date_is_priced_until_then(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _tool(engine, "mojeek", "mojeek")
    _price(engine, "search", "mojeek", "2000", at=datetime.now(UTC) + timedelta(days=30))
    _price(engine, "model", "otari:web_search", "3000")

    assert _upgraded(config, engine)[("search", "mojeek")] == [(Decimal(3000), "migration"), (Decimal(2000), "api")]


def test_a_tool_with_nothing_to_copy_stays_unpriced(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _tool(engine, "mojeek", "mojeek")

    assert ("search", "mojeek") not in _upgraded(config, engine)


def test_the_downgrade_deletes_what_the_upgrade_wrote(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _tool(engine, "own", "exa")
    _price(engine, "search", "own", "2000")
    _upgraded(config, engine)

    command.downgrade(config, _BEFORE)

    with engine.connect() as connection:
        rows = connection.execute(text("SELECT model_key, origin FROM model_pricing WHERE kind != 'model'")).all()
    assert [tuple(row) for row in rows] == [("own", "api")]
