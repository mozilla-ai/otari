"""The backfill that records why each hosted model already off is off, on SQLite.

The rows predate the column, so their reason is read from timestamps: a model
offered at a rate has that rate stored before its row, and a row whose first
rate came later, with nothing changing the row since, was held for a price that
an earlier refresh found without switching it on. Each case is a row seeded at
the revision before and read back after the upgrade.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "ae8d42792fd8"
_BEFORE = "6664b766de86"
_OFFERED = datetime(2026, 10, 1, tzinfo=UTC)


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before this one."""
    url = f"sqlite:///{tmp_path / 'reasons.db'}"
    config = _alembic_config(url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _stamp(value: datetime | None) -> str | None:
    return None if value is None else value.strftime("%Y-%m-%d %H:%M:%S.%f")


def _row(
    engine: Engine,
    model: str,
    *,
    enabled: bool = False,
    updated_at: datetime | None = None,
    priced_at: datetime | None = None,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO hosted_provider_models (id, provider, model, enabled, created_at, updated_at) "
                "VALUES (:id, 'fireworks', :model, :enabled, :created_at, :updated_at)"
            ),
            {
                "id": uuid.uuid4().hex,
                "model": model,
                "enabled": enabled,
                "created_at": _stamp(_OFFERED),
                "updated_at": _stamp(updated_at),
            },
        )
        if priced_at is not None:
            connection.execute(
                text(
                    "INSERT INTO model_pricing (model_key, effective_at, input_price_per_million, "
                    "output_price_per_million, pricing_tiers, unit, origin, created_at, updated_at) "
                    "VALUES (:key, :at, 1, 2, '[]', 'tokens', 'seed', :at, :at)"
                ),
                {"key": f"fireworks:{model}", "at": _stamp(priced_at)},
            )


def _reasons(engine: Engine) -> dict[str, str | None]:
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT model, disabled_reason FROM hosted_provider_models")).all()
    return {model: reason for model, reason in rows}


def _served_unpriced(engine: Engine) -> dict[str, bool]:
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT model, serves_unpriced FROM hosted_provider_models")).all()
    return {model: bool(served) for model, served in rows}


def test_each_row_already_off_gets_the_reason_its_timestamps_support(before: tuple[Config, Engine]) -> None:
    config, engine = before
    hour = timedelta(hours=1)
    _row(engine, "on", enabled=True, priced_at=_OFFERED - hour)
    _row(engine, "never-priced")
    # Switched off after the offer by an operator or the health checks, which nothing recorded.
    _row(engine, "never-priced-and-switched-off", updated_at=_OFFERED + hour)
    # The case an earlier refresh left behind: priced after the offer, the switch never touched.
    _row(engine, "priced-by-a-later-refresh", priced_at=_OFFERED + hour)
    # Switched off by the old refresh when its rate went, then priced again.
    _row(engine, "off-then-priced-again", updated_at=_OFFERED + hour, priced_at=_OFFERED + 2 * hour)
    _row(engine, "offered-priced-then-switched-off", updated_at=_OFFERED + hour, priced_at=_OFFERED - hour)
    _row(engine, "switched-after-its-price-came", updated_at=_OFFERED + 2 * hour, priced_at=_OFFERED + hour)

    command.upgrade(config, _REVISION)

    assert _reasons(engine) == {
        "on": None,
        "never-priced": "unpriced",
        "never-priced-and-switched-off": "operator",
        "priced-by-a-later-refresh": "unpriced",
        "off-then-priced-again": "unpriced",
        "offered-priced-then-switched-off": "operator",
        "switched-after-its-price-came": "operator",
    }


def test_the_downgrade_drops_the_column_and_the_upgrade_runs_again(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _row(engine, "never-priced")
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)
    with engine.connect() as connection:
        columns = {row[1] for row in connection.execute(text("PRAGMA table_info(hosted_provider_models)"))}
    assert {"disabled_reason", "serves_unpriced"}.isdisjoint(columns)

    command.upgrade(config, _REVISION)
    assert _reasons(engine) == {"never-priced": "unpriced"}


def test_a_row_on_without_a_rate_is_marked_as_served_unpriced(before: tuple[Config, Engine]) -> None:
    """What a moved provider's models look like: switched on, priced by nothing."""
    config, engine = before
    _row(engine, "moved-and-unpriced", enabled=True)
    _row(engine, "on-and-priced", enabled=True, priced_at=_OFFERED - timedelta(hours=1))
    _row(engine, "off-and-unpriced")

    command.upgrade(config, _REVISION)

    assert _served_unpriced(engine) == {
        "moved-and-unpriced": True,
        "on-and-priced": False,
        "off-and-unpriced": False,
    }
