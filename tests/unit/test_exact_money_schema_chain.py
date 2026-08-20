"""The money-column revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default, and this revision retypes eleven columns,
which PostgreSQL takes as ``ALTER TABLE`` and SQLite can only do by rebuilding
the table. A rebuild is where a constraint goes missing: SQLite cannot reflect a
CHECK, so ``organization_model_pricing``'s five non-negative-rate checks survive
only because the revision hands batch mode an explicit ``copy_from``. Nothing
else in the suite migrates SQLite, so this is the only coverage of that path.

The repricing half of the file answers the question the change has to answer
before it can be trusted: **does a row priced before the conversion still price
to the same amount after it?** It is asked over the rates a deployment actually
stores, taken from the genai-prices catalog the default price list is built
from, rather than over numbers chosen to make the arithmetic come out even.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.core.metered_pricing import COST_QUANTUM, calculate_token_cost
from gateway.models.entities import ModelPricing, UsageLog
from gateway.services.pricing_service import default_model_pricing

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_MONEY_REVISION = "a7c3e5d9b1f4"
_BEFORE_MONEY = "b6d8f0a2c4e7"

_RATE_COLUMNS = (
    "input_price_per_million",
    "output_price_per_million",
    "cache_read_price_per_million",
    "cache_write_price_per_million",
    "cache_write_1h_price_per_million",
)
_RATE_CHECKS = {
    "ck_organization_model_pricing_input_non_negative",
    "ck_organization_model_pricing_output_non_negative",
    "ck_organization_model_pricing_cache_read_non_negative",
    "ck_organization_model_pricing_cache_write_non_negative",
    "ck_organization_model_pricing_cache_write_1h_non_negative",
    "ck_organization_model_pricing_period_ordered",
}

# Real models, so the rates under test are the ones an operator's price list
# actually holds after ``otari`` seeds or refreshes it.
_CATALOG_MODELS = (
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("openai", "text-embedding-3-small"),
    ("anthropic", "claude-3-5-sonnet-latest"),
    ("anthropic", "claude-sonnet-4-5"),
    ("google", "gemini-2.5-pro"),
    ("deepseek", "deepseek-chat"),
    ("mistral", "mistral-large-latest"),
)

# One usage shape per convention, covering fresh input, cached input on both
# TTLs, and output.
_USAGE_SHAPES = (
    {"input_tokens": 1_000, "output_tokens": 500, "cache_tokens_included": True},
    {
        "input_tokens": 250_000,
        "output_tokens": 3_137,
        "cache_read_tokens": 199_999,
        "cache_tokens_included": True,
    },
    {
        "input_tokens": 7_919,
        "output_tokens": 1_013,
        "cache_read_tokens": 40_009,
        "cache_write_tokens": 5_003,
        "cache_write_1h_tokens": 1_009,
        "cache_tokens_included": False,
    },
)


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def _catalog_rates() -> list[tuple[str, dict[str, Any]]]:
    """The stored form of each catalog model's rates, as floats, as today's column holds them."""
    as_of = datetime.now(UTC)
    stored: list[tuple[str, dict[str, Any]]] = []
    for provider, model in _CATALOG_MODELS:
        pricing = default_model_pricing(provider, model, as_of)
        assert pricing is not None, f"{provider}:{model} is no longer in the genai-prices catalog"
        rates = {field: getattr(pricing, field) for field in _RATE_COLUMNS}
        stored.append(
            (
                f"{provider}:{model}",
                {
                    # float() is the point: this is how the rate is stored today,
                    # and what the migration has to carry across unchanged.
                    **{field: (None if value is None else float(value)) for field, value in rates.items()},
                    "pricing_tiers": pricing.pricing_tiers or [],
                },
            )
        )
    return stored


def _float_era_cost(rates: dict[str, Any], tiers: list[dict[str, Any]], **usage: Any) -> float:
    """The cost the replaced float implementation would have settled.

    Kept as an independent transcription of the old arithmetic rather than a
    call into anything, so the comparison below is against what the code did and
    not against a rearranged version of what it does now.
    """
    included = usage["cache_tokens_included"]
    prompt = usage["input_tokens"]
    completion = usage["output_tokens"]
    cache_read = usage.get("cache_read_tokens", 0)
    cache_write = usage.get("cache_write_tokens", 0)
    cache_write_1h = min(usage.get("cache_write_1h_tokens", 0), cache_write)
    if included:
        cache_read = min(cache_read, prompt)
        cache_write = min(cache_write, prompt - cache_read)
        cache_write_1h = min(cache_write_1h, cache_write)
        total_input = prompt
    else:
        total_input = prompt + cache_read + cache_write

    effective = dict(rates)
    applicable = [tier for tier in tiers if int(tier.get("min_input_tokens", 0)) <= total_input]
    if applicable:
        winner = max(applicable, key=lambda tier: int(tier["min_input_tokens"]))
        for field in _RATE_COLUMNS:
            if winner.get(field) is not None:
                effective[field] = float(winner[field])

    read_rate = effective["cache_read_price_per_million"]
    write_rate = effective["cache_write_price_per_million"]
    write_1h_rate = effective["cache_write_1h_price_per_million"]
    if write_1h_rate is None:
        write_1h_rate = write_rate

    fresh = total_input
    cost = 0.0
    if read_rate is not None:
        fresh -= cache_read
        cost += cache_read * read_rate / 1_000_000
    if write_rate is not None:
        fresh -= cache_write - cache_write_1h
        cost += (cache_write - cache_write_1h) * write_rate / 1_000_000
    if write_1h_rate is not None:
        fresh -= cache_write_1h
        cost += cache_write_1h * write_1h_rate / 1_000_000
    cost += fresh * effective["input_price_per_million"] / 1_000_000
    cost += completion * effective["output_price_per_million"] / 1_000_000
    return float(cost)


@pytest.fixture
def sqlite_before_money(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database migrated to the revision before the money columns."""
    database_url = f"sqlite:///{tmp_path / 'money.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _BEFORE_MONEY)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _column_types(engine: Engine, table: str) -> dict[str, str]:
    return {column["name"]: str(column["type"]) for column in inspect(engine).get_columns(table)}


def _table_sql(engine: Engine, table: str) -> str:
    with engine.connect() as connection:
        return str(
            connection.execute(
                text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :name"),
                {"name": table},
            ).scalar_one()
        )


def _seed_float_rows(engine: Engine) -> list[tuple[str, dict[str, Any]]]:
    """Write today's float rates, and a priced usage row, into the pre-migration schema."""
    catalog = _catalog_rates()
    now = datetime.now(UTC).isoformat()
    with engine.begin() as connection:
        for model_key, stored in catalog:
            connection.execute(
                text(
                    "INSERT INTO model_pricing (model_key, effective_at, input_price_per_million, "
                    "output_price_per_million, cache_read_price_per_million, cache_write_price_per_million, "
                    "cache_write_1h_price_per_million, pricing_tiers, created_at, updated_at) "
                    "VALUES (:model_key, :now, :input_price_per_million, :output_price_per_million, "
                    ":cache_read_price_per_million, :cache_write_price_per_million, "
                    ":cache_write_1h_price_per_million, '[]', :now, :now)"
                ),
                {
                    "model_key": model_key,
                    "now": now,
                    **{field: stored[field] for field in _RATE_COLUMNS},
                },
            )
        connection.execute(
            text(
                "INSERT INTO usage_logs (id, workspace_id, timestamp, model, endpoint, source, "
                "counts_toward_budget, cost, status) "
                "VALUES ('row-1', :workspace, :now, 'openai:gpt-4o', '/v1/chat/completions', 'gateway', "
                "1, 0.1234567, 'success')"
            ),
            {"workspace": "0" * 32, "now": now},
        )
    return catalog


def test_the_rate_and_cost_columns_become_numeric(sqlite_before_money: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before_money
    _seed_float_rows(engine)
    assert _column_types(engine, "model_pricing")["input_price_per_million"] == "FLOAT"

    command.upgrade(config, _MONEY_REVISION)

    for table in ("model_pricing", "organization_model_pricing"):
        types = _column_types(engine, table)
        for column in _RATE_COLUMNS:
            assert types[column] == "NUMERIC(18, 8)", f"{table}.{column}"
    assert _column_types(engine, "usage_logs")["cost"] == "NUMERIC(18, 6)"


def test_stored_rates_survive_the_conversion_unchanged(sqlite_before_money: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before_money
    catalog = _seed_float_rows(engine)

    command.upgrade(config, _MONEY_REVISION)

    # Read through the ORM, which is what applies the column type: on SQLite the
    # value is still a REAL in the file, and the type is what re-forms it as the
    # decimal that was written.
    with Session(engine) as session:
        rows = {row.model_key: row for row in session.execute(select(ModelPricing)).scalars()}
    for model_key, stored in catalog:
        for column in _RATE_COLUMNS:
            expected = None if stored[column] is None else Decimal(str(stored[column]))
            assert getattr(rows[model_key], column) == expected, f"{model_key}.{column}"


def test_a_settled_cost_survives_the_conversion_rounded_to_the_micro_dollar(
    sqlite_before_money: tuple[Config, Engine],
) -> None:
    """The cost column keeps every digit down to 1e-6 and drops what is below it.

    Which way the dropped digits round is the engine's business for a value that
    was already in the column: on SQLite the row is still a REAL and is read back
    formatted to the column's scale. The half-up rule this change defines governs
    what the gateway *writes* (see ``tests/unit/test_money_columns.py``).
    """
    config, engine = sqlite_before_money
    _seed_float_rows(engine)

    command.upgrade(config, _MONEY_REVISION)

    with Session(engine) as session:
        row = session.get(UsageLog, "row-1")
    assert row is not None
    assert row.cost == Decimal("0.123457")


def test_the_organization_rate_checks_survive_the_rebuild(sqlite_before_money: tuple[Config, Engine]) -> None:
    """SQLite cannot reflect a CHECK, so a rebuild is where one goes missing."""
    config, engine = sqlite_before_money
    before = _table_sql(engine, "organization_model_pricing")
    assert _RATE_CHECKS <= set(before.split())

    command.upgrade(config, _MONEY_REVISION)

    after = _table_sql(engine, "organization_model_pricing")
    assert _RATE_CHECKS <= set(after.split())
    with engine.connect() as connection:
        indexes = {index["name"] for index in inspect(engine).get_indexes("organization_model_pricing")}
        assert "uq_organization_model_pricing_period_start" in indexes
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO organization_model_pricing (id, organization_id, model_key, "
                    "input_price_per_million, output_price_per_million, pricing_tiers, effective_from, "
                    "created_at, updated_at) VALUES ('a', 'b', 'openai:gpt-4o', -1, 1, '[]', "
                    "'2026-01-01', '2026-01-01', '2026-01-01')"
                )
            )


def test_the_conversion_is_reversible(sqlite_before_money: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before_money
    catalog = _seed_float_rows(engine)

    command.upgrade(config, _MONEY_REVISION)
    command.downgrade(config, _BEFORE_MONEY)

    assert _column_types(engine, "model_pricing")["input_price_per_million"] == "FLOAT"
    assert _RATE_CHECKS <= set(_table_sql(engine, "organization_model_pricing").split())
    with engine.connect() as connection:
        rows = {
            key: value
            for key, value in connection.execute(text("SELECT model_key, input_price_per_million FROM model_pricing"))
        }
    for model_key, stored in catalog:
        assert rows[model_key] == stored["input_price_per_million"]


@pytest.mark.parametrize("usage", _USAGE_SHAPES, ids=("plain", "cached_inclusive", "cached_additive"))
def test_stored_rates_reprice_to_the_same_amount(usage: dict[str, Any]) -> None:
    """Every catalog rate prices a request to what the float implementation settled.

    The rounding this change defines is the whole of the difference: the exact
    total is rounded once, half-up, to the micro-dollar, so a repriced row can
    differ from its float-era cost by at most half a micro-dollar.
    """
    tolerance = COST_QUANTUM / 2

    for model_key, stored in _catalog_rates():
        rates = {field: stored[field] for field in _RATE_COLUMNS}
        tiers = stored["pricing_tiers"]
        # The rate row as the column holds it after the migration.
        pricing = type(
            "StoredPricing",
            (),
            {
                **{field: (None if value is None else Decimal(str(value))) for field, value in rates.items()},
                "pricing_tiers": tiers,
            },
        )

        repriced = calculate_token_cost(pricing, **usage)
        before = Decimal(str(_float_era_cost(rates, tiers, **usage)))

        assert abs(repriced - before) <= tolerance, f"{model_key} repriced {before} as {repriced}"
