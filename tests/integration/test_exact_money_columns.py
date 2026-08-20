"""Exact money columns on PostgreSQL: the conversion, the round trip, the sum.

The SQLite half of this change is declarative (that engine has no numeric
storage class), so PostgreSQL is where "exact" has to be shown. Three things
matter here and nowhere else:

- the ``double precision`` to ``numeric`` conversion the migration runs does not
  move a stored rate,
- a rate and a settled cost written through the ORM read back as the same
  decimal,
- summing settled costs in SQL is exact, which is what makes ``usage_logs`` an
  accounting truth rather than an approximation of one (mozilla-ai/otari-ai#1751).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlmodel import col

from conftest import seed_workspace_id
from gateway.core.metered_pricing import calculate_metered_cost
from gateway.core.usage import GatewayUsage
from gateway.models.entities import ModelPricing, OrganizationModelPricing, UsageLog
from gateway.models.tenancy import Organization
from gateway.services.pricing_service import default_model_pricing

# Rates an operator's price list actually holds, taken from the catalog the
# default price list is built from rather than invented for the test.
_CATALOG_MODELS = (
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet-latest"),
    ("google", "gemini-2.5-pro"),
    ("deepseek", "deepseek-chat"),
)
_TS = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_MONEY_REVISION = "a7c3e5d9b1f4"
_BEFORE_MONEY = "7c5ba82a601b"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def _catalog_rates() -> list[tuple[str, Decimal]]:
    rates: list[tuple[str, Decimal]] = []
    for provider, model in _CATALOG_MODELS:
        pricing = default_model_pricing(provider, model, _TS)
        assert pricing is not None, f"{provider}:{model} is no longer in the genai-prices catalog"
        for field in ("input_price_per_million", "output_price_per_million", "cache_read_price_per_million"):
            value = getattr(pricing, field)
            if value is not None:
                rates.append((f"{provider}:{model}.{field}", Decimal(str(value))))
    return rates


def test_the_float_to_numeric_conversion_does_not_move_a_stored_rate(test_db: Session) -> None:
    """What the migration's ``USING col::numeric`` does to every rate already stored.

    Run against a scratch table rather than ``model_pricing`` itself, so the
    assertion is about the cast and not about whichever rows a fixture left
    behind.
    """
    rates = _catalog_rates()
    test_db.execute(text("CREATE TEMPORARY TABLE rate_cast (label text, rate double precision)"))
    for label, rate in rates:
        # Bound as a float, which is how the value sat in the column before.
        test_db.execute(
            text("INSERT INTO rate_cast VALUES (:label, :rate)"),
            {"label": label, "rate": float(rate)},
        )

    test_db.execute(text("ALTER TABLE rate_cast ALTER COLUMN rate TYPE numeric(18, 8) USING rate::numeric"))

    converted = {label: rate for label, rate in test_db.execute(text("SELECT label, rate FROM rate_cast"))}
    for label, rate in rates:
        assert converted[label] == rate, label


def test_a_rate_and_a_settled_cost_round_trip_as_the_same_decimal(test_db: Session) -> None:
    pricing = ModelPricing(
        model_key="openai:gpt-4o-mini",
        effective_at=_TS,
        input_price_per_million=0.15,  # a float writer, as config and the API still are
        output_price_per_million=Decimal("0.6"),
        cache_read_price_per_million=Decimal("0.075"),
    )
    test_db.add(pricing)
    test_db.add(
        UsageLog(
            id="settled-1",
            workspace_id=seed_workspace_id(test_db),
            timestamp=_TS,
            model="openai:gpt-4o-mini",
            provider="openai",
            endpoint="/v1/chat/completions",
            status="success",
            prompt_tokens=1_000,
            completion_tokens=500,
            cost=Decimal("0.000450"),
        )
    )
    test_db.commit()
    test_db.expunge_all()

    stored = test_db.execute(select(ModelPricing)).scalars().one()
    settled = test_db.get(UsageLog, "settled-1")

    assert stored.input_price_per_million == Decimal("0.15")
    assert stored.output_price_per_million == Decimal("0.6")
    assert stored.cache_read_price_per_million == Decimal("0.075")
    assert settled is not None
    assert settled.cost == Decimal("0.000450")
    # And the row's own rates reprice its own tokens to its own cost.
    cost, _, _ = calculate_metered_cost(
        stored, GatewayUsage(prompt_tokens=1_000, completion_tokens=500, total_tokens=1_500)
    )
    assert cost == settled.cost


def test_summing_settled_costs_is_exact(test_db: Session) -> None:
    """Ten ten-cent requests are a dollar, which is not what summing floats gives."""
    workspace_id = seed_workspace_id(test_db)
    for index in range(10):
        test_db.add(
            UsageLog(
                id=f"dime-{index}",
                workspace_id=workspace_id,
                timestamp=_TS,
                model="openai:gpt-4o",
                provider="openai",
                endpoint="/v1/chat/completions",
                status="success",
                cost=Decimal("0.1"),
            )
        )
    test_db.commit()

    total = test_db.execute(select(func.sum(UsageLog.cost))).scalar_one()

    assert total == Decimal("1.000000")
    # The same ten additions in binary floating point, which is what the column
    # used to hold. (``sum`` itself is compensated since 3.12, so this is the
    # accumulation a running total actually does.)
    running = 0.0
    for _ in range(10):
        running += 0.1
    assert running != 1.0


@pytest.mark.parametrize(
    ("written", "stored"),
    [
        (Decimal("0.0000005"), Decimal("0.000001")),
        (Decimal("0.0000004"), Decimal("0.000000")),
        (Decimal("1.9999995"), Decimal("2.000000")),
        # A tie where half-up and Python's default half-even disagree: half-even
        # would settle this at 0.000002.
        (Decimal("0.0000025"), Decimal("0.000003")),
    ],
)
def test_a_cost_below_the_column_scale_settles_half_up(
    test_db: Session, written: Decimal, stored: Decimal
) -> None:
    """What lands in the column when an amount has more precision than it holds.

    This pins the rule, not who applies it. PostgreSQL rounds a numeric tie away
    from zero, which is the same answer ``ROUND_HALF_UP`` gives for every value
    including negative ones, so on this engine no amount can distinguish the
    gateway rounding from the engine rounding: the two agree by construction.
    That the *gateway* is the one rounding is shown where the engine does no
    rounding at all, by ``tests/unit/test_money_columns.py`` against the bind
    processor with no database in the path.
    """
    test_db.add(
        UsageLog(
            id=f"rounded-{written}",
            workspace_id=seed_workspace_id(test_db),
            timestamp=_TS,
            model="openai:gpt-4o",
            provider="openai",
            endpoint="/v1/chat/completions",
            status="success",
            cost=written,
        )
    )
    test_db.commit()
    test_db.expunge_all()

    row = test_db.get(UsageLog, f"rounded-{written}")

    assert row is not None
    assert row.cost == stored


def test_the_rate_checks_still_refuse_a_negative_rate_after_the_conversion(test_db: Session) -> None:
    """The constraints have to survive the retype, and mean the same thing after it.

    PostgreSQL keeps a CHECK written against the column's old type, so the
    migration recreates these five against the new one. That is a rewrite of a
    money guard, which is worth an error-path test on the engine that does it
    rather than a reading of the DDL.
    """
    # ``col()`` because ``Organization`` is a SQLModel class; see the backend skill.
    organization_id = test_db.execute(select(col(Organization.id))).scalars().first()
    assert organization_id is not None
    override = {
        "organization_id": organization_id,
        "model_key": "openai:gpt-4o",
        "output_price_per_million": Decimal("1"),
        "pricing_tiers": [],
        "effective_from": _TS,
    }

    test_db.add(OrganizationModelPricing(input_price_per_million=Decimal("-1"), **override))
    with pytest.raises(IntegrityError):
        test_db.flush()
    test_db.rollback()

    test_db.add(OrganizationModelPricing(input_price_per_million=Decimal("0.5"), **override))
    test_db.flush()


def test_the_migration_round_trips_on_postgresql_with_rows_in_the_table(
    test_db: Session, postgres_url: str
) -> None:
    """Downgrade and upgrade again, with data, on the engine CI actually migrates.

    The SQLite half of the chain is covered by
    ``tests/unit/test_exact_money_schema_chain.py``. This is the PostgreSQL half:
    it is the ``ALTER COLUMN ... TYPE ... USING`` path rather than a table
    rebuild, and it is what runs against a real deployment's rows.
    """
    test_db.add(
        ModelPricing(
            model_key="anthropic:claude-3-5-sonnet-latest",
            effective_at=_TS,
            input_price_per_million=Decimal("3"),
            output_price_per_million=Decimal("15"),
            cache_read_price_per_million=Decimal("0.3"),
            cache_write_price_per_million=Decimal("3.75"),
        )
    )
    test_db.add(
        UsageLog(
            id="round-trip",
            workspace_id=seed_workspace_id(test_db),
            timestamp=_TS,
            model="anthropic:claude-3-5-sonnet-latest",
            provider="anthropic",
            endpoint="/v1/chat/completions",
            status="success",
            cost=Decimal("0.123457"),
        )
    )
    test_db.commit()
    # Released before the ALTER: an idle transaction would block it.
    test_db.close()

    config = _alembic_config(postgres_url)
    command.downgrade(config, _BEFORE_MONEY)
    command.upgrade(config, _MONEY_REVISION)

    stored = test_db.execute(select(ModelPricing)).scalars().one()
    settled = test_db.get(UsageLog, "round-trip")

    assert stored.input_price_per_million == Decimal("3")
    assert stored.cache_read_price_per_million == Decimal("0.3")
    assert stored.cache_write_price_per_million == Decimal("3.75")
    assert settled is not None
    assert settled.cost == Decimal("0.123457")
