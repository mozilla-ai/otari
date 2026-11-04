"""A row the offered-models surface seeds carries the community default's cost block."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from gateway.models.pricing import ModelPricing, rates_of
from gateway.services.providers._org_provider_model_service import _apply_default, _seeded_row

_AT = datetime(2026, 6, 1, tzinfo=UTC)
_BLOCK = {
    "text": {"input": {"price": 2.5, "cache": {"read": 1.25}}, "output": {"price": 10}},
    "web_search": {"price": 10},
}


def _default(**given: object) -> ModelPricing:
    return ModelPricing(
        model_key="openai:gpt-4o",
        effective_at=_AT,
        input_price_per_million=Decimal("2.5"),
        output_price_per_million=Decimal(10),
        cache_read_price_per_million=Decimal("1.25"),
        pricing_tiers=[],
        **given,
    )


def test_a_seeded_row_carries_the_default_s_block() -> None:
    row = _seeded_row(uuid.uuid4(), "openai:gpt-4o", _default(rates=_BLOCK), _AT)

    assert row.rates == _BLOCK
    assert row.input_price_per_million == Decimal("2.5")


def test_a_default_with_no_block_seeds_the_block_its_columns_say() -> None:
    default = _default()

    row = _seeded_row(uuid.uuid4(), "openai:gpt-4o", default, _AT)

    assert row.rates == rates_of(default)
    assert row.rates["text"]["input"]["cache"] == {"read": Decimal("1.25")}


def test_a_refresh_moves_the_block_with_the_columns() -> None:
    row = _seeded_row(uuid.uuid4(), "openai:gpt-4o", _default(), _AT)
    moved = _default(rates=_BLOCK)
    moved.input_price_per_million = Decimal(3)

    _apply_default(row, moved)

    assert (row.input_price_per_million, row.rates) == (Decimal(3), _BLOCK)
