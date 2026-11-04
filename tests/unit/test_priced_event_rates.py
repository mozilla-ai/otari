"""The per-token rates a priced event carries, which a tool call's cost estimate is read with."""

from decimal import Decimal

import pytest

from gateway.core.metered_pricing import billable_usage
from gateway.models.pricing import ModelPricing
from gateway.services.external_usage_service import _rates


@pytest.mark.parametrize(
    ("cache_read", "cache_write", "expected"),
    [
        (Decimal(0), Decimal(0), (Decimal(0), Decimal(0))),
        (None, None, (Decimal(3), Decimal(3))),
        (Decimal("0.3"), Decimal("3.75"), (Decimal("0.3"), Decimal("3.75"))),
    ],
)
def test_a_cache_meter_bills_at_input_only_with_no_rate_of_its_own(
    cache_read: Decimal | None, cache_write: Decimal | None, expected: tuple[Decimal, Decimal]
) -> None:
    pricing = ModelPricing(
        model_key="anthropic:claude-haiku-4-5",
        input_price_per_million=Decimal(3),
        output_price_per_million=Decimal(15),
        cache_read_price_per_million=cache_read,
        cache_write_price_per_million=cache_write,
    )

    rates = _rates(pricing, billable_usage(input_tokens=100, output_tokens=10, cache_tokens_included=True))

    assert (rates["cache_read"], rates["cache_write"]) == expected
