"""The per-token rates a priced event carries, which a tool call's cost estimate is read with."""

from datetime import UTC, datetime
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

    usage = billable_usage(input_tokens=100, output_tokens=10, cache_tokens_included=True)

    rates = _rates(pricing, usage, datetime(2026, 11, 2, tzinfo=UTC))

    assert (rates["cache_read"], rates["cache_write"]) == expected


def test_an_imported_event_is_priced_at_the_rules_of_its_own_time() -> None:
    rates = {
        "text": {"input": {"price": 3}, "output": {"price": 15}},
        "multipliers": [{"when": {"date": {"gte": "2027-01-01"}}, "factor": 2}],
    }
    pricing = ModelPricing(
        model_key="anthropic:claude-haiku-4-5",
        input_price_per_million=Decimal(3),
        output_price_per_million=Decimal(15),
        unit="token",
        rates=rates,
    )
    usage = billable_usage(input_tokens=100, output_tokens=10, cache_tokens_included=True)

    before = _rates(pricing, usage, datetime(2026, 12, 31, tzinfo=UTC))
    after = _rates(pricing, usage, datetime(2027, 1, 2, tzinfo=UTC))

    assert (before["input"], after["input"]) == (Decimal(3), Decimal(6))
