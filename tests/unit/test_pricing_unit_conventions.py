"""How a rate in a unit other than tokens reads ``input_price_per_million``, and what it holds before dispatch.

A per-request rate is USD per million requests and an image rate is raw USD per
image. Which meter a rate prices is ``price_usage``'s rule, pinned in
``test_usage_pricing.py``; this file pins the readings the estimates and the
search units rest on.
"""

from decimal import Decimal

import pytest

from gateway.core.metered_pricing import price_request
from gateway.models.pricing import ModelPricing
from gateway.services.budgets import count_estimate, estimate_cost, rerank_estimate
from gateway.services.pricing import billed_search_units, price_usage
from gateway.services.pricing_service import (
    flat_request_cost,
    metered_search_units,
    search_unit_cost,
)


def _pricing(rate: object, *, unit: str = "tokens") -> ModelPricing:
    return ModelPricing(
        model_key="openai:whatever",
        input_price_per_million=rate,
        output_price_per_million=Decimal(1),
        unit=unit,
    )


@pytest.mark.parametrize("rate", [Decimal("0.15"), 0.15, "0.15"])
def test_a_per_request_rate_is_read_the_same_however_the_row_carries_it(rate: object) -> None:
    """A stored row hands back a ``Decimal``; a transient one may not."""
    assert flat_request_cost(_pricing(rate)) == Decimal("0.00000015")
    assert isinstance(flat_request_cost(_pricing(rate)), Decimal)


def test_the_per_request_reading_treats_an_absent_or_unusable_rate_as_free() -> None:
    """Its callers settle an unpriced model at $0 by design."""
    assert flat_request_cost(_pricing(float("nan"))) == Decimal(0)
    assert flat_request_cost(None) == Decimal(0)


def test_a_per_request_endpoint_and_a_per_request_model_write_the_same_charge_line() -> None:
    """Audio and moderations settle through the same line the cost core writes for a completion."""
    pricing = _pricing(Decimal(5000), unit="requests")

    cost, meters, lines = price_request(pricing)
    charge = price_usage(pricing, tokens=None, units={"requests": 1}, legacy_unit_reading=True)

    assert charge is not None
    assert (charge.cost, charge.meters, charge.lines) == (cost, meters, lines)
    assert cost == flat_request_cost(pricing)


def test_the_budget_estimate_for_a_per_request_model_is_its_flat_price() -> None:
    pricing = _pricing(Decimal(5000), unit="requests")

    estimate = estimate_cost(pricing, prompt_chars=40_000, max_output_tokens=None, default_output_tokens=4096)

    assert estimate == Decimal("0.005")


def _per_request_pricing(rate: object = Decimal(2000)) -> ModelPricing:
    return _pricing(rate, unit="requests")


def _per_token_pricing() -> ModelPricing:
    return ModelPricing(
        model_key="voyage:rerank-2.5",
        input_price_per_million=Decimal(50),
        output_price_per_million=Decimal(0),
    )


def test_search_units_bill_at_the_per_request_rate() -> None:
    """Each search unit costs what one request costs, exactly."""
    pricing = _per_request_pricing()

    assert search_unit_cost(1, pricing) == flat_request_cost(pricing) == Decimal("0.002")
    assert search_unit_cost(3, pricing) == Decimal("0.006")


def test_search_units_on_an_unpriced_or_unusable_rate_cost_nothing() -> None:
    assert search_unit_cost(2, None) == Decimal(0)
    assert search_unit_cost(2, _per_request_pricing(float("nan"))) == Decimal(0)
    assert search_unit_cost(-1, _per_request_pricing()) == Decimal(0)


@pytest.mark.parametrize(
    ("documents", "units"),
    [
        # Short documents: one unit per hundred.
        ([10] * 100, 1),
        ([10] * 101, 2),
        ([10] * 250, 3),
        # A document over 500 tokens with the query (2,000 characters) splits into chunks.
        ([4_000] * 50, 1),
        ([4_400] * 50, 2),
        ([100_000], 1),
        # Fifty chunks each: three such documents are 150 chunks, two units.
        ([100_000] * 3, 2),
        ([], 1),
    ],
)
def test_rerank_holds_the_search_units_its_documents_imply(documents: list[int], units: int) -> None:
    pricing = _per_request_pricing()

    held = rerank_estimate(pricing, query_chars=0, document_chars=documents)

    assert held == search_unit_cost(units, pricing)


def test_the_query_counts_toward_every_documents_chunks() -> None:
    """Cohere splits a document whose length with the query passes the chunk size."""
    pricing = _per_request_pricing()

    assert rerank_estimate(pricing, query_chars=400, document_chars=[1_800] * 100) == search_unit_cost(2, pricing)


def test_rerank_on_a_token_rate_holds_its_input_tokens() -> None:
    assert rerank_estimate(_per_token_pricing(), query_chars=1000, document_chars=[1500, 1500]) == estimate_cost(
        _per_token_pricing(), prompt_chars=4000, max_output_tokens=None, default_output_tokens=0
    )


def test_a_count_holds_exactly_what_it_settles_at() -> None:
    images_rate = _pricing(Decimal(40000), unit="images")

    assert count_estimate(images_rate, meter="images", count=3) == Decimal("0.12")
    assert count_estimate(_per_request_pricing(), meter="images", count=3) == Decimal("0.002")
    assert count_estimate(_per_request_pricing(), meter="requests", count=1) == Decimal("0.002")
    # A token rate keeps the legacy reading on these endpoints, so it holds what it settles at.
    assert count_estimate(_pricing(Decimal("0.04")), meter="images", count=3) == Decimal("0.12")


@pytest.mark.parametrize("count", [-5, 10**12])
def test_a_count_estimate_is_held_within_the_reportable_range(count: int) -> None:
    """A caller's ``n`` cannot make the hold raise or go negative; the provider refuses what it cannot serve."""
    assert count_estimate(_pricing(Decimal(40000), unit="images"), meter="images", count=count) >= 0


def test_an_unpriced_count_holds_nothing() -> None:
    assert count_estimate(None, meter="images", count=2) == Decimal(0)
    assert count_estimate(_pricing(Decimal("0.04"), unit="images"), meter="requests", count=1) == Decimal(0)


def test_billed_search_units_round_up_and_floor_at_one() -> None:
    assert billed_search_units({"search_units": 2.1}) == 3
    assert billed_search_units({"search_units": 0}) == 1
    assert billed_search_units(None) == 1
    assert billed_search_units({"search_units": float("nan")}) == 1


def test_metered_search_units_are_reported_only_when_billed_per_unit() -> None:
    assert metered_search_units(_per_request_pricing(), {"search_units": 3}) == 3
    assert metered_search_units(_per_token_pricing(), {"search_units": 3}) is None
    assert metered_search_units(None, {"search_units": 3}) is None
