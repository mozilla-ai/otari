"""The wire shape of a price row carries its whole cost block."""

from decimal import Decimal
from typing import Any

from gateway.models.pricing import ModelPricing
from gateway.services.merged_catalog_service import pricing_info


def _row(rates: dict[str, Any]) -> ModelPricing:
    return ModelPricing(
        model_key="fal:flux-2",
        input_price_per_million=Decimal(0),
        output_price_per_million=Decimal(40_000),
        unit="image",
        rates=rates,
    )


def test_a_price_carries_its_block_with_its_rules() -> None:
    rates = {
        "image": {
            "output": {
                "price": Decimal("0.04"),
                "unit": "image",
                "overrides": [{"when": {"resolution": {"equals": "2048x2048"}}, "price": Decimal("0.08")}],
            }
        }
    }

    info = pricing_info(_row(rates))

    assert info.rates is not None
    assert info.rates.model_dump(mode="json") == {
        "image": {
            "output": {
                "price": 0.04,
                "unit": "image",
                "overrides": [{"when": {"resolution": {"equals": "2048x2048"}}, "price": 0.08}],
            }
        }
    }


def test_an_unreadable_block_costs_only_its_own_detail() -> None:
    info = pricing_info(_row({"image": {"output": {"price": Decimal(-1), "unit": "image"}}}))

    assert info.rates is None
    assert info.output_price_per_million == 40_000
    assert info.unit == "image"
