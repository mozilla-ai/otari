"""Unit tests for surfacing the genai-prices default rate on model objects."""

from collections.abc import Iterator

import pytest

from gateway.api.routes.models import ModelObject, ModelPricingInfo, _apply_default_pricing
from gateway.services.pricing_service import configure_default_pricing


@pytest.fixture
def default_pricing_on() -> Iterator[None]:
    configure_default_pricing(True)
    try:
        yield
    finally:
        configure_default_pricing(False)


def test_fills_default_rate_when_enabled(default_pricing_on: None) -> None:
    obj = ModelObject(id="openai:gpt-4o", created=0, owned_by="openai")
    _apply_default_pricing(obj)

    assert obj.pricing is not None
    assert obj.pricing_source == "default"
    assert obj.pricing.input_price_per_million > 0


def test_noop_when_disabled() -> None:
    configure_default_pricing(False)
    obj = ModelObject(id="openai:gpt-4o", created=0, owned_by="openai")
    _apply_default_pricing(obj)

    assert obj.pricing is None
    assert obj.pricing_source == "none"


def test_does_not_override_configured_price(default_pricing_on: None) -> None:
    obj = ModelObject(
        id="openai:gpt-4o",
        created=0,
        owned_by="openai",
        pricing=ModelPricingInfo(input_price_per_million=5.0, output_price_per_million=10.0),
        pricing_source="configured",
    )
    _apply_default_pricing(obj)

    assert obj.pricing is not None
    assert obj.pricing.input_price_per_million == 5.0
    assert obj.pricing_source == "configured"
