"""Unit tests for surfacing the genai-prices default rate on model objects."""

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from gateway.api.routes.models import ModelObject, ModelPricingInfo, _alias_model, _apply_default_pricing
from gateway.core.config import GatewayConfig
from gateway.models.entities import ModelPricing
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


def test_prices_from_selector_when_given(default_pricing_on: None) -> None:
    # The alias case: the id is a display name the fallback cannot resolve, so
    # the rate must come from the selector instead.
    obj = ModelObject(id="fast-model", created=0, owned_by="otari")
    _apply_default_pricing(obj, pricing_selector="openai:gpt-4o")

    assert obj.pricing is not None
    assert obj.pricing_source == "default"


# ---------------------------------------------------------------------------
# _alias_model: an alias reports its target's price, and where it came from
# ---------------------------------------------------------------------------

ALIAS_CONFIG = GatewayConfig(aliases={"fast-model": "openai:gpt-4o"})


def _priced(model_key: str, input_rate: float, output_rate: float) -> ModelPricing:
    return ModelPricing(
        model_key=model_key,
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        input_price_per_million=input_rate,
        output_price_per_million=output_rate,
    )


def test_alias_reports_target_db_price_as_configured() -> None:
    obj = _alias_model(ALIAS_CONFIG, "fast-model", {"openai:gpt-4o": _priced("openai:gpt-4o", 2.5, 10.0)})

    assert obj.owned_by == "otari"
    assert obj.pricing is not None
    assert obj.pricing.input_price_per_million == 2.5
    # The price is a real database row; that it reached us via an alias does not
    # make its origin any less "configured".
    assert obj.pricing_source == "configured"


def test_alias_falls_back_to_target_default_price(default_pricing_on: None) -> None:
    # No DB row for the target. The gateway still bills this request at the
    # genai-prices rate, so the catalog has to report that rate rather than
    # claiming the model is unpriced.
    obj = _alias_model(ALIAS_CONFIG, "fast-model", {})

    assert obj.pricing is not None
    assert obj.pricing.input_price_per_million > 0
    assert obj.pricing_source == "default"


def test_alias_never_priced_by_its_display_name(default_pricing_on: None) -> None:
    # "gpt-4o" as a display name would resolve in genai-prices on its own. The
    # alias must be priced from its target, so a lookup keyed on the display name
    # would quietly report the wrong model's rate.
    config = GatewayConfig(aliases={"gpt-4o": "openai:gpt-4o-mini"})
    obj = _alias_model(config, "gpt-4o", {})
    mini = ModelObject(id="openai:gpt-4o-mini", created=0, owned_by="openai")
    _apply_default_pricing(mini)

    assert obj.pricing is not None
    assert mini.pricing is not None
    assert obj.pricing.input_price_per_million == mini.pricing.input_price_per_million


def test_alias_unpriced_when_target_has_no_price() -> None:
    obj = _alias_model(GatewayConfig(aliases={"a": "openai:nonexistent-model-xyz"}), "a", {})

    assert obj.pricing is None
    assert obj.pricing_source == "none"
