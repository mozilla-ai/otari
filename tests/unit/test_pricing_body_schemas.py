"""A pricing body may give its prices as `rates` alone, so its schema requires no per-million field."""

import pytest
from pydantic import BaseModel

from gateway.api.routes.organization_pricing import OrganizationModelPricingCreate, OrganizationModelPricingUpdate
from gateway.api.routes.pricing import SetPricingRequest


@pytest.mark.parametrize("body", [SetPricingRequest, OrganizationModelPricingCreate, OrganizationModelPricingUpdate])
def test_a_body_priced_by_its_rates_needs_no_per_million_field(body: type[BaseModel]) -> None:
    required = body.model_json_schema(mode="validation").get("required", [])

    assert "input_price_per_million" not in required
    assert "output_price_per_million" not in required
