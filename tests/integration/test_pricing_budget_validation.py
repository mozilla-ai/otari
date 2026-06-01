"""Tests for pricing and budget value validation."""

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import PricingConfig
from gateway.services.pricing_service import pricing_required_but_missing


def test_negative_pricing_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that negative input pricing is rejected."""
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": -1.0,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_negative_output_pricing_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that negative output pricing is rejected."""
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": -5.0,
        },
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_zero_pricing_accepted(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that zero pricing is accepted (free models)."""
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 0.0,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200


def test_negative_budget_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that negative max_budget is rejected."""
    response = client.post(
        "/v1/budgets",
        json={"max_budget": -100.0},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_zero_duration_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that zero budget_duration_sec is rejected."""
    response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0, "budget_duration_sec": 0},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_negative_duration_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that negative budget_duration_sec is rejected."""
    response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0, "budget_duration_sec": -86400},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_valid_budget_accepted(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that valid budget values are accepted."""
    response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0, "budget_duration_sec": 86400},
        headers=master_key_header,
    )
    assert response.status_code == 200


def test_positive_pricing_accepted(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that positive pricing values are accepted."""
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["input_price_per_million"] == 2.5
    assert data["output_price_per_million"] == 10.0


def test_update_budget_negative_max_budget_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that updating a budget with negative max_budget is rejected."""
    create_response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0},
        headers=master_key_header,
    )
    budget_id = create_response.json()["budget_id"]

    response = client.patch(
        f"/v1/budgets/{budget_id}",
        json={"max_budget": -50.0},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_update_budget_zero_duration_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that updating a budget with zero budget_duration_sec is rejected."""
    create_response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0},
        headers=master_key_header,
    )
    budget_id = create_response.json()["budget_id"]

    response = client.patch(
        f"/v1/budgets/{budget_id}",
        json={"budget_duration_sec": 0},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_update_budget_negative_duration_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that updating a budget with negative budget_duration_sec is rejected."""
    create_response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0},
        headers=master_key_header,
    )
    budget_id = create_response.json()["budget_id"]

    response = client.patch(
        f"/v1/budgets/{budget_id}",
        json={"budget_duration_sec": -86400},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_pricing_config_rejects_negative_input_price() -> None:
    """Test that PricingConfig rejects negative input_price_per_million."""
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        PricingConfig(input_price_per_million=-1.0, output_price_per_million=10.0)


def test_pricing_config_rejects_negative_output_price() -> None:
    """Test that PricingConfig rejects negative output_price_per_million."""
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        PricingConfig(input_price_per_million=10.0, output_price_per_million=-1.0)


def test_pricing_config_accepts_zero_prices() -> None:
    """Test that PricingConfig accepts zero prices for free models."""
    config = PricingConfig(input_price_per_million=0.0, output_price_per_million=0.0)
    assert config.input_price_per_million == 0.0
    assert config.output_price_per_million == 0.0


def test_pricing_required_but_missing_true_when_unpriced_and_required() -> None:
    """With require_pricing on, a missing pricing row must reject (F3 budget-bypass fix)."""
    assert pricing_required_but_missing(None, require_pricing=True) is True


def test_pricing_required_but_missing_false_when_not_required() -> None:
    """With require_pricing off, a missing pricing row is allowed (legacy behavior)."""
    assert pricing_required_but_missing(None, require_pricing=False) is False


def test_pricing_required_but_missing_false_when_priced() -> None:
    """A model that has pricing is never rejected, regardless of require_pricing."""
    pricing = object()  # any non-None pricing row
    assert pricing_required_but_missing(pricing, require_pricing=True) is False  # type: ignore[arg-type]
