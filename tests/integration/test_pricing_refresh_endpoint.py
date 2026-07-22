"""Integration tests for the reviewable genai-prices refresh API."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from gateway.api.routes import pricing as pricing_route
from gateway.services.pricing_refresh_service import PricingRefreshPreview


def test_preview_pricing_refresh_reports_protected_custom_prices(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Previewing defaults does not overwrite, and identifies, custom prices."""

    configured = client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 0.123,
            "output_price_per_million": 0.456,
        },
        headers=master_key_header,
    )
    assert configured.status_code == 200

    async def preview(_: object) -> PricingRefreshPreview:
        return PricingRefreshPreview(
            fetched_at=datetime.now(UTC),
            added_count=1,
            changed_count=2,
            removed_count=3,
            changes=[],
            changes_truncated=False,
        )

    monkeypatch.setattr(pricing_route, "prepare_price_refresh", preview)

    response = client.post("/v1/pricing/refresh", headers=master_key_header)

    assert response.status_code == 200
    data = response.json()
    assert data["added_count"] == 1
    assert data["changed_count"] == 2
    assert data["removed_count"] == 3
    assert data["protected_model_count"] == 1
    assert data["changes"] == []
    assert data["changes_truncated"] is False


def test_confirm_pricing_refresh_requires_pending_preview(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Confirmation cannot activate data that was never reviewed."""

    response = client.post("/v1/pricing/refresh/confirm", headers=master_key_header)

    assert response.status_code == 409


def test_reject_pricing_refresh_requires_pending_preview(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Rejecting cannot mutate state when there is no reviewed snapshot."""

    response = client.post("/v1/pricing/refresh/reject", headers=master_key_header)

    assert response.status_code == 409


def test_pricing_refresh_requires_master_key(client: TestClient) -> None:
    """Fetching operator-controlled pricing data is master-key-only."""

    response = client.post("/v1/pricing/refresh")

    assert response.status_code == 401
