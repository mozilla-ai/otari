"""Integration tests for the reviewable genai-prices refresh API."""

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

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


_RAW_SNAPSHOT = (
    '[{"id":"test","name":"Test","api_pattern":"","models":['
    '{"id":"model","match":{"equals":"model"},"prices":{"input_mtok":"1","output_mtok":"2"}}]}]'
)


def test_nothing_pending_is_a_404_not_a_refresh(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.get("/v1/pricing/refresh/pending", headers=master_key_header)
    assert response.status_code == 404


def test_a_pending_update_is_previewed_without_fetching_and_accepting_it_is_remembered(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """What the scheduled refresh leaves behind under the review policy."""
    from gateway.models.entities import PricingSnapshot
    from gateway.services.pricing_refresh_service import GENAI_PRICES_PENDING_SOURCE, reset_price_refresh_state

    session = db_session_factory()
    try:
        session.add(PricingSnapshot(source=GENAI_PRICES_PENDING_SOURCE, snapshot=_RAW_SNAPSHOT))
        session.commit()
    finally:
        session.close()

    try:
        pending = client.get("/v1/pricing/refresh/pending", headers=master_key_header)
        assert pending.status_code == 200, pending.text
        # The bundled dataset has no provider called "test", so its one model is an addition.
        assert pending.json()["added_count"] == 1

        assert client.get("/v1/pricing/snapshots", headers=master_key_header).json() == []
        confirmed = client.post("/v1/pricing/refresh/confirm", headers=master_key_header)
        assert confirmed.status_code == 200, confirmed.text

        history = client.get("/v1/pricing/snapshots", headers=master_key_header).json()
        assert len(history) == 1
        assert history[0]["accepted_by"] == "operator"
        assert history[0]["model_count"] == 1
        assert client.get("/v1/pricing/refresh/pending", headers=master_key_header).status_code == 404
    finally:
        # The confirm applied the tiny snapshot process-wide; put the bundled one back.
        reset_price_refresh_state()


def test_drift_puts_a_stored_rate_beside_todays_default(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    turned_on = client.patch("/v1/settings", json={"default_pricing": True}, headers=master_key_header)
    assert turned_on.status_code == 200, turned_on.text
    try:
        priced = client.post(
            "/v1/pricing",
            json={"model_key": "openai:gpt-4o-mini", "input_price_per_million": 0.3, "output_price_per_million": 1.2},
            headers=master_key_header,
        )
        assert priced.status_code == 200
        tool = client.post(
            "/v1/pricing",
            json={
                "model_key": "otari:web_search",
                "input_price_per_million": 5000,
                "output_price_per_million": 0,
                "unit": "requests",
            },
            headers=master_key_header,
        )
        assert tool.status_code == 200

        rows = client.get("/v1/pricing/drift", headers=master_key_header).json()
        # A per-request tool has no default to drift from and is left out.
        assert [row["model_key"] for row in rows] == ["openai:gpt-4o-mini"]
        row = rows[0]
        assert row["origin"] == "api"
        assert row["default_input_price_per_million"] == 0.15
        assert row["default_reference"] == "openai:gpt-4o-mini"
        assert row["input_delta_percent"] == 100.0
        assert row["output_delta_percent"] == 100.0
    finally:
        client.patch("/v1/settings", json={"default_pricing": False}, headers=master_key_header)
