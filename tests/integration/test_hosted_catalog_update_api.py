"""The catalog update routes over HTTP, and the deprecated price approval that now runs them.

The operation's rules are pinned in ``test_hosted_catalog_update.py``; this file
pins that each route reaches it, how a stale review refuses, and that only the
operator reaches it. The models.dev steps and model discovery are stubbed.
"""

from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT
from gateway.models.pricing import ModelPricing
from gateway.services import pricing_refresh_service
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.pricing_refresh_service import PendingSnapshotChanged, PricingRefreshPreview
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.org_provider_key_service import reset_org_provider_cache

PATH = f"{API_ROOT}/hosted-providers"
ORG_KEYS = f"{API_ROOT}/organizations/me/provider-keys"
KEY = "sk-live-hosted-1234"
DIGEST = "d" * 64


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()


def _listing(*models: str) -> Callable[..., Awaitable[ProviderDiscovery]]:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id=m, object="model", created=0, owned_by=impl_name) for m in models]
        )

    return _stub


def _lists(monkeypatch: pytest.MonkeyPatch, *models: str) -> None:
    monkeypatch.setattr(
        "gateway.services.providers._hosted_provider_service.test_provider_credentials", _listing(*models)
    )


def _community(monkeypatch: pytest.MonkeyPatch, rates: dict[str, tuple[str, str]]) -> None:
    """Make the community dataset an organization key seeds from price exactly ``rates``."""

    def _stub(provider: str | None, model: str, as_of: object, **_: object) -> ModelPricing | None:
        if model not in rates:
            return None
        input_rate, output_rate = rates[model]
        return ModelPricing(
            model_key=f"{provider}:{model}",
            input_price_per_million=Decimal(input_rate),
            output_price_per_million=Decimal(output_rate),
            pricing_tiers=[],
            unit="token",
        )

    monkeypatch.setattr("gateway.services.organization_pricing_service.default_model_pricing", _stub)


def _prices(monkeypatch: pytest.MonkeyPatch, *, approve: object = True) -> list[str]:
    """Stub the models.dev steps; returns the digests approved."""
    approved: list[str] = []

    async def fetch(_self: object) -> PricingRefreshPreview:
        return PricingRefreshPreview(
            fetched_at=datetime.now(UTC),
            added_count=2,
            changed_count=1,
            removed_count=0,
            changes=[],
            changes_truncated=False,
            digest=DIGEST,
        )

    async def approve_digest(_self: object, digest: str) -> bool:
        approved.append(digest)
        if isinstance(approve, Exception):
            raise approve
        return bool(approve)

    monkeypatch.setattr(pricing_refresh_service.ModelsDevPriceUpdates, "fetch", fetch)
    monkeypatch.setattr(pricing_refresh_service.ModelsDevPriceUpdates, "approve", approve_digest)
    return approved


def _provider(client: TestClient, headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    _lists(monkeypatch, "gpt-4o")
    assert client.post(PATH, json={"provider": "openai", "api_key": KEY}, headers=headers).status_code == 201
    _lists(monkeypatch, "gpt-4o", "gpt-5-1017")


def test_the_preview_reports_the_prices_and_what_each_provider_newly_lists(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _provider(client, master_key_header, monkeypatch)
    _prices(monkeypatch)

    response = client.post(f"{PATH}/catalog-update/preview", json={}, headers=master_key_header)

    assert response.status_code == 200
    body = response.json()
    assert (body["prices"]["digest"], body["prices"]["added_count"], body["price_error"]) == (DIGEST, 2, None)
    assert [(p["provider"], p["added"]) for p in body["catalog"]["providers"]] == [("openai", ["gpt-5-1017"])]


def test_the_update_applies_the_reviewed_prices_then_offers_what_is_new(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _provider(client, master_key_header, monkeypatch)
    approved = _prices(monkeypatch)

    response = client.post(f"{PATH}/catalog-update", json={"digest": DIGEST}, headers=master_key_header)

    assert response.status_code == 200
    assert approved == [DIGEST]
    assert response.json()["prices_applied"] is True
    models = client.get(f"{PATH}/openai/models", headers=master_key_header).json()["data"]
    assert sorted(model["model"] for model in models) == ["gpt-4o", "gpt-5-1017"]


def test_the_update_prices_an_organization_keys_model_the_applied_prices_cover(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No Refresh pricing press per key: the update carries the new prices to organization keys.

    On DeepSeek, which the shared config does not declare, so the deployment does not own its prices.
    """
    _provider(client, master_key_header, monkeypatch)
    _prices(monkeypatch)
    monkeypatch.setattr(
        "gateway.services.providers._org_provider_model_service.test_provider_credentials",
        _listing("deepseek-chat"),
    )
    _community(monkeypatch, {})
    created = client.post(
        ORG_KEYS, json={"provider": "deepseek", "name": "primary", "api_key": "sk-org-1234"}, headers=master_key_header
    )
    assert created.status_code == 201, created.text
    key = f"{ORG_KEYS}/{created.json()['id']}"
    assert client.post(f"{key}/default", headers=master_key_header).status_code == 200
    models = f"{key}/models"
    [before] = client.get(models, headers=master_key_header).json()["data"]
    assert before["price_source"] is None

    _community(monkeypatch, {"deepseek-chat": ("0.28", "0.42")})
    response = client.post(f"{PATH}/catalog-update", json={"digest": DIGEST}, headers=master_key_header)

    assert response.status_code == 200
    assert response.json()["organization_models_repriced"] == 1
    [after] = client.get(models, headers=master_key_header).json()["data"]
    assert (after["price_source"], after["input_price_per_million"]) == ("defaults", 0.28)


@pytest.mark.parametrize("approve", [PendingSnapshotChanged("replaced"), False])
def test_an_update_whose_prices_changed_since_the_preview_is_a_409_that_offers_nothing(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch, approve: object
) -> None:
    _provider(client, master_key_header, monkeypatch)
    _prices(monkeypatch, approve=approve)

    response = client.post(f"{PATH}/catalog-update", json={"digest": DIGEST}, headers=master_key_header)

    assert response.status_code == 409
    models = client.get(f"{PATH}/openai/models", headers=master_key_header).json()["data"]
    assert [model["model"] for model in models] == ["gpt-4o"]


def test_the_catalog_update_refuses_a_request_with_no_credential(client: TestClient) -> None:
    assert client.post(f"{PATH}/catalog-update/preview", json={}).status_code == 401
    assert client.post(f"{PATH}/catalog-update", json={}).status_code == 401
