"""Hosted providers take part in the operator's discovery and health views (otari#1294).

A hosted provider is dialed on its stored credential beside the configured
instances, deployment-wide, and cached under its provider name. Its upstream
listing stays out of the catalog, which lists the roster the operator switched
on. Every dial is stubbed at the hosted-provider service's boundary.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.api.deps import build_hosted_discovery
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.model_discovery_service import ProviderDiscovery, get_model_cache
from gateway.services.secret_box import generate_secret_key

from .conftest import shared_test_config


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so nothing configured shadows a hosted one."""
    return shared_test_config(postgres_url, {})


KEY = "sk-live-hosted-1234"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    get_model_cache().clear()
    yield
    get_model_cache().clear()


@pytest.fixture(autouse=True)
def _dials(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def _stub(impl_name: str, **kwargs: Any) -> ProviderDiscovery:
        calls.append({"provider": impl_name, **kwargs})
        return ProviderDiscovery(
            provider=impl_name,
            models=[
                Model(id=name, object="model", created=0, owned_by=impl_name) for name in ("gpt-4o", "gpt-4o-mini")
            ],
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)
    return calls


def _hosted(client: TestClient, headers: dict[str, str], **extra: Any) -> None:
    response = client.post(
        f"{API_ROOT}/hosted-providers", json={"provider": "openai", "api_key": KEY, **extra}, headers=headers
    )
    assert response.status_code == 201, response.text


def test_the_discoverable_listing_carries_a_hosted_provider(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _hosted(client, master_key_header)

    response = client.get(f"{API_ROOT}/models/discoverable", params={"refresh": True}, headers=master_key_header)

    assert response.status_code == 200, response.text
    [provider] = [row for row in response.json()["providers"] if row["provider"] == "openai"]
    assert provider["ok"] is True
    assert provider["checked_at"] is not None
    assert [model["key"] for model in provider["models"]] == ["openai:gpt-4o", "openai:gpt-4o-mini"]
    assert KEY not in response.text


def test_provider_health_counts_a_hosted_provider(
    client: TestClient, master_key_header: dict[str, str], _dials: list[dict[str, Any]]
) -> None:
    _hosted(client, master_key_header)
    _dials.clear()

    response = client.get(f"{API_ROOT}/providers/health", params={"refresh": True}, headers=master_key_header)

    assert response.status_code == 200, response.text
    body = response.json()
    [openai] = [row for row in body["providers"] if row["instance"] == "openai"]
    assert openai["ok"] is True
    assert openai["model_count"] == 2
    assert body["healthy"] == body["total"]
    assert [call["api_key"] for call in _dials] == [KEY]
    assert KEY not in response.text


def test_a_provider_switched_off_leaves_both_views(client: TestClient, master_key_header: dict[str, str]) -> None:
    _hosted(client, master_key_header, enabled=False)

    discoverable = client.get(f"{API_ROOT}/models/discoverable", headers=master_key_header).json()
    health = client.get(f"{API_ROOT}/providers/health", headers=master_key_header).json()

    assert "openai" not in {row["provider"] for row in discoverable["providers"]}
    assert "openai" not in {row["instance"] for row in health["providers"]}


def test_an_undecryptable_provider_reads_as_unhealthy(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    _dials: list[dict[str, Any]],
) -> None:
    _hosted(client, master_key_header)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _dials.clear()

    body = client.get(f"{API_ROOT}/providers/health", params={"refresh": True}, headers=master_key_header).json()

    [openai] = [row for row in body["providers"] if row["instance"] == "openai"]
    assert openai["ok"] is False
    assert "OTARI_SECRET_KEY" in openai["error"]
    assert _dials == []


def test_the_catalog_still_lists_the_roster_alone(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A model the upstream lists but the operator switched off stays out, once discovery has dialed it."""
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    _hosted(client, master_key_header)
    client.get(f"{API_ROOT}/models/discoverable", params={"refresh": True}, headers=master_key_header)

    ids = {model["id"] for model in client.get(f"{API_ROOT}/models", headers=master_key_header).json()["data"]}

    assert "openai/gpt-4o" in ids
    assert "openai/gpt-4o-mini" not in ids


def test_the_refresher_reads_the_hosted_providers_in_a_transaction_of_its_own(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    _dials: list[dict[str, Any]],
) -> None:
    """The lifespan worker has no request, so it opens its own unit of work for each tick."""
    _hosted(client, master_key_header)
    _dials.clear()

    async def tick() -> list[str]:
        dials = await build_hosted_discovery(test_config)()
        return [(await dial()).provider for dial in dials.values()]

    assert client.portal is not None
    assert client.portal.call(tick) == ["openai"]
    assert [call["api_key"] for call in _dials] == [KEY]
