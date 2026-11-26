"""A model an organization reaches through an OpenAI-compatible proxy key is listed with its metadata.

The key is stored under ``openai``, the wire format it speaks, and models.dev
lists no DeepSeek model there, so the listing describes the model from the
entry of the model it groups under: DeepSeek's own ``deepseek-flash``.
"""

from collections.abc import Generator

import pytest
from any_llm.types.model import Model
from fastapi import status
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.services import catalog as selectors
from gateway.services import model_catalog_service as mcs
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.pricing import bundled_catalog
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.org_provider_key_service import reset_org_provider_cache

from .conftest import shared_test_config

pytestmark = pytest.mark.usefixtures("public_dns")


_MASTER_HEADER = {API_KEY_HEADER: "Bearer test-master-key"}
_KEYS = f"{API_ROOT}/organizations/me/provider-keys"
_MODEL = "deepseek-v4.1-flash-1017"
_CATALOG_ID = "deepseek/deepseek-v4.1-flash-1017"


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so only the organization's key serves the model."""
    return shared_test_config(postgres_url, {})


@pytest.fixture(autouse=True)
def _secret_key_and_clean_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()
    selectors.reset_selector_index()


@pytest.fixture(autouse=True)
def _proxy_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id=_MODEL, object="model", created=0, owned_by=impl_name)], error=None
        )

    monkeypatch.setattr("gateway.services.providers._org_provider_model_service.test_provider_credentials", _stub)


@pytest.fixture(autouse=True)
def _bundled_models_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcs, "_fallback", bundled_catalog)


@pytest.fixture
def api_key_header(client: TestClient) -> dict[str, str]:
    """Offer the model on a slugged OpenAI-compatible key, rebuild the index, and mint a workspace API key."""
    created = client.post(
        _KEYS,
        json={
            "provider": "openai",
            "name": "Chat proxy",
            "slug": "chat-proxy",
            "api_base": "https://llm-proxy.example.com/v1",
            "api_key": "sk-proxy-1234",
        },
        headers=_MASTER_HEADER,
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    offered = client.get(f"{_KEYS}/{created.json()['id']}/models", headers=_MASTER_HEADER)
    assert offered.status_code == status.HTTP_200_OK, offered.text
    assert [(row["model"], row["enabled"]) for row in offered.json()["data"]] == [(_MODEL, True)]
    rebuilt = client.post(f"{API_ROOT}/catalog/selectors/refresh", headers=_MASTER_HEADER)
    assert rebuilt.status_code == status.HTTP_200_OK, rebuilt.text
    minted = client.post(f"{API_ROOT}/keys", json={"key_name": "app"}, headers=_MASTER_HEADER)
    assert minted.status_code == status.HTTP_200_OK, minted.text
    return {API_KEY_HEADER: f"Bearer {minted.json()['key']}"}


def test_the_listing_describes_a_model_served_on_an_openai_compatible_key(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    listed = client.get(f"{API_ROOT}/models", headers=api_key_header)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    (row,) = [model for model in listed.json()["data"] if model["id"] == _CATALOG_ID]

    assert row["name"] == "DeepSeek V4.1 Flash 1017"
    assert row["type"] == "chat"
    assert row["context_window"] is not None
    assert row["supported_parameters"]
