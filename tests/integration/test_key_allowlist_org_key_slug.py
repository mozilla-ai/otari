"""A key's model allow-list can name a model an organization's slugged key serves.

A key with a custom endpoint is named by its slug (``acme-ds:deepseek-flash``),
not by its provider, so an allow-list entry spelled ``deepseek:deepseek-flash``
matches nothing it serves. The allow-list has to accept the slug, and a catalog
ID has to stand for the slugged offering, or a restricted key reaches nothing
on the organization's own key.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
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
_SLUG = "acme-ds"
_MODEL = "deepseek-flash"
_CATALOG_ID = "deepseek/deepseek-v4.1-flash-1017"


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so only the organization's key serves DeepSeek."""
    return shared_test_config(postgres_url, {})


@pytest.fixture(autouse=True)
def _secret_key_and_clean_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()
    selectors.reset_selector_index()


@pytest.fixture(autouse=True)
def _deepseek_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id=_MODEL, object="model", created=0, owned_by=impl_name)], error=None
        )

    monkeypatch.setattr("gateway.services.providers._org_provider_model_service.test_provider_credentials", _stub)


@pytest.fixture(autouse=True)
def _bundled_models_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcs, "_fallback", bundled_catalog)


def _offer(client: TestClient, **key: str) -> None:
    """Store a DeepSeek key, offer ``deepseek-flash`` on it, and rebuild the selector index."""
    created = client.post(
        _KEYS, json={"provider": "deepseek", "name": "DeepSeek", "api_key": "sk-ds-1234", **key}, headers=_MASTER_HEADER
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    offered = client.get(f"{_KEYS}/{created.json()['id']}/models", headers=_MASTER_HEADER)
    assert offered.status_code == status.HTTP_200_OK, offered.text
    rebuilt = client.post(f"{API_ROOT}/catalog/selectors/refresh", headers=_MASTER_HEADER)
    assert rebuilt.status_code == status.HTTP_200_OK, rebuilt.text


def _mint(client: TestClient, allowed_models: list[str]) -> tuple[dict[str, str], list[str]]:
    minted = client.post(
        f"{API_ROOT}/keys", json={"key_name": "app", "allowed_models": allowed_models}, headers=_MASTER_HEADER
    )
    assert minted.status_code == status.HTTP_200_OK, minted.text
    stored = client.get(f"{API_ROOT}/keys/{minted.json()['id']}", headers=_MASTER_HEADER).json()["allowed_models"]
    return {API_KEY_HEADER: f"Bearer {minted.json()['key']}"}, stored


def _chat(client: TestClient, headers: dict[str, str], model: str) -> Any:
    completion = ChatCompletion(
        id="chatcmpl-slug",
        object="chat.completion",
        created=0,
        model=_MODEL,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )
    with patch("gateway.api.routes.chat.acompletion", return_value=completion):
        return client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
            headers=headers,
        )


def _listed(client: TestClient, headers: dict[str, str]) -> set[str]:
    listed = client.get(f"{API_ROOT}/models", headers=headers)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    return {model["id"] for model in listed.json()["data"]}


def test_a_key_without_a_base_url_is_reached_by_its_provider(client: TestClient) -> None:
    _offer(client)
    headers, stored = _mint(client, [f"deepseek:{_MODEL}"])

    assert stored == [f"deepseek:{_MODEL}"]
    assert _CATALOG_ID in _listed(client, headers)
    assert _chat(client, headers, f"deepseek:{_MODEL}").status_code == status.HTTP_200_OK


@pytest.mark.parametrize("entry", [f"{_SLUG}:{_MODEL}", f"{_SLUG}:*", _CATALOG_ID])
def test_an_allow_list_reaches_a_slugged_key(client: TestClient, entry: str) -> None:
    _offer(client, slug=_SLUG, api_base="https://deepseek-proxy.example.com/v1")
    headers, stored = _mint(client, [entry])

    assert stored == [entry if entry.startswith(_SLUG) else f"{_SLUG}:{_MODEL}"]
    assert _CATALOG_ID in _listed(client, headers)
    assert _chat(client, headers, f"{_SLUG}:{_MODEL}").status_code == status.HTTP_200_OK
    assert _chat(client, headers, _CATALOG_ID).status_code == status.HTTP_200_OK


def test_the_provider_name_does_not_reach_a_slugged_key(client: TestClient) -> None:
    """``deepseek:`` names the provider, which the slugged key is not reached by, so nothing is listed."""
    _offer(client, slug=_SLUG, api_base="https://deepseek-proxy.example.com/v1")
    headers, _ = _mint(client, [f"deepseek:{_MODEL}"])

    assert _CATALOG_ID not in _listed(client, headers)


def test_another_organizations_slug_is_refused(client: TestClient) -> None:
    minted = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "app", "allowed_models": [f"{_SLUG}:{_MODEL}"]},
        headers=_MASTER_HEADER,
    )

    assert minted.status_code == status.HTTP_400_BAD_REQUEST, minted.text
