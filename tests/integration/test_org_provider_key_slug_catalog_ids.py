"""A model an organization's slugged key offers is listed under the catalog id a request can send (#3018).

A key with a custom endpoint is named by its slug, which is no configured
instance, so only the key says which provider's models.dev entry names the
model. DeepSeek's ``deepseek-flash`` is named "DeepSeek V4.1 Flash 1017" there,
so the catalog id the selector index builds differs from the raw model name.
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
    """The bundled models.dev catalog, which names ``deepseek-flash``."""
    monkeypatch.setattr(mcs, "_fallback", bundled_catalog)


@pytest.fixture
def api_key_header(client: TestClient) -> dict[str, str]:
    """Offer ``deepseek-flash`` on a slugged DeepSeek key, rebuild the index, and mint a workspace API key."""
    created = client.post(
        _KEYS,
        json={
            "provider": "deepseek",
            "name": "DeepSeek proxy",
            "slug": _SLUG,
            "api_base": "https://deepseek-proxy.example.com/v1",
            "api_key": "sk-ds-1234",
        },
        headers=_MASTER_HEADER,
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    key_id = created.json()["id"]
    offered = client.get(f"{_KEYS}/{key_id}/models", headers=_MASTER_HEADER)
    assert offered.status_code == status.HTTP_200_OK, offered.text
    assert [(row["model"], row["enabled"]) for row in offered.json()["data"]] == [(_MODEL, True)]
    rebuilt = client.post(f"{API_ROOT}/catalog/selectors/refresh", headers=_MASTER_HEADER)
    assert rebuilt.status_code == status.HTTP_200_OK, rebuilt.text
    minted = client.post(f"{API_ROOT}/keys", json={"key_name": "app"}, headers=_MASTER_HEADER)
    assert minted.status_code == status.HTTP_200_OK, minted.text
    return {API_KEY_HEADER: f"Bearer {minted.json()['key']}"}


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


def test_the_listing_names_the_model_by_the_catalog_id_the_index_builds(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    listed = client.get(f"{API_ROOT}/models", headers=api_key_header)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    ids = {model["id"] for model in listed.json()["data"]}

    assert _CATALOG_ID in ids
    assert f"deepseek/{_MODEL}" not in ids
    detail = client.get(f"{API_ROOT}/models/{_CATALOG_ID}", headers=api_key_header)
    assert detail.status_code == status.HTTP_200_OK, detail.text
    assert detail.json()["id"] == _CATALOG_ID


def test_every_listed_id_is_served_on_the_organizations_key(client: TestClient, api_key_header: dict[str, str]) -> None:
    listed = client.get(f"{API_ROOT}/models", headers=api_key_header)
    assert listed.status_code == status.HTTP_200_OK, listed.text

    for model in listed.json()["data"]:
        response = _chat(client, api_key_header, model["id"])
        assert response.status_code == status.HTTP_200_OK, (model["id"], response.text)
        assert response.headers["Otari-Provider"] == _SLUG
        assert response.headers["Otari-Model"] == _MODEL


def test_the_catalog_ids_bare_name_is_served_on_the_organizations_key(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """A client that sends a model's bare name, as Claude Code does, reaches it with no alias."""
    bare = _CATALOG_ID.partition("/")[2]

    response = _chat(client, api_key_header, bare)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.headers["Otari-Provider"] == _SLUG
    assert response.headers["Otari-Model"] == _MODEL
    assert response.json()["model"] == bare
