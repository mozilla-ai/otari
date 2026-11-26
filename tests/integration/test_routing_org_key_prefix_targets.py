"""A routing policy or alias may name an organization key by its request prefix.

A key with a custom endpoint is reached as ``<slug>:<model>`` (#2899), which
dispatch already resolves per workspace. Saving a policy or alias checked the
prefix against config instances, hosted providers and any-llm's providers only,
so a fallback between two keys on the same endpoint could not be stored.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.model import Model
from fastapi import status
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.org_provider_key_service import reset_org_provider_cache

from .conftest import shared_test_config

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_KEYS = f"{API_ROOT}/organizations/me/provider-keys"
_BASE = "https://api.synthetic.new/openai/v1"
_MODEL = "hf:zai-org/GLM-4.7"
_PRIMARY = f"synth-a:{_MODEL}"
_BACKUP = f"synth-b:{_MODEL}"


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so only the organization's keys serve."""
    return shared_test_config(postgres_url, {})


@pytest.fixture(autouse=True)
def _secret_key_and_clean_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _lists_the_model(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id=_MODEL, object="model", created=0, owned_by=impl_name)], error=None
        )

    monkeypatch.setattr(
        "gateway.services.providers._org_provider_model_service.test_provider_credentials", _lists_the_model
    )
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()


def _add_key(client: TestClient, slug: str, secret: str) -> str:
    created = client.post(
        _KEYS,
        json={"provider": "openai", "name": slug, "slug": slug, "api_base": _BASE, "api_key": secret},
        headers=_MASTER,
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    key_id: str = created.json()["id"]
    offered = client.get(f"{_KEYS}/{key_id}/models", headers=_MASTER)
    assert offered.status_code == status.HTTP_200_OK, offered.text
    return key_id


@pytest.fixture
def keys(client: TestClient) -> dict[str, str]:
    return {
        "synth-a": _add_key(client, "synth-a", "sk-synth-aaaa"),
        "synth-b": _add_key(client, "synth-b", "sk-synth-bbbb"),
    }


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-prefix",
        object="chat.completion",
        created=0,
        model=_MODEL,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )


def _chat(client: TestClient, model: str, *, refusing_key: str | None = None) -> tuple[Any, list[str]]:
    """Send ``model``, answering 429 for the key ``refusing_key``, and return the keys each attempt used."""
    used: list[str] = []

    async def upstream(**kwargs: Any) -> ChatCompletion:
        used.append(kwargs["api_key"])
        if kwargs["api_key"] == refusing_key:
            request = httpx.Request("POST", _BASE)
            raise httpx.HTTPStatusError("429", request=request, response=httpx.Response(429, request=request))
        return _completion()

    minted = client.post(f"{API_ROOT}/keys", json={"key_name": "app"}, headers=_MASTER)
    assert minted.status_code == status.HTTP_200_OK, minted.text
    with patch("gateway.api.routes.chat.acompletion", new=upstream):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
            headers={API_KEY_HEADER: f"Bearer {minted.json()['key']}"},
        )
    return response, used


def test_a_fallback_between_two_keys_on_one_endpoint_is_stored_and_falls_over(
    client: TestClient, keys: dict[str, str]
) -> None:
    created = client.post(
        f"{API_ROOT}/routing/policies",
        json={"name": "synthetic", "spec": {"select": [{"default": _PRIMARY}], "on_failure": [_BACKUP]}},
        headers=_MASTER,
    )
    assert created.status_code == status.HTTP_200_OK, created.text

    response, used = _chat(client, "synthetic", refusing_key="sk-synth-aaaa")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert used == ["sk-synth-aaaa", "sk-synth-bbbb"]
    assert response.headers["Otari-Provider"] == "synth-b"


def test_an_alias_may_name_a_key_by_its_prefix(client: TestClient, keys: dict[str, str]) -> None:
    created = client.post(f"{API_ROOT}/aliases", json={"name": "glm", "target": _BACKUP}, headers=_MASTER)
    assert created.status_code == status.HTTP_200_OK, created.text

    response, used = _chat(client, "glm")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert used == ["sk-synth-bbbb"]


def test_a_prefix_no_key_of_the_workspace_answers_to_is_still_refused(client: TestClient, keys: dict[str, str]) -> None:
    refused = client.post(
        f"{API_ROOT}/routing/policies",
        json={"name": "nowhere", "spec": {"select": [{"default": f"synth-c:{_MODEL}"}]}},
        headers=_MASTER,
    )
    assert refused.status_code == status.HTTP_400_BAD_REQUEST, refused.text
    assert "synth-c" in refused.json()["detail"]


def test_a_key_the_workspace_turned_off_is_not_a_target(client: TestClient, keys: dict[str, str]) -> None:
    [workspace] = client.get(f"{API_ROOT}/workspaces", headers=_MASTER).json()["data"]
    disabled = client.patch(
        f"{API_ROOT}/workspaces/{workspace['id']}/provider-keys/{keys['synth-a']}",
        json={"disabled": True},
        headers=_MASTER,
    )
    assert disabled.status_code == status.HTTP_200_OK, disabled.text

    refused = client.post(f"{API_ROOT}/aliases", json={"name": "glm-off", "target": _PRIMARY}, headers=_MASTER)

    assert refused.status_code == status.HTTP_400_BAD_REQUEST, refused.text
