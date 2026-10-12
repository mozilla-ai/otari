"""A provider's undated name reaches the dated build an organization's key offers.

Anthropic lists ``claude-haiku-4-5-20251001`` and documents
``claude-haiku-4-5`` as its alias. Through the routes, so the offer, the
selector index, the serving switch the dispatch gate reads and the usage row
are checked as one path: the undated name is served as the dated build, a name
with two dated builds is refused with both named, and a dated build switched
off refuses its undated name as not serving.
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
from gateway.core.error_codes import MODEL_NOT_FOUND, MODEL_NOT_SERVING
from gateway.services import catalog as selectors
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.org_provider_key_service import reset_org_provider_cache

from .conftest import build_test_client

_MASTER_HEADER = {API_KEY_HEADER: "Bearer test-master-key"}
_KEYS = f"{API_ROOT}/organizations/me/provider-keys"
_HAIKU = "claude-haiku-4-5-20251001"
_LISTED = (_HAIKU, "claude-sonnet-4-6", "claude-twin-20250101", "claude-twin-20250601")


@pytest.fixture(autouse=True)
def _secret_key_and_clean_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()
    selectors.reset_selector_index()


@pytest.fixture(autouse=True)
def _anthropic_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every dial lists Anthropic's models the way its API does: most by dated id."""

    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name,
            models=[Model(id=model, object="model", created=0, owned_by=impl_name) for model in _LISTED],
            error=None,
        )

    monkeypatch.setattr("gateway.services.providers._org_provider_model_service.test_provider_credentials", _stub)


def _offer_key(client: TestClient) -> tuple[str, dict[str, str]]:
    """Store an Anthropic key for the operator's organization; return it and its offered rows' ids by model."""
    created = client.post(
        _KEYS, json={"provider": "anthropic", "name": "primary", "api_key": "sk-ant-1234"}, headers=_MASTER_HEADER
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    key_id = created.json()["id"]
    assert client.post(f"{_KEYS}/{key_id}/default", headers=_MASTER_HEADER).status_code == status.HTTP_200_OK
    listed = client.get(f"{_KEYS}/{key_id}/models", headers=_MASTER_HEADER)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    rows = {row["model"]: row for row in listed.json()["data"]}
    assert set(rows) == set(_LISTED)
    assert all(row["enabled"] for row in rows.values())
    return key_id, {model: row["id"] for model, row in rows.items()}


def _refresh_index(client: TestClient) -> None:
    rebuilt = client.post(f"{API_ROOT}/catalog/selectors/refresh", headers=_MASTER_HEADER)
    assert rebuilt.status_code == status.HTTP_200_OK, rebuilt.text


def _api_key_header(client: TestClient) -> dict[str, str]:
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "app"}, headers=_MASTER_HEADER)
    assert created.status_code == status.HTTP_200_OK, created.text
    return {API_KEY_HEADER: f"Bearer {created.json()['key']}"}


def _chat(client: TestClient, headers: dict[str, str], model: str, captured: dict[str, Any] | None = None) -> Any:
    async def _completion(**kwargs: Any) -> ChatCompletion:
        if captured is not None:
            captured.update(kwargs)
        return ChatCompletion(
            id="chatcmpl-dated",
            object="chat.completion",
            created=0,
            model=_HAIKU,
            choices=[
                Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")
            ],
            usage=CompletionUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )

    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _completion
        return client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
            headers=headers,
        )


def test_the_undated_name_is_served_as_the_one_dated_build_the_key_offers(client: TestClient) -> None:
    _offer_key(client)
    _refresh_index(client)
    captured: dict[str, Any] = {}

    response = _chat(client, _api_key_header(client), "anthropic:claude-haiku-4-5", captured)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert captured["model"] == f"anthropic:{_HAIKU}"
    # Relabeled like an alias: the caller sees the spelling they sent.
    assert response.json()["model"] == "anthropic:claude-haiku-4-5"
    [row] = client.get(f"{API_ROOT}/usage", headers=_MASTER_HEADER).json()
    assert (row["model"], row["provider"], row["status"]) == (_HAIKU, "anthropic", "success")


def test_a_name_with_two_dated_builds_is_not_found_and_both_are_named(client: TestClient) -> None:
    _offer_key(client)
    _refresh_index(client)

    response = _chat(client, _api_key_header(client), "anthropic:claude-twin")

    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text
    assert response.json()["code"] == MODEL_NOT_FOUND
    assert response.json()["detail"] == (
        "Provider key 'primary' does not offer 'anthropic:claude-twin'. "
        "It offers 'anthropic:claude-twin-20250101', 'anthropic:claude-twin-20250601'; send one of those."
    )


def test_the_undated_name_of_a_dated_build_switched_off_is_not_serving(client: TestClient) -> None:
    key_id, rows = _offer_key(client)
    switched = client.patch(f"{_KEYS}/{key_id}/models/{rows[_HAIKU]}", json={"enabled": False}, headers=_MASTER_HEADER)
    assert switched.status_code == status.HTTP_200_OK, switched.text
    _refresh_index(client)

    response = _chat(client, _api_key_header(client), "anthropic:claude-haiku-4-5")

    assert response.status_code == status.HTTP_403_FORBIDDEN, response.text
    assert response.json()["code"] == MODEL_NOT_SERVING
    assert "as 'anthropic:claude-haiku-4-5-20251001' but not serving" in response.json()["detail"]


def test_a_bare_alias_sharing_the_undated_name_does_not_disclose_the_dated_builds(client: TestClient) -> None:
    """The alias's spelling matches its target's model, so only its provenance keeps the target off the wire."""
    _offer_key(client)
    _refresh_index(client)
    aliased = client.post(
        f"{API_ROOT}/aliases", json={"name": "claude-twin", "target": "anthropic:claude-twin"}, headers=_MASTER_HEADER
    )
    assert aliased.status_code == status.HTTP_200_OK, aliased.text

    response = _chat(client, _api_key_header(client), "claude-twin")

    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text
    assert response.json()["code"] == MODEL_NOT_FOUND
    assert "2025" not in response.json()["detail"]


@pytest.fixture
def instance_client(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    """A gateway whose ``config.yml`` also declares an ``anthropic`` instance."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        providers={"anthropic": {"api_key": "sk-ant-deployment"}},
    )
    yield from build_test_client(config)


def test_a_providers_instance_receives_the_undated_name_as_written(instance_client: TestClient) -> None:
    """The organization's offer is indexed, but an instance-addressed selector never reaches its key."""
    _offer_key(instance_client)
    _refresh_index(instance_client)
    captured: dict[str, Any] = {}

    response = _chat(instance_client, _api_key_header(instance_client), "anthropic:claude-haiku-4-5", captured)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert (captured["model"], captured["api_key"]) == ("anthropic:claude-haiku-4-5", "sk-ant-deployment")
