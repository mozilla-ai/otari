"""``require_pricing`` governs an organization provider key the way it governs
``providers:``: off, an unpriced model the key offers is served and logged at no
cost; on, it stays switched off and the switch refuses it.

Through the routes, so the offer, the serving switch, the overlay the dispatch
gate reads and the usage row are checked as one path rather than one at a time.
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
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.org_provider_key_service import reset_org_provider_cache

from .conftest import build_test_client

_MASTER_HEADER = {API_KEY_HEADER: "Bearer test-master-key"}
_KEYS = f"{API_ROOT}/organizations/me/provider-keys"
_MODEL = "gpt-6-unreleased"


@pytest.fixture(autouse=True)
def _secret_key_and_clean_cache(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()


@pytest.fixture
def _unpriced_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every dial lists one model, and nothing prices it."""

    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id=_MODEL, object="model", created=0, owned_by=impl_name)], error=None
        )

    monkeypatch.setattr("gateway.services.providers._org_provider_model_service.test_provider_credentials", _stub)
    monkeypatch.setattr(
        "gateway.services.organization_pricing_service.default_model_pricing", lambda *_args, **_kwargs: None
    )


@pytest.fixture
def strict_client(postgres_url: str) -> Generator[TestClient]:
    """A gateway with the production default, ``require_pricing=True``."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=True,
        default_pricing=False,
    )
    yield from build_test_client(config)


def _offer_key(client: TestClient) -> tuple[str, dict[str, Any]]:
    """Store an OpenAI key for the operator's organization and return it with its one offered model."""
    created = client.post(
        _KEYS, json={"provider": "openai", "name": "primary", "api_key": "sk-org-key-1234"}, headers=_MASTER_HEADER
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    key_id = created.json()["id"]
    assert client.post(f"{_KEYS}/{key_id}/default", headers=_MASTER_HEADER).status_code == status.HTTP_200_OK
    listed = client.get(f"{_KEYS}/{key_id}/models", headers=_MASTER_HEADER)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    [offered] = listed.json()["data"]
    assert offered["model"] == _MODEL
    assert offered["price_source"] is None
    return key_id, offered


def _api_key_header(client: TestClient) -> dict[str, str]:
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "app"}, headers=_MASTER_HEADER)
    assert created.status_code == status.HTTP_200_OK, created.text
    return {API_KEY_HEADER: f"Bearer {created.json()['key']}"}


async def _completion(**_kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-unpriced",
        object="chat.completion",
        created=0,
        model=_MODEL,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )


def _chat(client: TestClient, headers: dict[str, str]) -> Any:
    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _completion
        return client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": f"openai:{_MODEL}", "messages": [{"role": "user", "content": "hi"}]},
            headers=headers,
        )


@pytest.mark.usefixtures("_unpriced_discovery")
def test_with_require_pricing_off_an_unpriced_model_on_an_org_key_serves_at_no_cost(client: TestClient) -> None:
    """The shared client runs with ``require_pricing=False``."""
    _key_id, offered = _offer_key(client)
    assert offered["enabled"] is True

    response = _chat(client, _api_key_header(client))

    assert response.status_code == status.HTTP_200_OK, response.text
    assert "cost_usd" not in response.json()["usage"]
    [row] = client.get(f"{API_ROOT}/usage", headers=_MASTER_HEADER).json()
    assert row["model"] == _MODEL
    assert row["status"] == "success"
    assert row["cost"] is None
    assert row["prompt_tokens"] == 5


@pytest.mark.usefixtures("_unpriced_discovery")
def test_with_require_pricing_on_an_unpriced_model_on_an_org_key_stays_off(strict_client: TestClient) -> None:
    key_id, offered = _offer_key(strict_client)
    assert offered["enabled"] is False

    switched = strict_client.patch(
        f"{_KEYS}/{key_id}/models/{offered['id']}", json={"enabled": True}, headers=_MASTER_HEADER
    )
    assert switched.status_code == status.HTTP_400_BAD_REQUEST
    assert "Nothing prices" in switched.json()["detail"]

    response = _chat(strict_client, _api_key_header(strict_client))
    assert response.status_code == status.HTTP_403_FORBIDDEN
    # Refused at the gate, before dispatch: logged as the error it is, and spent nothing.
    [row] = strict_client.get(f"{API_ROOT}/usage", headers=_MASTER_HEADER).json()
    assert row["status"] == "error"
    assert row["status_code"] == status.HTTP_403_FORBIDDEN
    assert row["cost"] is None
