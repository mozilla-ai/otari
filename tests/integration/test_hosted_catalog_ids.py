"""A hosted model a write switches on is reachable by its catalog id at once.

The selector index a catalog id resolves through is a process-wide snapshot the
background refresher rebuilds every minute. That refresher is stopped here, so
only the write's own rebuild can make the catalog id resolve: without it,
``GET /models`` lists the model while a request by the same id answers 400
``invalid_model``. The provider listing and the provider call are stubbed.
"""

from collections.abc import Generator, Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client, shared_test_config

PATH = f"{API_ROOT}/hosted-providers"
MODEL = "mistralai/mistral-small-4"
OTARI = {"provider": "otari", "api_key": "sk-otari", "api_base": "https://api.otari.example"}


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared config with no ``providers:`` and one hosted otari.ai, as ``hosted_providers:`` declares it."""
    return shared_test_config(postgres_url, {}).model_copy(update={"hosted_providers": {"otari-ai": OTARI}})


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """otari.ai lists one model at a price, so a refresh switches it on."""
    pricing = {"input_price_per_million": 0.1, "output_price_per_million": 0.3}

    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        model = Model(id=MODEL, object="model", created=0, owned_by="mistralai", pricing=pricing)  # type: ignore[call-arg]
        return ProviderDiscovery(provider=impl_name, models=[model])

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


@pytest.fixture
def client(test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    """A client whose selector index only a request rebuilds, so no background tick can mask a missed rebuild."""

    async def _idle(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("gateway.main.run_selector_index_refresher", _idle)
    yield from build_test_client(test_config)


@pytest.fixture
def headers(client: TestClient, master_key_header: dict[str, str]) -> dict[str, str]:
    response = client.post(f"{API_ROOT}/users", json={"user_id": "u1"}, headers=master_key_header)
    assert response.status_code == 200, response.text
    return master_key_header


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content="hi"))],
        created=0,
        model=MODEL,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _chat(client: TestClient, headers: dict[str, str]) -> tuple[Any, AsyncMock]:
    provider = AsyncMock(return_value=_completion())
    with patch("gateway.api.routes.chat.acompletion", new=provider):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "user": "u1"},
            headers=headers,
        )
    return response, provider


def _assert_served(client: TestClient, headers: dict[str, str]) -> None:
    response, provider = _chat(client, headers)
    assert response.status_code == 200, response.text
    assert provider.await_args is not None
    assert provider.await_args.kwargs["model"] == f"otari:{MODEL}"


def _assert_unknown(client: TestClient, headers: dict[str, str]) -> None:
    response, provider = _chat(client, headers)
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "invalid_model"
    provider.assert_not_awaited()


def _row(client: TestClient, headers: dict[str, str], provider: str = "otari-ai") -> dict[str, Any]:
    rows: list[dict[str, Any]] = client.get(f"{PATH}/{provider}/models", headers=headers).json()["data"]
    [row] = rows
    return row


def test_a_model_a_refresh_switches_on_is_reachable_by_its_catalog_id_at_once(
    client: TestClient, headers: dict[str, str]
) -> None:
    _assert_unknown(client, headers)

    refreshed = client.post(f"{PATH}/otari-ai/models/refresh", headers=headers)

    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["added"] == [MODEL]
    listed = [model["id"] for model in client.get(f"{API_ROOT}/models", headers=headers).json()["data"]]
    assert MODEL in listed
    _assert_served(client, headers)


def test_switching_a_model_off_and_on_moves_its_catalog_id_at_once(client: TestClient, headers: dict[str, str]) -> None:
    assert client.post(f"{PATH}/otari-ai/models/refresh", headers=headers).status_code == 200
    row = _row(client, headers)

    off = client.patch(f"{PATH}/otari-ai/models/{row['id']}", json={"enabled": False}, headers=headers)
    assert off.status_code == 200, off.text
    _assert_unknown(client, headers)

    on = client.post(f"{PATH}/otari-ai/models/switch", json={"vendor": row["vendor"], "enabled": True}, headers=headers)
    assert on.status_code == 200, on.text
    _assert_served(client, headers)


def test_a_provider_added_at_runtime_serves_its_models_by_catalog_id_at_once(
    client: TestClient, headers: dict[str, str]
) -> None:
    created = client.post(PATH, json={**OTARI, "name": "otari-two"}, headers=headers)

    assert created.status_code == 201, created.text
    assert _row(client, headers, "otari-two")["enabled"] is True
    _assert_served(client, headers)
