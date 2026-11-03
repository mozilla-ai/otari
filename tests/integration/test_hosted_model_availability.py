"""A hosted model its provider refuses is switched off and reported, and a model sent to the wrong endpoint is refused.

The provider call is mocked at the any-llm boundary. The provider's own model
listing, which says what kind each model is, is read from the discovery cache,
stubbed here so the background refresher cannot replace it mid-test.
"""

import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.mail import MailDelivery, Mailer, MailMessage
from gateway.services.model_discovery_service import ModelCache, ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

from .conftest import shared_test_config


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so nothing configured shadows a hosted one."""
    return shared_test_config(postgres_url, {})


NOT_SERVERLESS = (
    "Unable to access non-serverless model gpt-4o. Please visit https://api.together.ai/models/gpt-4o "
    "to create and start a new dedicated endpoint for the model."
)


class _UpstreamError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class _Outbox:
    def __init__(self) -> None:
        self.sent: list[tuple[str, MailMessage]] = []

    async def __call__(self, *, to: str, message: MailMessage) -> MailDelivery:
        self.sent.append((to, message))
        return MailDelivery(delivered=True, transport="console", reason=None)


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name,
            models=[Model(id=name, object="model", created=0, owned_by=impl_name) for name in ("gpt-4o", "embedder")],
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


@pytest.fixture
def listing(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """What the provider's listing says of each model, by ``instance:model``."""
    fields: dict[str, dict[str, Any]] = {}

    def _listed(_cache: ModelCache, provider: str, model_id: str) -> Model | None:
        extra = fields.get(f"{provider}:{model_id}")
        return None if extra is None else Model(id=model_id, object="model", created=0, owned_by=provider, **extra)

    monkeypatch.setattr(ModelCache, "listed_model", _listed)
    return fields


@pytest.fixture
def hosted(client: TestClient, master_key_header: dict[str, str]) -> dict[str, str]:
    """A hosted OpenAI serving ``gpt-4o`` and ``embedder``, priced, and a user to bill."""
    headers = master_key_header
    assert client.post(f"{API_ROOT}/users", json={"user_id": "u1", "alias": "u1"}, headers=headers).status_code == 200
    for model in ("gpt-4o", "embedder"):
        response = client.post(
            f"{API_ROOT}/pricing",
            json={"model_key": f"openai:{model}", "input_price_per_million": 1, "output_price_per_million": 1},
            headers=headers,
        )
        assert response.status_code == 200, response.text
    response = client.post(
        f"{API_ROOT}/hosted-providers", json={"provider": "openai", "api_key": "sk-x"}, headers=headers
    )
    assert response.status_code == 201, response.text
    return headers


def _chat(client: TestClient, headers: dict[str, str], model: str, provider_call: AsyncMock) -> Any:
    with patch("gateway.api.routes.chat.acompletion", new=provider_call):
        return client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=headers,
        )


def _row(client: TestClient, headers: dict[str, str], model: str) -> dict[str, Any]:
    rows = client.get(f"{API_ROOT}/hosted-providers/openai/models", headers=headers).json()["data"]
    return next(row for row in rows if row["model"] == model)


def _until_switched_off(client: TestClient, headers: dict[str, str], model: str) -> dict[str, Any]:
    """The switch-off runs beside the request, so it may land a moment after the answer."""
    deadline = time.monotonic() + 10
    while True:
        row = _row(client, headers, model)
        if not row["enabled"] or time.monotonic() > deadline:
            return row
        time.sleep(0.05)


def test_a_model_its_provider_refuses_is_switched_off_and_the_operators_are_told_once(
    client: TestClient, hosted: dict[str, str], test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")
    settings = client.patch(
        f"{API_ROOT}/health-checks/settings", json={"alert_emails": ["ops@example.com"]}, headers=hosted
    )
    assert settings.status_code == 200, settings.text
    refusal = AsyncMock(side_effect=_UpstreamError(400, NOT_SERVERLESS))
    outbox = _Outbox()

    with patch.object(Mailer, "send", new=outbox):
        first = _chat(client, hosted, "openai:gpt-4o", refusal)
        row = _until_switched_off(client, hosted, "gpt-4o")
        second = _chat(client, hosted, "openai:gpt-4o", refusal)
        # The alert follows the switch-off, so it is awaited while the outbox still catches it.
        deadline = time.monotonic() + 10
        while not outbox.sent and time.monotonic() < deadline:
            time.sleep(0.05)

    # The request that met the refusal is answered as before.
    assert first.status_code == 400
    assert row["enabled"] is False
    assert row["disabled_reason"] == "unavailable"
    # Off, so the next request is refused here rather than sent upstream.
    assert second.status_code != 200
    assert refusal.await_count == 1
    assert [to for to, _message in outbox.sent] == ["ops@example.com"]
    message = outbox.sent[0][1]
    assert message.subject == "Otari on otari.example.com: openai:gpt-4o switched off"
    assert "non-serverless model gpt-4o" in message.text
    assert "https://otari.example.com/#/admin/hosted-providers" in message.text
    listed = [model["id"] for model in client.get(f"{API_ROOT}/models", headers=hosted).json()["data"]]
    assert "openai/gpt-4o" not in listed
    assert "embedder" in listed


def test_an_operator_switches_an_unavailable_model_back_on_and_a_refresh_leaves_it_on(
    client: TestClient, hosted: dict[str, str]
) -> None:
    _chat(client, hosted, "openai:gpt-4o", AsyncMock(side_effect=_UpstreamError(400, NOT_SERVERLESS)))
    row = _until_switched_off(client, hosted, "gpt-4o")
    assert row["disabled_reason"] == "unavailable"

    response = client.patch(
        f"{API_ROOT}/hosted-providers/openai/models/{row['id']}", json={"enabled": True}, headers=hosted
    )
    assert response.status_code == 200, response.text
    refreshed = client.post(f"{API_ROOT}/hosted-providers/openai/models/refresh", headers=hosted)
    assert refreshed.status_code == 200, refreshed.text

    row = _row(client, hosted, "gpt-4o")
    assert row["enabled"] is True
    assert row["disabled_reason"] is None


def test_a_refresh_does_not_switch_an_unavailable_model_back_on(client: TestClient, hosted: dict[str, str]) -> None:
    _chat(client, hosted, "openai:gpt-4o", AsyncMock(side_effect=_UpstreamError(400, NOT_SERVERLESS)))
    _until_switched_off(client, hosted, "gpt-4o")

    refreshed = client.post(f"{API_ROOT}/hosted-providers/openai/models/refresh", headers=hosted)
    assert refreshed.status_code == 200, refreshed.text

    row = _row(client, hosted, "gpt-4o")
    assert row["enabled"] is False
    assert row["disabled_reason"] == "unavailable"


def test_an_embeddings_refusal_switches_the_model_off(client: TestClient, hosted: dict[str, str]) -> None:
    refusal = AsyncMock(side_effect=_UpstreamError(400, NOT_SERVERLESS.replace("gpt-4o", "embedder")))

    with patch("gateway.api.routes.embeddings.aembedding", new=refusal):
        response = client.post(
            f"{API_ROOT}/embeddings", json={"model": "openai:embedder", "input": "hi", "user": "u1"}, headers=hosted
        )

    assert response.status_code == 400
    row = _until_switched_off(client, hosted, "embedder")
    assert row["enabled"] is False
    assert row["disabled_reason"] == "unavailable"


def test_a_refusal_that_may_be_the_requests_fault_switches_nothing_off(
    client: TestClient, hosted: dict[str, str]
) -> None:
    refusal = AsyncMock(side_effect=_UpstreamError(404, "The model `gpt-4o` does not exist"))

    assert _chat(client, hosted, "openai:gpt-4o", refusal).status_code == 404
    time.sleep(0.2)

    assert _row(client, hosted, "gpt-4o")["enabled"] is True


def test_a_chat_request_for_an_embedding_model_is_refused_before_the_provider_is_called(
    client: TestClient, hosted: dict[str, str], listing: dict[str, dict[str, Any]]
) -> None:
    listing["openai:embedder"] = {"type": "embedding"}
    provider_call = AsyncMock()

    response = _chat(client, hosted, "openai:embedder", provider_call)

    assert response.status_code == 400
    assert response.headers["Otari-Error-Code"] == "wrong_model_type"
    assert response.json()["detail"] == (
        "'openai:embedder' is an embedding model, which cannot serve a chat request. Send it to /api/v1/embeddings."
    )
    provider_call.assert_not_called()


def test_an_embeddings_request_for_a_listed_chat_model_is_refused(
    client: TestClient, hosted: dict[str, str], listing: dict[str, dict[str, Any]]
) -> None:
    listing["openai:gpt-4o"] = {"type": "chat"}
    provider_call = AsyncMock()

    with patch("gateway.api.routes.embeddings.aembedding", new=provider_call):
        response = client.post(
            f"{API_ROOT}/embeddings", json={"model": "openai:gpt-4o", "input": "hi", "user": "u1"}, headers=hosted
        )

    assert response.status_code == 400
    assert response.headers["Otari-Error-Code"] == "wrong_model_type"
    assert response.json()["detail"] == "'openai:gpt-4o' is a chat model, which cannot create embeddings."
    provider_call.assert_not_called()
    # Recorded as a gateway refusal, as a chat request for an embedding model is.
    errors = client.get(f"{API_ROOT}/usage", params={"status": "error"}, headers=hosted).json()
    assert [(row["endpoint"], row["status_code"]) for row in errors] == [("/v1/embeddings", 400)]


def test_the_model_list_says_what_kind_each_model_is(
    client: TestClient, hosted: dict[str, str], listing: dict[str, dict[str, Any]]
) -> None:
    listing["openai:embedder"] = {"type": "embedding"}
    listing["openai:gpt-4o"] = {"type": "chat"}

    models = {model["id"]: model for model in client.get(f"{API_ROOT}/models", headers=hosted).json()["data"]}

    assert models["embedder"]["type"] == "embedding"
    assert models["openai/gpt-4o"]["type"] == "chat"
