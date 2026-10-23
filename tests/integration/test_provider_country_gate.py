"""A provider type's ``provider_supported_countries`` on the completion routes."""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.error_codes import COUNTRY_NOT_SUPPORTED, ERROR_CODE_HEADER
from gateway.models.routing import RoutingConfig

from .conftest import build_test_client
from .test_routing_policies import _awaited_model, _completion
from .test_upstream_user_identifier import _logged_identifiers

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}
_COUNTRY_HEADER = "CF-IPCountry"


def _client_with_alice(config: GatewayConfig) -> Generator[TestClient]:
    for client in build_test_client(config):
        resp = client.post(f"{API_ROOT}/users", json={"user_id": "alice"}, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        yield client


@pytest.fixture
def client(postgres_url: str) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}, "anthropic": {"api_key": "sk-ant"}},
        provider_supported_countries={"openai": ["US"]},
        client_country_header=_COUNTRY_HEADER,
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "fast": {"select": [{"default": "openai:gpt-5-mini"}], "on_failure": ["anthropic:claude-haiku-4-5"]}
                }
            }
        ),
    )
    yield from _client_with_alice(config)


def _chat(client: TestClient, model: str, country: str | None) -> Any:
    headers = {**HEADERS, _COUNTRY_HEADER: country} if country is not None else HEADERS
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": "alice"},
        headers=headers,
    )


def test_a_listed_country_reaches_the_provider(client: TestClient) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-5-mini"))) as mock:
        resp = _chat(client, "openai:gpt-5-mini", "US")

    assert resp.status_code == 200, resp.text
    assert _awaited_model(mock) == "openai:gpt-5-mini"


@pytest.mark.parametrize("country", ["FR", None], ids=["unlisted", "unknown"])
def test_another_country_is_refused_before_the_provider(
    client: TestClient, postgres_url: str, country: str | None
) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock()) as mock:
        resp = _chat(client, "openai:gpt-5-mini", country)

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    mock.assert_not_awaited()
    rows = client.get(f"{API_ROOT}/usage", params={"status": "error"}, headers=HEADERS).json()
    assert [(row["provider"], row["status_code"]) for row in rows] == [("openai", 403)]
    # Refused after the user is resolved, so the row traces back to them like any other.
    assert _logged_identifiers(postgres_url, 1)[0] is not None


def test_a_provider_with_no_list_serves_every_country(client: TestClient) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("claude-haiku-4-5"))):
        resp = _chat(client, "anthropic:claude-haiku-4-5", "FR")

    assert resp.status_code == 200, resp.text


def test_a_policy_falls_over_to_a_provider_the_country_may_reach(client: TestClient) -> None:
    with patch(
        "gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("claude-haiku-4-5"))
    ) as mock:
        resp = _chat(client, "fast", "FR")

    assert resp.status_code == 200, resp.text
    assert _awaited_model(mock) == "anthropic:claude-haiku-4-5"


@pytest.fixture
def vision_client(postgres_url: str) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}, "ollama": {"api_base": "http://ollama:11434"}},
        provider_supported_countries={"openai": ["US"]},
        client_country_header=_COUNTRY_HEADER,
        vision_strategy="describe",
        vision_describe_model="openai:gpt-4o-mini",
    )
    yield from _client_with_alice(config)


@pytest.mark.parametrize(("country", "described"), [("US", True), ("FR", False)])
def test_the_image_describe_call_stays_within_the_gate(
    vision_client: TestClient, monkeypatch: pytest.MonkeyPatch, country: str, described: bool
) -> None:
    """A text-only model's image is captioned only where the vision model's provider serves the country."""
    describe = AsyncMock(return_value=_completion("gpt-4o-mini"))
    monkeypatch.setattr("gateway.services.vision.acompletion", describe)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("llama3"))):
        resp = vision_client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "ollama:llama3", "messages": [{"role": "user", "content": [image]}], "user": "alice"},
            headers={**HEADERS, _COUNTRY_HEADER: country},
        )

    assert resp.status_code == 200, resp.text
    assert describe.await_count == (1 if described else 0)


def test_a_passthrough_endpoint_is_refused_for_another_country(client: TestClient) -> None:
    with patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock()) as mock:
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hi", "user": "alice"},
            headers={**HEADERS, _COUNTRY_HEADER: "FR"},
        )

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    mock.assert_not_awaited()
    rows = client.get(f"{API_ROOT}/usage", params={"status": "error"}, headers=HEADERS).json()
    assert [(row["provider"], row["status_code"]) for row in rows] == [("openai", 403)]


def test_a_batch_is_refused_for_another_country(client: TestClient) -> None:
    with patch("gateway.api.routes.batches.acreate_batch", new=AsyncMock()) as mock:
        resp = client.post(
            f"{API_ROOT}/batches",
            json={
                "model": "openai:gpt-4o-mini",
                "user": "alice",
                "requests": [{"custom_id": "req-1", "body": {"messages": [{"role": "user", "content": "hi"}]}}],
            },
            headers={**HEADERS, _COUNTRY_HEADER: "FR"},
        )

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    mock.assert_not_awaited()


@pytest.fixture
def decisions_client(postgres_url: str) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        decision_providers={"openrouter": {"api_key": "or-secret"}},
        provider_supported_countries={"openrouter": ["US"]},
        client_country_header=_COUNTRY_HEADER,
    )
    yield from _client_with_alice(config)


def test_a_decision_is_refused_for_another_country(decisions_client: TestClient) -> None:
    with patch("gateway.api.routes._passthrough.request_decision", new=AsyncMock()) as mock:
        resp = decisions_client.post(
            f"{API_ROOT}/decisions",
            json={
                "model": "openrouter:some-model",
                "user": "alice",
                "state": "hi",
                "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}},
            },
            headers={**HEADERS, _COUNTRY_HEADER: "FR"},
        )

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    mock.assert_not_awaited()
