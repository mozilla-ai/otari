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
    yield from build_test_client(config)


def _chat(client: TestClient, model: str, country: str | None) -> Any:
    headers = {**HEADERS, _COUNTRY_HEADER: country} if country is not None else HEADERS
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": "alice"},
        headers=headers,
    )


@pytest.fixture(autouse=True)
def _alice(client: TestClient) -> None:
    resp = client.post(f"{API_ROOT}/users", json={"user_id": "alice"}, headers=HEADERS)
    assert resp.status_code == 200, resp.text


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
