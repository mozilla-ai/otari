"""An ``otari-ai`` hosted provider, as a connection stores it, served directly and as a fallback candidate.

Its model ids carry the vendor (``openai/gpt-4o``), and a model its listing
does not price is priced from the vendor's models.dev entry.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.routing import RoutingConfig
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key
from gateway.types.models_dev import parse_catalog_v2

from .conftest import build_test_client

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}
PLATFORM_KEY = "tk-platform-1234"
PLATFORM_BASE = "https://api.platform.test"
SELECTOR = "otari-ai:openai/gpt-4o"


@pytest.fixture
def client(postgres_url: str, clean_database: None, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    async def _listing(impl_name: str, **_: object) -> ProviderDiscovery:
        models = [Model(id=name, object="model", created=0, owned_by=impl_name) for name in ("openai/gpt-4o", "x/y")]
        return ProviderDiscovery(provider=impl_name, models=models)

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _listing)
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        default_pricing=True,
        model_discovery=False,
        import_configured_providers=False,
        providers={"openai": {"api_key": "sk-openai"}},
        routing=RoutingConfig.model_validate(
            {"policies": {"answers": {"select": [{"default": "openai:gpt-5-mini"}], "on_failure": [SELECTOR]}}}
        ),
    )
    for test_client in build_test_client(config):
        created = test_client.post(
            f"{API_ROOT}/hosted-providers",
            json={"provider": "otari", "name": "otari-ai", "api_key": PLATFORM_KEY, "api_base": PLATFORM_BASE},
            headers=HEADERS,
        )
        assert created.status_code == 201, created.text
        user = test_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=HEADERS)
        assert user.status_code == 200, user.text
        yield test_client


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content="hi"))],
        created=0,
        model="gpt-4o",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _chat(client: TestClient, model: str, provider: AsyncMock) -> Any:
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "user": "test-user"}
    with patch("gateway.api.routes.chat.acompletion", new=provider):
        return client.post(f"{API_ROOT}/chat/completions", json=body, headers=HEADERS)


def test_a_vendor_prefixed_model_is_priced_from_the_vendor_and_switched_on(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dev = parse_catalog_v2({"openai": {"models": {"gpt-4o": {"type": "chat"}}}})
    monkeypatch.setattr(
        "gateway.services.providers._hosted_provider_service.cached_models_dev_catalog", lambda: models_dev
    )
    models = client.get(f"{API_ROOT}/hosted-providers/otari-ai/models", headers=HEADERS).json()["data"]
    by_name = {model["model"]: model for model in models}

    priced = by_name["openai/gpt-4o"]
    # Its kind is read from the vendor's entry too, so the panel does not file it under Unknown.
    assert priced["type"] == "chat"
    assert by_name["x/y"]["type"] is None
    assert priced["enabled"] is True
    assert priced["price_source"] == "defaults"
    assert priced["input_price_per_million"] == 2.5
    # A vendor models.dev does not know stays off until it is priced.
    assert by_name["x/y"]["enabled"] is False
    assert by_name["x/y"]["disabled_reason"] == "unpriced"


def test_a_fallback_candidate_dispatches_as_the_direct_request_does(client: TestClient) -> None:
    direct = AsyncMock(return_value=_completion())
    assert _chat(client, SELECTOR, direct).status_code == 200
    assert direct.await_args is not None
    expected = direct.await_args.kwargs
    assert (expected["model"], expected["api_key"], expected["api_base"]) == (
        "otari:openai/gpt-4o",
        PLATFORM_KEY,
        PLATFORM_BASE,
    )

    request = httpx.Request("POST", "http://upstream")
    failed = httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))
    chained = AsyncMock(side_effect=[failed, _completion()])
    response = _chat(client, "answers", chained)

    assert response.status_code == 200, response.text
    head, fallback = (call.kwargs for call in chained.await_args_list)
    assert head["model"] == "openai:gpt-5-mini"
    for field in ("model", "api_key", "api_base"):
        assert fallback[field] == expected[field]
