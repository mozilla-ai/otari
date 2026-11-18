"""A provider instance's ``request_fields``: extra chat fields a caller may send that provider, and only it.

A field a chat request carries beyond its schema is dropped before dispatch,
unless the instance the request reaches lists it, as an OpenAI-compatible Exa
instance lists Exa's ``text``. Then it goes to that provider in ``extra_body``.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.routing import RoutingConfig

from .conftest import build_test_client

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}


@pytest.fixture
def client(postgres_url: str) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={
            "exa": {"provider_type": "openai", "api_key": "exa-key", "request_fields": ["text"]},
            "openai": {"api_key": "sk-openai"},
        },
        routing=RoutingConfig.model_validate(
            {"policies": {"answers": {"select": [{"default": "exa:exa"}], "on_failure": ["openai:gpt-5-mini"]}}}
        ),
    )
    yield from build_test_client(config)


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content="hi"))],
        created=0,
        model="exa",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _chat(client: TestClient, model: str, provider: AsyncMock) -> Any:
    assert client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=HEADERS).status_code == 200
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "user": "test-user",
        "text": True,
        "not_listed": "anywhere",
    }
    with patch("gateway.api.routes.chat.acompletion", new=provider):
        return client.post(f"{API_ROOT}/chat/completions", json=body, headers=HEADERS)


def test_a_listed_field_reaches_its_provider_in_extra_body(client: TestClient) -> None:
    provider = AsyncMock(return_value=_completion())

    response = _chat(client, "exa:exa", provider)

    assert response.status_code == 200, response.text
    assert provider.await_args is not None
    kwargs = provider.await_args.kwargs
    assert kwargs["extra_body"] == {"text": True}
    assert "text" not in kwargs
    assert "not_listed" not in kwargs


def test_an_instance_listing_nothing_gets_no_extra_field(client: TestClient) -> None:
    provider = AsyncMock(return_value=_completion())

    response = _chat(client, "openai:gpt-5-mini", provider)

    assert response.status_code == 200, response.text
    assert provider.await_args is not None
    kwargs = provider.await_args.kwargs
    assert "extra_body" not in kwargs
    assert "text" not in kwargs


def test_a_fallback_candidate_gets_only_its_own_listed_fields(client: TestClient) -> None:
    request = httpx.Request("POST", "http://upstream")
    failed = httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))
    provider = AsyncMock(side_effect=[failed, _completion()])

    response = _chat(client, "answers", provider)

    assert response.status_code == 200, response.text
    first, second = (call.kwargs for call in provider.await_args_list)
    assert first["extra_body"] == {"text": True}
    assert "extra_body" not in second
