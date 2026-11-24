"""The model name a caller sent is kept on its usage rows as ``requested_model``.

A row's ``model`` is what served the request, after an alias resolved it, so the
name the caller wrote is recorded beside it, and the usage list returns both.
"""

from collections.abc import AsyncIterator, Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Usage,
)
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig

from .conftest import build_test_client

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}
USER = "requested-model-user"


@pytest.fixture
def alias_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"anthropic": {"api_key": "sk-ant"}, "openai": {"api_key": "sk-openai"}},
        aliases={"myopusmodel": "anthropic:claude-opus-4", "embedder": "openai:text-embedding-3-small"},
    )


@pytest.fixture
def client(alias_config: GatewayConfig) -> Generator[TestClient]:
    yield from build_test_client(alias_config)


@pytest.fixture(autouse=True)
def _user(client: TestClient) -> None:
    resp = client.post(f"{API_ROOT}/users", json={"user_id": USER}, headers=HEADERS)
    assert resp.status_code == 200, resp.text


def _rows(client: TestClient) -> list[dict[str, Any]]:
    resp = client.get(f"{API_ROOT}/usage", params={"user_id": USER}, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    rows: list[dict[str, Any]] = resp.json()
    return rows


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-requested",
        object="chat.completion",
        created=0,
        model="claude-opus-4",
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat(client: TestClient, model: str, **extra: Any) -> Any:
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": USER, **extra},
        headers=HEADERS,
    )


def test_an_aliased_completion_records_the_alias_beside_the_model_that_served(client: TestClient) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion())):
        resp = _chat(client, "myopusmodel")
    assert resp.status_code == 200, resp.text

    (row,) = _rows(client)
    assert row["requested_model"] == "myopusmodel"
    assert row["model"] == "claude-opus-4"
    assert row["provider"] == "anthropic"


def test_a_plain_selector_is_recorded_as_sent(client: TestClient) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion())):
        resp = _chat(client, "anthropic:claude-opus-4")
    assert resp.status_code == 200, resp.text

    (row,) = _rows(client)
    assert row["requested_model"] == "anthropic:claude-opus-4"


def test_a_streamed_completion_records_the_alias(client: TestClient) -> None:
    async def chunks() -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            id="chatcmpl-requested",
            object="chat.completion.chunk",
            created=0,
            model="claude-opus-4",
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason=None)],
        )
        yield ChatCompletionChunk(
            id="chatcmpl-requested",
            object="chat.completion.chunk",
            created=0,
            model="claude-opus-4",
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def stream(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        return chunks()

    with patch("gateway.api.routes.chat.acompletion", new=stream):
        resp = _chat(client, "myopusmodel", stream=True)
        assert resp.status_code == 200, resp.text
        resp.read()

    (row,) = _rows(client)
    assert row["status"] == "success"
    assert row["requested_model"] == "myopusmodel"


def test_a_failed_completion_keeps_the_name_it_was_sent_under(client: TestClient) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=RuntimeError("upstream down"))):
        resp = _chat(client, "myopusmodel")
    assert resp.status_code >= 500

    (row,) = _rows(client)
    assert row["status"] == "error"
    assert row["requested_model"] == "myopusmodel"


def test_a_gateway_refusal_keeps_the_name_it_was_sent_under(client: TestClient, alias_config: GatewayConfig) -> None:
    # The app holds this config object, so the pricing gate reads the flipped flag.
    alias_config.require_pricing = True
    resp = _chat(client, "myopusmodel")
    assert resp.status_code == 402, resp.text

    (row,) = _rows(client)
    assert row["status"] == "error"
    assert row["status_code"] == 402
    assert row["requested_model"] == "myopusmodel"


def test_a_passthrough_request_records_the_alias(client: TestClient) -> None:
    result = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )
    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=result):
        resp = client.post(
            f"{API_ROOT}/embeddings", json={"model": "embedder", "input": "hello", "user": USER}, headers=HEADERS
        )
    assert resp.status_code == 200, resp.text

    (row,) = _rows(client)
    assert row["requested_model"] == "embedder"
    assert row["model"] == "text-embedding-3-small"
