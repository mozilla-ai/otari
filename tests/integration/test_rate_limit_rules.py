"""The ``rate_limits`` rules end to end, through the chat completions route."""

from collections.abc import AsyncIterator, Generator
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig, RateLimitRule
from gateway.ports.rate_limit_store_port import RateLimitStorePort

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}


def _client(postgres_url: str, rules: list[dict[str, Any]]) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        rate_limits=[RateLimitRule(**rule) for rule in rules],
    )
    yield from build_test_client(config)


@pytest.fixture
def per_key_rpm_client(postgres_url: str) -> Generator[TestClient]:
    yield from _client(postgres_url, [{"name": "keys", "per": "key", "rpm": 2}])


@pytest.fixture
def tpm_client(postgres_url: str) -> Generator[TestClient]:
    yield from _client(postgres_url, [{"name": "tpm", "per": "user", "tpm": 1000, "tpm_admission": "estimate"}])


@pytest.fixture
def used_tpm_client(postgres_url: str) -> Generator[TestClient]:
    yield from _client(postgres_url, [{"name": "tpm", "per": "user", "tpm": 1000}])


@pytest.fixture
def concurrency_client(postgres_url: str) -> Generator[TestClient]:
    yield from _client(postgres_url, [{"name": "inflight", "per": "deployment", "max_concurrent": 1}])


def _completion(total_tokens: int) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o-mini",
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=total_tokens - 1, completion_tokens=1, total_tokens=total_tokens),
    )


def _key(client: TestClient, user_id: str) -> dict[str, str]:
    assert client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=_MASTER).status_code == 200
    response = client.post(f"{API_ROOT}/keys", json={"key_name": user_id, "user_id": user_id}, headers=_MASTER)
    assert response.status_code == 200
    return {API_KEY_HEADER: f"Bearer {response.json()['key']}"}


def _chat(client: TestClient, headers: dict[str, str], **extra: Any) -> Any:
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], **extra},
        headers=headers,
    )


def test_a_per_key_rule_refuses_the_request_past_its_limit(per_key_rpm_client: TestClient) -> None:
    alice, bob = _key(per_key_rpm_client, "alice"), _key(per_key_rpm_client, "bob")

    async def completion(**kwargs: Any) -> ChatCompletion:
        return _completion(6)

    with patch("gateway.api.routes.chat.acompletion", new=completion):
        statuses = [_chat(per_key_rpm_client, alice).status_code for _ in range(3)]
        bob_status = _chat(per_key_rpm_client, bob).status_code
        refused = _chat(per_key_rpm_client, alice)

    assert statuses == [200, 200, 429]
    assert bob_status == 200
    assert refused.json()["detail"] == "Rate limit 'keys' exceeded: 2 requests per minute"
    assert "Retry-After" in refused.headers
    assert refused.headers["Otari-Error-Code"] == "rate_limited"


def test_a_completed_request_is_charged_the_tokens_it_used(tpm_client: TestClient) -> None:
    """Each request is admitted on max_tokens; without settling, the third would not fit."""
    headers = _key(tpm_client, "carol")

    async def completion(**kwargs: Any) -> ChatCompletion:
        return _completion(10)

    with patch("gateway.api.routes.chat.acompletion", new=completion):
        statuses = [_chat(tpm_client, headers, max_tokens=400).status_code for _ in range(3)]

    assert statuses == [200, 200, 200]


def test_a_failed_request_is_charged_no_tokens(tpm_client: TestClient) -> None:
    headers = _key(tpm_client, "dave")

    async def failing(**kwargs: Any) -> ChatCompletion:
        raise RuntimeError("upstream down")

    with patch("gateway.api.routes.chat.acompletion", new=failing):
        statuses = [_chat(tpm_client, headers, max_tokens=400).status_code for _ in range(3)]

    assert 429 not in statuses


def test_a_request_too_large_for_the_limit_is_refused(tpm_client: TestClient) -> None:
    headers = _key(tpm_client, "erin")

    response = _chat(tpm_client, headers, max_tokens=5000)

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail.startswith("Request needs an estimated 5,")
    assert detail.endswith("tokens; rate limit 'tpm' allows 1,000 per minute")


def test_by_default_a_large_max_tokens_is_admitted_and_what_was_used_counts(used_tpm_client: TestClient) -> None:
    """A client that always asks for far more than it uses is limited by its usage, not refused outright."""
    headers = _key(used_tpm_client, "gina")

    async def completion(**kwargs: Any) -> ChatCompletion:
        return _completion(400)

    with patch("gateway.api.routes.chat.acompletion", new=completion):
        statuses = [_chat(used_tpm_client, headers, max_tokens=5000).status_code for _ in range(4)]

    assert statuses == [200, 200, 200, 429]


def test_a_concurrency_slot_is_given_back_when_the_response_ends(concurrency_client: TestClient) -> None:
    headers = _key(concurrency_client, "frank")

    async def completion(**kwargs: Any) -> ChatCompletion:
        return _completion(6)

    with patch("gateway.api.routes.chat.acompletion", new=completion):
        statuses = [_chat(concurrency_client, headers).status_code for _ in range(3)]

    assert statuses == [200, 200, 200]


def test_a_concurrency_slot_is_given_back_when_the_request_fails(concurrency_client: TestClient) -> None:
    headers = _key(concurrency_client, "grace")

    async def failing(**kwargs: Any) -> ChatCompletion:
        raise RuntimeError("upstream down")

    with patch("gateway.api.routes.chat.acompletion", new=failing):
        statuses = [_chat(concurrency_client, headers).status_code for _ in range(3)]

    assert 429 not in statuses


def _stream_chunks(total_tokens: int) -> list[ChatCompletionChunk]:
    content = ChatCompletionChunk(
        id="chatcmpl-test",
        object="chat.completion.chunk",
        created=0,
        model="gpt-4o-mini",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="hi"), finish_reason=None)],
    )
    last = ChatCompletionChunk(
        id="chatcmpl-test",
        object="chat.completion.chunk",
        created=0,
        model="gpt-4o-mini",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=total_tokens - 1, completion_tokens=1, total_tokens=total_tokens),
    )
    return [content, last]


def _stream(client: TestClient, headers: dict[str, str], **extra: Any) -> int:
    with client.stream(
        "POST",
        f"{API_ROOT}/chat/completions",
        json={
            "model": "openai:gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            **extra,
        },
        headers=headers,
    ) as response:
        "".join(response.iter_text())
        return response.status_code


def test_a_streamed_request_is_charged_the_tokens_it_used(tpm_client: TestClient) -> None:
    headers = _key(tpm_client, "heidi")

    async def open_stream(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def chunks() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in _stream_chunks(10):
                yield chunk

        return chunks()

    with patch("gateway.api.routes.chat.acompletion", side_effect=open_stream):
        statuses = [_stream(tpm_client, headers, max_tokens=400) for _ in range(3)]

    assert statuses == [200, 200, 200]


def test_a_stream_holds_its_slot_until_the_body_ends(concurrency_client: TestClient) -> None:
    headers = _key(concurrency_client, "ivan")
    store: RateLimitStorePort = concurrency_client.app.state.rate_limit_store  # type: ignore[attr-defined]
    free_during_stream: list[bool] = []

    async def open_stream(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def chunks() -> AsyncIterator[ChatCompletionChunk]:
            lease = await store.acquire("rule:inflight:all:concurrent", 1, 30)
            free_during_stream.append(lease is not None)
            for chunk in _stream_chunks(6):
                yield chunk

        return chunks()

    with patch("gateway.api.routes.chat.acompletion", side_effect=open_stream):
        statuses = [_stream(concurrency_client, headers) for _ in range(2)]

    assert statuses == [200, 200]
    assert free_during_stream == [False, False]


def test_a_request_its_budget_refuses_is_charged_no_tokens(tpm_client: TestClient) -> None:
    """The rules admit before the budget refuses, so without settling the third would get a 429."""
    tpm_client.post(f"{API_ROOT}/users", json={"user_id": "judy", "blocked": True}, headers=_MASTER)

    statuses = [_chat(tpm_client, _MASTER, user="judy", max_tokens=400).status_code for _ in range(3)]

    assert statuses == [403, 403, 403]


def test_a_stream_that_fails_midway_is_charged_only_what_it_reported(tpm_client: TestClient) -> None:
    headers = _key(tpm_client, "kim")

    async def open_stream(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def chunks() -> AsyncIterator[ChatCompletionChunk]:
            yield _stream_chunks(10)[0]
            raise RuntimeError("upstream dropped")

        return chunks()

    with patch("gateway.api.routes.chat.acompletion", side_effect=open_stream):
        statuses = [_stream(tpm_client, headers, max_tokens=400) for _ in range(3)]

    assert statuses == [200, 200, 200]
