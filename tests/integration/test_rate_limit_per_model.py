"""``per: model`` rate limits with ``router: priority``, end to end through the chat completions route.

The requirement this covers: model 1 takes every request until its limit, the
rest spill to model 2, and a caller sees a 429 only when no candidate has room.
"""

from collections.abc import AsyncIterator, Generator
from typing import Any
from unittest.mock import patch

import httpx
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

from gateway.core.config import (
    API_KEY_HEADER,
    API_ROOT,
    ATTEMPT_COUNT_HEADER,
    FALLBACK_HEADER,
    PROVIDER_HEADER,
    GatewayConfig,
    RateLimitRule,
)
from gateway.models.routing import RoutingConfig

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
FIRST = "openai:gpt-4o-mini"
SECOND = "anthropic:claude-sonnet-4-5"
BACKUP = "openai:gpt-4o"
USER = "spill-user"


def _client(postgres_url: str, rules: list[dict[str, Any]]) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}, "anthropic": {"api_key": "sk-ant"}},
        rate_limits=[RateLimitRule(**rule) for rule in rules],
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "spill": {
                        "select": [{"router": "priority", "candidates": [FIRST, SECOND]}, {"default": SECOND}],
                        "on_failure": [BACKUP],
                    }
                }
            }
        ),
    )
    client_gen = build_test_client(config)
    client = next(client_gen)
    try:
        assert client.post(f"{API_ROOT}/users", json={"user_id": USER}, headers=_MASTER).status_code == 200
        yield client
    finally:
        client_gen.close()


@pytest.fixture
def first_capped(postgres_url: str) -> Generator[TestClient]:
    yield from _client(postgres_url, [{"name": "first-cap", "per": "model", "models": [FIRST], "rpm": 2}])


@pytest.fixture
def all_capped(postgres_url: str) -> Generator[TestClient]:
    yield from _client(postgres_url, [{"name": "cap", "per": "model", "models": [FIRST, SECOND, BACKUP], "rpm": 1}])


def _completion(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    )


def _chat(client: TestClient, model: str, *, fail: set[str] | None = None) -> tuple[Any, list[str]]:
    """POST a chat request, returning the response and the models the provider was sent."""
    sent: list[str] = []

    async def completion(**kwargs: Any) -> ChatCompletion:
        sent.append(kwargs["model"])
        if fail and kwargs["model"] in fail:
            request = httpx.Request("POST", "http://upstream")
            raise httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))
        return _completion(kwargs["model"])

    with patch("gateway.api.routes.chat.acompletion", new=completion):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": USER},
            headers=_MASTER,
        )
    return response, sent


def test_traffic_stays_on_the_first_model_until_its_limit_then_spills(first_capped: TestClient) -> None:
    results = [_chat(first_capped, "spill") for _ in range(4)]

    assert [response.status_code for response, _ in results] == [200, 200, 200, 200]
    assert [sent for _, sent in results] == [[FIRST], [FIRST], [SECOND], [SECOND]]
    assert all(response.json()["model"] == "spill" for response, _ in results)


def _usage(client: TestClient) -> list[dict[str, Any]]:
    response = client.get(f"{API_ROOT}/usage", params={"limit": 50}, headers=_MASTER)
    assert response.status_code == 200, response.text
    return list(response.json())


def test_a_spilled_request_records_the_model_it_skipped(first_capped: TestClient) -> None:
    for _ in range(3):
        _chat(first_capped, "spill")

    rows = _usage(first_capped)
    spilled_group = next(row["request_group_id"] for row in rows if row["model"] == "claude-sonnet-4-5")
    group = sorted(
        (row for row in rows if row["request_group_id"] == spilled_group), key=lambda row: row["attempt_position"]
    )

    assert [(row["provider"], row["model"], row["status"]) for row in group] == [
        ("openai", "gpt-4o-mini", "absorbed"),
        ("anthropic", "claude-sonnet-4-5", "success"),
    ]
    assert group[0]["status_code"] == 429
    assert group[0]["error_message"] == "Skipped: Rate limit 'first-cap' exceeded: 2 requests per minute"


def test_a_spilled_request_is_a_fallback_sent_to_one_candidate(first_capped: TestClient) -> None:
    for _ in range(2):
        _chat(first_capped, "spill")

    response, sent = _chat(first_capped, "spill")

    assert sent == [SECOND]
    assert response.headers[PROVIDER_HEADER] == "anthropic"
    # The full model was skipped without being called, so one candidate was sent the request.
    assert response.headers[ATTEMPT_COUNT_HEADER] == "1"
    assert response.headers[FALLBACK_HEADER] == "true"


def test_a_direct_call_to_a_full_model_is_refused(first_capped: TestClient) -> None:
    for _ in range(2):
        _chat(first_capped, "spill")

    response, sent = _chat(first_capped, FIRST)

    assert response.status_code == 429
    assert response.json()["detail"] == "Rate limit 'first-cap' for openai:gpt-4o-mini exceeded: 2 requests per minute"
    assert "Retry-After" in response.headers
    assert sent == []


def test_the_limit_is_shared_by_a_direct_call_and_the_policy(first_capped: TestClient) -> None:
    """One count per model, however a request reaches it."""
    _chat(first_capped, FIRST)
    _chat(first_capped, FIRST)

    _, sent = _chat(first_capped, "spill")

    assert sent == [SECOND]


def test_a_failure_on_a_model_with_room_still_falls_through(first_capped: TestClient) -> None:
    response, sent = _chat(first_capped, "spill", fail={FIRST})

    assert response.status_code == 200
    assert sent == [FIRST, SECOND]


def test_a_full_model_is_skipped_in_on_failure_too(all_capped: TestClient) -> None:
    _chat(all_capped, BACKUP)

    response, sent = _chat(all_capped, "spill", fail={FIRST, SECOND})

    assert sent == [FIRST, SECOND]
    assert response.status_code == 502


def test_a_caller_is_refused_only_when_every_candidate_is_full(all_capped: TestClient) -> None:
    results = [_chat(all_capped, "spill") for _ in range(4)]

    assert [response.status_code for response, _ in results] == [200, 200, 200, 429]
    assert [sent for _, sent in results] == [[FIRST], [SECOND], [BACKUP], []]
    # Through a policy the refusal names the rule, not the policy's target.
    assert results[3][0].json()["detail"] == "Rate limit 'cap' exceeded: 1 request per minute"


def test_a_streamed_request_spills_the_same_way(first_capped: TestClient) -> None:
    sent: list[str] = []

    async def open_stream(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        sent.append(kwargs["model"])

        async def chunks() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk(
                id="chatcmpl-test",
                object="chat.completion.chunk",
                created=0,
                model=kwargs["model"],
                choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="hi"), finish_reason="stop")],
                usage=CompletionUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
            )

        return chunks()

    with patch("gateway.api.routes.chat.acompletion", side_effect=open_stream):
        for _ in range(3):
            with first_capped.stream(
                "POST",
                f"{API_ROOT}/chat/completions",
                json={
                    "model": "spill",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "user": USER,
                },
                headers=_MASTER,
            ) as response:
                "".join(response.iter_text())
                assert response.status_code == 200

    assert sent == [FIRST, FIRST, SECOND]
