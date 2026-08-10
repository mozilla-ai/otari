"""GET /v1/usage/in-flight reports what the gateway is serving right now.

The usage log only records requests that have settled, so a slow backend (a local
model taking 30 seconds) is invisible while it runs. These tests pin the two
halves that make the live view trustworthy: a request is registered *during* its
provider call, and the entry is always gone once the response has been sent,
including on the streaming and failure paths where the route handler returns long
before the request is over.

The mid-flight assertions read the registry directly rather than calling the
endpoint: ``TestClient`` drives the app through a portal, so issuing a second
request from inside a stubbed provider call would re-enter the running loop. The
serialization the dashboard actually consumes is covered separately, against a
seeded registry.
"""

from collections.abc import AsyncIterator
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
    CreateEmbeddingResponse,
    Embedding,
    Usage,
)
from fastapi.testclient import TestClient

from gateway.inflight import InFlightRegistry

from .conftest import MODEL_NAME

IN_FLIGHT = "/v1/usage/in-flight"
_MESSAGES = [{"role": "user", "content": "Hello"}]


def _registry(client: TestClient) -> InFlightRegistry:
    registry: InFlightRegistry = client.app.state.inflight  # type: ignore[attr-defined]
    return registry


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-inflight",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-inflight",
        object="chat.completion.chunk",
        created=0,
        model=MODEL_NAME,
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="hi"), finish_reason="stop")],
    )


def _chat(client: TestClient, headers: dict[str, str], **extra: Any) -> Any:
    return client.post(
        "/v1/chat/completions",
        json={"model": MODEL_NAME, "messages": _MESSAGES, **extra},
        headers=headers,
    )


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------


def test_in_flight_requires_the_master_key(client: TestClient, api_key_header: dict[str, str]) -> None:
    assert client.get(IN_FLIGHT).status_code == 401
    assert client.get(IN_FLIGHT, headers=api_key_header).status_code == 401


def test_an_idle_gateway_reports_nothing_in_flight(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.get(IN_FLIGHT, headers=master_key_header)

    assert resp.status_code == 200
    assert resp.json() == {"requests": [], "total": 0}


def test_a_tracked_request_is_serialized_for_the_dashboard(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    registry = _registry(client)
    request_id = registry.begin(
        endpoint="/v1/chat/completions",
        model="gemini-2.5-flash",
        provider="gemini",
        user_id="default",
        api_key_id="key-1",
        policy_name="cheap-first",
    )
    try:
        payload = client.get(IN_FLIGHT, headers=master_key_header).json()
    finally:
        registry.finish(request_id)

    assert payload["total"] == 1
    (entry,) = payload["requests"]
    assert entry["id"] == request_id
    assert entry["endpoint"] == "/v1/chat/completions"
    assert entry["model"] == "gemini-2.5-flash"
    assert entry["provider"] == "gemini"
    assert entry["user_id"] == "default"
    assert entry["api_key_id"] == "key-1"
    assert entry["policy_name"] == "cheap-first"
    assert entry["elapsed_ms"] >= 0
    assert entry["started_at"]


def test_the_longest_running_requests_are_the_ones_reported(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The list is what a caller is still waiting on, so it leads with the oldest."""
    registry = _registry(client)
    ids = [registry.begin(endpoint="/v1/chat/completions", model=f"m{index}") for index in range(3)]
    try:
        payload = client.get(IN_FLIGHT, headers=master_key_header).json()
    finally:
        for request_id in ids:
            registry.finish(request_id)

    assert [entry["model"] for entry in payload["requests"]] == ["m0", "m1", "m2"]
    assert payload["total"] == 3


# ---------------------------------------------------------------------------
# Registration and cleanup on the real request paths
# ---------------------------------------------------------------------------


def test_a_request_is_registered_while_its_provider_call_runs(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """The whole point: the request is visible before it has settled.

    The registry is read from inside the stubbed provider call, which is exactly
    the window an operator is staring at a spinner during.
    """
    registry = _registry(client)
    seen: list[Any] = []

    async def slow_acompletion(**_kwargs: Any) -> ChatCompletion:
        seen.append(registry.snapshot())
        return _completion()

    with patch("gateway.api.routes.chat.acompletion", side_effect=slow_acompletion):
        assert _chat(client, api_key_header).status_code == 200

    assert len(seen) == 1
    (entry,) = seen[0]
    assert entry.endpoint == "/v1/chat/completions"
    assert entry.model == "gemini-2.5-flash"
    assert entry.provider == "gemini"
    assert entry.user_id == "default"
    assert entry.api_key_id
    assert entry.policy_name is None

    # And it is gone afterwards, which is what keeps the panel from accumulating
    # a ghost of every request the gateway ever served.
    assert len(registry) == 0


def test_a_failed_request_does_not_stay_in_flight(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    with patch("gateway.api.routes.chat.acompletion", side_effect=RuntimeError("provider down")):
        assert _chat(client, api_key_header).status_code >= 400

    assert len(_registry(client)) == 0


def test_a_stream_stays_in_flight_until_its_body_is_consumed(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """A streaming response outlives its route handler.

    The handler returns as soon as the upstream stream is open, so cleanup keyed on
    the handler returning would drop the entry while the caller is still receiving
    tokens, which is the longest and most interesting part of a slow request.
    """
    registry = _registry(client)
    tracked_during_stream: list[int] = []

    async def open_stream(**_kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def chunks() -> AsyncIterator[ChatCompletionChunk]:
            tracked_during_stream.append(len(registry))
            yield _chunk()

        return chunks()

    with patch("gateway.api.routes.chat.acompletion", side_effect=open_stream):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": _MESSAGES, "stream": True},
            headers=api_key_header,
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    assert "data: " in body
    assert tracked_during_stream == [1]
    assert len(registry) == 0


def test_a_pass_through_request_is_registered_too(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """Embeddings, images, audio and friends run through their own scaffold.

    They write activity rows like any other request, and an image generation
    routinely runs longer than a completion, so the live view has to cover them or
    it silently reports only part of what the gateway is doing.
    """
    registry = _registry(client)
    seen: list[Any] = []

    async def slow_embedding(**_kwargs: Any) -> CreateEmbeddingResponse:
        seen.append(registry.snapshot())
        return CreateEmbeddingResponse(
            data=[Embedding(embedding=[0.1], index=0, object="embedding")],
            model="text-embedding-3-small",
            object="list",
            usage=Usage(prompt_tokens=10, total_tokens=10),
        )

    with patch("gateway.api.routes.embeddings.aembedding", side_effect=slow_embedding):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    (entry,) = seen[0]
    assert entry.endpoint == "/v1/embeddings"
    assert entry.model == "text-embedding-3-small"
    assert entry.provider == "openai"
    assert len(registry) == 0


@pytest.mark.parametrize(
    ("model", "expected_status"),
    [("", 400), ("nosuchprovider:nosuchmodel", 400)],
)
def test_a_refused_request_is_never_registered(
    client: TestClient,
    api_key_header: dict[str, str],
    model: str,
    expected_status: int,
) -> None:
    """A request the gateway rejected was never in progress.

    It already leaves a usage row of its own, so reporting it here would show
    dropped traffic as live work.
    """
    resp = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": _MESSAGES},
        headers=api_key_header,
    )

    assert resp.status_code == expected_status
    assert len(_registry(client)) == 0
