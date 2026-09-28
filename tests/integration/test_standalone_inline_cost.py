"""A standalone response names its request and carries the cost it settled at.

The gateway prices every standalone request into its own usage row. These pin
that the same amount reaches the caller on the response, under the fields a
hybrid response uses, together with a request id, so a client that bills per
request needs neither a second lookup nor a price table of its own.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from decimal import Decimal
from typing import Any
from unittest.mock import patch

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
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, REQUEST_ID_HEADER
from gateway.models.usage import UsageLog

from .conftest import MODEL_NAME

# $1.00 per million tokens each way, so 100k in plus 200k out settles at $0.30.
_USAGE = CompletionUsage(prompt_tokens=100_000, completion_tokens=200_000, total_tokens=300_000)


def _price_model(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 1.0, "output_price_per_million": 1.0},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-inline",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=_USAGE,
    )


def _chunks() -> list[ChatCompletionChunk]:
    return [
        ChatCompletionChunk(
            id="chunk-1",
            object="chat.completion.chunk",
            created=0,
            model=MODEL_NAME,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason="stop")],
        ),
        ChatCompletionChunk(
            id="chunk-1",
            object="chat.completion.chunk",
            created=0,
            model=MODEL_NAME,
            choices=[],
            usage=_USAGE,
        ),
    ]


def _chat(client: TestClient, headers: dict[str, str], *, stream: bool = False) -> Any:
    async def _acompletion(**_kwargs: Any) -> Any:
        if not stream:
            return _completion()

        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in _chunks():
                yield chunk

        return _stream()

    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _acompletion
        return client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "user": "inline-cost-user",
                "stream": stream,
            },
            headers=headers,
        )


def _logged_cost(make_session: Callable[[], Session], *, timeout: float = 3.0) -> Decimal | None:
    """The request's row cost, polled with a fresh session so a background log writer cannot race the read."""
    deadline = time.monotonic() + timeout
    while True:
        with make_session() as db:
            row = db.execute(select(UsageLog).where(UsageLog.user_id == "inline-cost-user")).scalar_one_or_none()
            if row is not None:
                return row.cost
        assert time.monotonic() < deadline, "the usage row was never written"
        time.sleep(0.1)


def _create_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(f"{API_ROOT}/users", json={"user_id": "inline-cost-user"}, headers=master_key_header)
    assert response.status_code == 200, response.text


def test_non_streaming_response_carries_its_request_id_and_settled_cost(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    _price_model(client, master_key_header)
    _create_user(client, master_key_header)

    response = _chat(client, master_key_header)

    assert response.status_code == 200, response.text
    uuid.UUID(response.headers[REQUEST_ID_HEADER])
    usage = response.json()["usage"]
    assert usage["cost_usd"] == "0.300000"
    assert usage["pricing_source"] == "deployment"
    assert _logged_cost(db_session_factory) == Decimal(usage["cost_usd"])


def test_streaming_response_carries_cost_on_its_terminal_usage_chunk(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    _price_model(client, master_key_header)
    _create_user(client, master_key_header)

    response = _chat(client, master_key_header, stream=True)

    assert response.status_code == 200, response.text
    uuid.UUID(response.headers[REQUEST_ID_HEADER])
    frames = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    usages = [json.loads(frame)["usage"] for frame in frames[:-1] if json.loads(frame).get("usage")]
    assert len(usages) == 1
    assert usages[0]["cost_usd"] == "0.300000"
    assert usages[0]["pricing_source"] == "deployment"
    assert _logged_cost(db_session_factory) == Decimal("0.300000")


def test_unpriced_model_response_has_a_request_id_and_no_cost(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    _create_user(client, master_key_header)

    response = _chat(client, master_key_header)

    assert response.status_code == 200, response.text
    uuid.UUID(response.headers[REQUEST_ID_HEADER])
    usage = response.json()["usage"]
    assert "cost_usd" not in usage
    assert "pricing_source" not in usage


def test_each_request_gets_its_own_request_id(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    _create_user(client, master_key_header)

    first = _chat(client, master_key_header)
    second = _chat(client, master_key_header)

    assert first.headers[REQUEST_ID_HEADER] != second.headers[REQUEST_ID_HEADER]


def _settlement(client: TestClient, request_id: str, headers: dict[str, str], *, timeout: float = 3.0) -> Any:
    """The request's settlement, polled because the background log writer can lag the response."""
    deadline = time.monotonic() + timeout
    while True:
        response = client.get(f"{API_ROOT}/usage/requests/{request_id}", headers=headers)
        if response.status_code != 404 or time.monotonic() >= deadline:
            return response
        time.sleep(0.1)


def test_a_stream_that_fails_after_reporting_usage_can_be_looked_up_by_request_id(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The caller never receives a cost on a failed stream, so the id is how it recovers one."""
    _price_model(client, master_key_header)
    _create_user(client, master_key_header)

    async def _acompletion(**_kwargs: Any) -> Any:
        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            yield _chunks()[1]
            raise RuntimeError("upstream broke")

        return _stream()

    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _acompletion
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "user": "inline-cost-user",
                "stream": True,
            },
            headers=master_key_header,
        )

    assert response.status_code == 200, response.text
    assert "An error occurred during streaming" in response.text
    settled = _settlement(client, response.headers[REQUEST_ID_HEADER], master_key_header)
    assert settled.status_code == 200, settled.text
    body = settled.json()
    assert body["status"] == "error"
    assert body["cost_usd"] == "0.300000"
    assert body["total_tokens"] == 300_000
    assert body["row_count"] == 1


def test_a_successful_request_settles_to_the_cost_its_response_carried(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    _price_model(client, master_key_header)
    _create_user(client, master_key_header)

    response = _chat(client, master_key_header)

    request_id = response.headers[REQUEST_ID_HEADER]
    settled = _settlement(client, request_id, master_key_header)
    assert settled.status_code == 200, settled.text
    assert settled.json() == {
        "request_id": request_id,
        "status": "success",
        "cost_usd": response.json()["usage"]["cost_usd"],
        "prompt_tokens": 100_000,
        "completion_tokens": 200_000,
        "total_tokens": 300_000,
        "row_count": 1,
    }


def test_an_unpriced_request_settles_with_no_cost(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    _create_user(client, master_key_header)

    response = _chat(client, master_key_header)

    settled = _settlement(client, response.headers[REQUEST_ID_HEADER], master_key_header)
    assert settled.status_code == 200, settled.text
    assert settled.json()["cost_usd"] is None


def test_an_api_key_sees_its_own_requests_and_no_other_keys(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
) -> None:
    _price_model(client, master_key_header)
    other = client.post(f"{API_ROOT}/keys", json={"key_name": "other-key"}, headers=master_key_header)
    assert other.status_code == 200, other.text
    other_header = {name: f"Bearer {other.json()['key']}" for name in api_key_header}

    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.return_value = _completion()
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
            headers=api_key_header,
        )
    assert response.status_code == 200, response.text
    request_id = response.headers[REQUEST_ID_HEADER]

    own = _settlement(client, request_id, api_key_header)
    assert own.status_code == 200, own.text
    assert own.json()["cost_usd"] == "0.300000"
    assert client.get(f"{API_ROOT}/usage/requests/{request_id}", headers=other_header).status_code == 404


def test_an_unknown_request_id_is_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.get(f"{API_ROOT}/usage/requests/{uuid.uuid4()}", headers=master_key_header)

    assert response.status_code == 404
