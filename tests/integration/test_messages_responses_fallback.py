"""Route tests for /api/v1/messages falling back to OpenAI's Responses API (otari#1630).

OpenAI's Chat Completions, which any-llm bridges Messages through, refuses
function tools with a reasoning effort on some models. The route must answer
such a request through Responses, bill it like any other, and leave every other
provider failure on its existing path.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import patch

import httpx
import openai
from fastapi.testclient import TestClient
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseOutputItemAddedEvent,
)
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.usage import UsageLog

MODEL = "openai:gpt-6-luna"
REFUSAL = (
    "Function tools with reasoning_effort are not supported for gpt-6-luna in /v1/chat/completions. "
    "To use function tools, use /v1/responses or set reasoning_effort to 'none'."
)
_FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": '{"city": "Paris"}',
    "status": "completed",
}


def _bad_request(message: str = REFUSAL, param: str = "reasoning_effort") -> openai.BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(
        message,
        response=httpx.Response(400, request=request),
        body={"message": message, "type": "invalid_request_error", "param": param, "code": None},
    )


def _refusing(exc: BaseException) -> Any:
    async def fake_amessages(**_kwargs: Any) -> Any:
        raise exc

    return fake_amessages


def _response(output: list[dict[str, Any]], *, status: str = "completed", usage: bool = True) -> Response:
    return Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "gpt-6-luna",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": status,
            "usage": {
                "input_tokens": 50,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 8},
                "total_tokens": 62,
            }
            if usage
            else None,
        }
    )


def _body(user_id: str, *, stream: bool = False) -> dict[str, Any]:
    return {
        "model": MODEL,
        "max_tokens": 256,
        "stream": stream,
        "metadata": {"user_id": user_id},
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Weather for a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "thinking": {"type": "enabled", "budget_tokens": 4096},
    }


def _seed_priced_user(client: TestClient, headers: dict[str, str], user_id: str) -> None:
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 100.0}, headers=headers)
    assert budget.status_code == 200, budget.text
    created = client.post(
        f"{API_ROOT}/users", json={"user_id": user_id, "budget_id": budget.json()["budget_id"]}, headers=headers
    )
    assert created.status_code == 200, created.text
    priced = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL, "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=headers,
    )
    assert priced.status_code == 200, priced.text


def _poll_usage_row(make_session: Callable[[], Session], user_id: str) -> UsageLog | None:
    deadline = time.time() + 3.0
    while True:
        db = make_session()
        try:
            row = db.query(UsageLog).filter(UsageLog.user_id == user_id).first()
            if row is not None or time.time() > deadline:
                return row
        finally:
            db.close()
        time.sleep(0.1)


def test_tools_with_reasoning_are_served_through_responses(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    user_id = "responses-fallback"
    _seed_priced_user(client, master_key_header, user_id)
    captured: dict[str, Any] = {}

    async def fake_aresponses(**kwargs: Any) -> Response:
        captured.update(kwargs)
        return _response([_FUNCTION_CALL])

    with (
        patch("gateway.api.routes.messages.amessages", new=_refusing(_bad_request())),
        patch("gateway.services.providers.messages_via_responses.aresponses", new=fake_aresponses),
    ):
        resp = client.post(f"{API_ROOT}/messages", json=_body(user_id), headers=master_key_header)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}},
    ]
    assert captured["reasoning"] == {"effort": "medium"}
    assert captured["tools"][0]["name"] == "get_weather"

    row = _poll_usage_row(db_session_factory, user_id)
    assert row is not None
    assert row.status == "success"
    assert (row.prompt_tokens, row.completion_tokens) == (50, 12)
    assert row.cost is not None and row.cost > 0


def test_a_streamed_request_is_served_through_the_responses_stream(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    user_id = "responses-fallback-stream"
    _seed_priced_user(client, master_key_header, user_id)

    async def fake_aresponses(**kwargs: Any) -> AsyncIterator[Any]:
        assert kwargs["stream"] is True

        async def events() -> AsyncIterator[Any]:
            yield ResponseCreatedEvent(
                type="response.created", response=_response([], status="in_progress", usage=False), sequence_number=0
            )
            yield ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                output_index=0,
                sequence_number=1,
                item={**_FUNCTION_CALL, "arguments": "", "status": "in_progress"},  # type: ignore[arg-type]
            )
            yield ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                output_index=0,
                item_id="fc_1",
                delta='{"city": "Paris"}',
                sequence_number=2,
            )
            yield ResponseCompletedEvent(
                type="response.completed", response=_response([_FUNCTION_CALL]), sequence_number=3
            )

        return events()

    with (
        patch("gateway.api.routes.messages.amessages", new=_refusing(_bad_request())),
        patch("gateway.services.providers.messages_via_responses.aresponses", new=fake_aresponses),
    ):
        resp = client.post(f"{API_ROOT}/messages", json=_body(user_id, stream=True), headers=master_key_header)
        assert resp.status_code == 200, resp.text
        text = resp.text

    payloads = [json.loads(line[len("data: ") :]) for line in text.splitlines() if line.startswith("data: ")]
    # The gateway's own trailing ``done`` frame carries no ``type``.
    events = [payload for payload in payloads if "type" in payload]
    types = [event["type"] for event in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    starts = [event for event in events if event["type"] == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "tool_use"
    delta = next(event for event in events if event["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"

    row = _poll_usage_row(db_session_factory, user_id)
    assert row is not None
    assert row.status == "success"
    assert (row.prompt_tokens, row.completion_tokens) == (50, 12)


def test_an_unrelated_rejection_keeps_the_chat_completions_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    async def fake_aresponses(**_kwargs: Any) -> Response:
        raise AssertionError("must not fall back")

    with (
        patch(
            "gateway.api.routes.messages.amessages",
            new=_refusing(_bad_request(message="Invalid schema for function 'get_weather'", param="tools")),
        ),
        patch("gateway.services.providers.messages_via_responses.aresponses", new=fake_aresponses),
    ):
        body = _body("unused")
        del body["metadata"]
        resp = client.post(f"{API_ROOT}/messages", json=body, headers=api_key_header)

    assert resp.status_code == 400, resp.text
    error = resp.json()["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert "Invalid schema" in error["message"]
