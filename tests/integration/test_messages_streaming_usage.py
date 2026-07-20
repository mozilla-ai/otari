"""Regression test for streaming /v1/messages token + cost metering.

Reproduces the surface of mozilla-ai/otari#256: a streaming ``/v1/messages``
request whose stream completes cleanly must record the request's tokens and
cost in the usage log, the same as the non-streaming path and as streaming
``/v1/chat/completions``. The undercount originated in any-llm (the messages
bridge did not request usage on streaming, so the translated Anthropic events
carried zero), fixed upstream in any-llm 1.21.0; otari's settlement path
(``_messages_stream_usage`` -> ``streaming_generator``) already merges the
usage from those events. This test locks otari's side: given a streaming
messages response that carries usage, the ``UsageLog`` row is non-zero.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any, cast
from unittest.mock import patch

from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    MessageUsage,
    TextBlock,
    TextDelta,
)
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import UsageLog

from .conftest import MODEL_NAME

_PRICING = {"input_price_per_million": 2.5, "output_price_per_million": 10.0}
_INPUT_TOKENS = 42
_OUTPUT_TOKENS = 7


def _seed_budgeted_user(client: TestClient, headers: dict[str, str], user_id: str) -> None:
    budget = client.post("/v1/budgets", json={"max_budget": 100.0}, headers=headers)
    assert budget.status_code == 200
    budget_id = budget.json()["budget_id"]
    created = client.post(
        "/v1/users",
        json={"user_id": user_id, "budget_id": budget_id},
        headers=headers,
    )
    assert created.status_code == 200


def _configure_pricing(client: TestClient, headers: dict[str, str], model_key: str) -> None:
    res = client.post("/v1/pricing", json={"model_key": model_key, **_PRICING}, headers=headers)
    assert res.status_code == 200


def _message_start() -> MessageStartEvent:
    return MessageStartEvent(
        type="message_start",
        message=MessageResponse(
            id="msg_test",
            type="message",
            role="assistant",
            model="claude-3-5-sonnet-20241022",
            content=[TextBlock(type="text", text="", citations=None)],
            stop_reason=None,
            stop_sequence=None,
            usage=MessageUsage(
                input_tokens=_INPUT_TOKENS,
                output_tokens=0,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
                cache_creation=None,
                server_tool_use=None,
                service_tier=None,
            ),
            container=None,
        ),
    )


def _text_delta(text: str) -> ContentBlockDeltaEvent:
    return ContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=cast(Any, TextDelta(type="text_delta", text=text)),
    )


def _message_delta() -> MessageDeltaEvent:
    # The trailing usage-bearing event any-llm now emits on streaming (#256).
    return MessageDeltaEvent(
        type="message_delta",
        delta=MessageDelta(stop_reason=cast(Any, "end_turn"), stop_sequence=None),
        usage=MessageDeltaUsage(
            input_tokens=None,
            output_tokens=_OUTPUT_TOKENS,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            server_tool_use=None,
        ),
    )


async def _stream_with_usage(**_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
    async def _gen() -> AsyncIterator[MessageStreamEvent]:
        yield _message_start()
        yield _text_delta("hello")
        yield _message_delta()
        yield MessageStopEvent(type="message_stop")

    return _gen()


def _poll_usage_row(
    make_session: Callable[[], Session], user_id: str, *, timeout: float = 3.0
) -> UsageLog | None:
    deadline = time.time() + timeout
    while True:
        db = make_session()
        try:
            row = db.query(UsageLog).filter(UsageLog.user_id == user_id).first()
            if row is not None or time.time() > deadline:
                return row
        finally:
            db.close()
        time.sleep(0.1)


def test_messages_streaming_records_tokens_and_cost(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    user_id = "stream-usage-messages"
    _seed_budgeted_user(client, master_key_header, user_id)
    _configure_pricing(client, master_key_header, MODEL_NAME)

    with patch("gateway.api.routes.messages.amessages", new=_stream_with_usage):
        response = client.post(
            "/v1/messages",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
                "stream": True,
                "metadata": {"user_id": user_id},
            },
            headers=master_key_header,
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        # Drain the stream so settlement (on_complete) runs.
        body = response.text

    assert "message_stop" in body

    row = _poll_usage_row(db_session_factory, user_id)
    assert row is not None, "streaming /v1/messages must record a usage row"
    assert row.status == "success"
    assert row.prompt_tokens == _INPUT_TOKENS
    assert row.completion_tokens == _OUTPUT_TOKENS
    assert row.cost is not None and row.cost > 0.0
