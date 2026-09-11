"""Regression test for time-to-first-token recording on the streaming path.

mozilla-ai/otari#431: the gateway already makes a first-chunk-timeout routing
decision on TTFT (``_platform.py``'s hybrid-mode fallback) but never recorded
the value it decided on. This pins the capture: a streaming request whose first
chunk is deliberately delayed records a ``ttft_ms`` that reflects that delay and
is never greater than the request's total ``latency_ms``.
"""

import asyncio
from collections.abc import AsyncIterator
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
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.entities import UsageLog

from .conftest import MODEL_NAME

_FIRST_CHUNK_DELAY_SECONDS = 0.2


def test_streaming_request_records_ttft_before_latency(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """A delayed first chunk shows up as a non-trivial ``ttft_ms`` that is
    still no greater than the request's total ``latency_ms``."""
    client.post(f"{API_ROOT}/users", json={"user_id": "ttft-user"}, headers=master_key_header)

    async def chunk_stream() -> AsyncIterator[ChatCompletionChunk]:
        await asyncio.sleep(_FIRST_CHUNK_DELAY_SECONDS)
        yield ChatCompletionChunk(
            id="chatcmpl-ttft",
            object="chat.completion.chunk",
            created=0,
            model=MODEL_NAME,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason=None)],
        )
        yield ChatCompletionChunk(
            id="chatcmpl-ttft",
            object="chat.completion.chunk",
            created=0,
            model=MODEL_NAME,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def mock_acompletion(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        return chunk_stream()

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "user": "ttft-user",
                "stream": True,
            },
            headers=master_key_header,
        )
        assert response.status_code == 200, response.text
        response.read()

    log = db_session.query(UsageLog).filter(UsageLog.user_id == "ttft-user").first()
    assert log is not None
    assert log.ttft_ms is not None
    assert log.ttft_ms >= round(_FIRST_CHUNK_DELAY_SECONDS * 1000)
    assert log.latency_ms is not None
    assert log.ttft_ms <= log.latency_ms


def test_non_streaming_request_records_no_ttft(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """A plain (non-streaming) completion has no first chunk to time."""
    client.post(f"{API_ROOT}/users", json={"user_id": "ttft-non-streaming-user"}, headers=master_key_header)

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            id="chatcmpl-no-ttft",
            object="chat.completion",
            created=0,
            model=MODEL_NAME,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hi"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "user": "ttft-non-streaming-user",
            },
            headers=master_key_header,
        )
        assert response.status_code == 200, response.text

    log = db_session.query(UsageLog).filter(UsageLog.user_id == "ttft-non-streaming-user").first()
    assert log is not None
    assert log.ttft_ms is None
