"""The request path with tracing on: a client's malformed body, a stream that fails late, and full content."""

import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
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
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, REQUEST_ID_HEADER, GatewayConfig
from gateway.models.traces import TraceSpan
from gateway.models.users import User
from gateway.services.secret_box import generate_secret_key

from .conftest import MODEL_NAME

# A replayed tool call whose ``function`` is a string rather than an object.
_MALFORMED = [
    {"role": "user", "content": "x"},
    {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": "Bash"}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
]


@pytest.fixture
def test_config(test_config: GatewayConfig, tmp_path: Path) -> GatewayConfig:
    return test_config.model_copy(update={"trace_content_capture_max": "full", "files_local_dir": str(tmp_path)})


@pytest.fixture(autouse=True)
def secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _completion(content: str = "the answer is 4") -> ChatCompletion:
    return ChatCompletion(
        id="c",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(index=0, message=ChatCompletionMessage(role="assistant", content=content), finish_reason="stop")
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _capture(client: TestClient, headers: dict[str, str], api_key_obj: dict[str, Any], level: str) -> None:
    workspace_id = client.get(f"{API_ROOT}/keys/{api_key_obj['id']}", headers=headers).json()["workspace_id"]
    response = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/trace-settings", json={"content_capture": level}, headers=headers
    )
    assert response.status_code == 200, response.text


def _spans(factory: Callable[[], Session], request_id: str, count: int) -> list[TraceSpan]:
    deadline = time.monotonic() + 10
    while True:
        with factory() as session:
            spans = list(session.scalars(select(TraceSpan).where(TraceSpan.request_id == request_id)))
        if len(spans) >= count or time.monotonic() > deadline:
            return spans
        time.sleep(0.1)


def test_a_malformed_body_without_credentials_is_refused_as_unauthenticated(client: TestClient) -> None:
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": MODEL_NAME, "messages": [{"role": "assistant", "tool_calls": 5}]},
    )

    assert response.status_code == 401


def test_a_malformed_body_with_capture_on_is_served_and_leaves_no_hold(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    _capture(client, master_key_header, api_key_obj, "full")

    with patch("gateway.api.routes.chat.acompletion", return_value=_completion()):
        response = client.post(
            f"{API_ROOT}/chat/completions", json={"model": MODEL_NAME, "messages": _MALFORMED}, headers=api_key_header
        )

    assert response.status_code == 200, response.text
    with db_session_factory() as session:
        assert session.scalar(select(func.coalesce(func.sum(User.reserved), 0))) == 0


def test_full_capture_keeps_the_answer_this_request_got(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    _capture(client, master_key_header, api_key_obj, "full")

    with patch("gateway.api.routes.chat.acompletion", return_value=_completion("the answer is 4")):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "what is 2+2"}]},
            headers=api_key_header,
        )
    request_id = response.headers[REQUEST_ID_HEADER]
    [step] = [span for span in _spans(db_session_factory, request_id, 2) if span.kind == "step"]

    read = client.post(
        f"{API_ROOT}/traces/{step.trace_id}/spans/{request_id}/content/break-glass",
        json={"reason": "Integration test of the content path"},
        headers=master_key_header,
    )

    assert read.status_code == 200, read.text
    assert read.json()["fields"]["output"] == "the answer is 4"


def test_a_stream_that_fails_after_its_200_is_a_failed_step(
    client: TestClient, api_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    async def open_stream(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def chunks() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk(
                id="c",
                object="chat.completion.chunk",
                created=0,
                model=MODEL_NAME,
                choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="hi"), finish_reason=None)],
            )
            raise RuntimeError("upstream went away")

        return chunks()

    with patch("gateway.api.routes.chat.acompletion", side_effect=open_stream):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "x"}], "stream": True},
            headers=api_key_header,
        )
        response.read()
    assert response.status_code == 200

    spans = _spans(db_session_factory, response.headers[REQUEST_ID_HEADER], 2)
    [step] = [span for span in spans if span.kind == "step"]

    assert step.outcome == "error"
