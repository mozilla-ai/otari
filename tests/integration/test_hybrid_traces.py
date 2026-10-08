"""A hybrid gateway files each request's trace under the workspace its resolve answer names."""

from collections.abc import Generator
from typing import Any, cast

import httpx
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import InstallControlPlane
from gateway.api.deps import reset_config
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.ports.trace_storage_port import TraceWrite

from .conftest import app_for

_WORKSPACE = "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"
_USER = "3fa85f64-5717-4562-b3fc-2c963f66afa6"


class _RecordingWriter:
    def __init__(self) -> None:
        self.writes: list[TraceWrite] = []

    def submit(self, trace: TraceWrite, *, truncated: int = 0) -> None:
        self.writes.append(trace)


@pytest.fixture
def hybrid_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    app = app_for(GatewayConfig(mode="hybrid", platform={"base_url": "http://platform.test/api/v1"}))
    with TestClient(app) as client:
        yield client
    reset_config()
    reset_db()


def _attempt(position: int, provider: str, model: str) -> dict[str, Any]:
    return {
        "attempt_id": f"attempt-{position}",
        "position": position,
        "provider": provider,
        "model": model,
        "api_key": "sk-platform",
        "api_base": None,
        "managed": True,
    }


def _platform(attempts: list[dict[str, Any]], *, workspace_id: str | None = _WORKSPACE) -> Any:
    async def post(url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            answer: dict[str, Any] = {
                "request_id": "req-hybrid-1",
                "fallback_enabled": len(attempts) > 1,
                "user_id": _USER,
                "attempts": attempts,
            }
            if workspace_id is not None:
                answer["workspace_id"] = workspace_id
            return httpx.Response(200, json=answer)
        return httpx.Response(200, json={"correlation_id": body["correlation_id"], "status": "completed"})

    return post


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="c",
        object="chat.completion",
        created=0,
        model="gpt-4o-mini",
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=7, total_tokens=17),
    )


def _send(client: TestClient, headers: dict[str, str] | None = None) -> Any:
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_token", **(headers or {})},
    )


def test_a_hybrid_request_is_traced_under_the_resolved_workspace_and_user(
    hybrid_client: TestClient, monkeypatch: pytest.MonkeyPatch, control_plane_transport: InstallControlPlane
) -> None:
    writer = _RecordingWriter()
    monkeypatch.setattr(cast(FastAPI, hybrid_client.app).state, "trace_writer", writer)
    control_plane_transport(_platform([_attempt(0, "openai", "gpt-4o-mini")]))
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", lambda **kwargs: _async(_completion()))

    response = _send(hybrid_client, {"Otari-Conversation-Id": "conv-1"})

    assert response.status_code == 200, response.text
    [write] = writer.writes
    assert (str(write.workspace_id), write.user_id, write.session_source) == (_WORKSPACE, _USER, "client")
    step, llm = sorted(write.spans, key=lambda span: span.kind != "step")
    assert (step.kind, step.outcome) == ("step", "ok")
    assert (llm.kind, llm.model, llm.input_tokens, llm.output_tokens, llm.cost_snapshot) == (
        "llm",
        "gpt-4o-mini",
        10,
        7,
        None,
    )


def test_a_failed_attempt_a_later_one_made_up_for_is_recovered(
    hybrid_client: TestClient, monkeypatch: pytest.MonkeyPatch, control_plane_transport: InstallControlPlane
) -> None:
    writer = _RecordingWriter()
    monkeypatch.setattr(cast(FastAPI, hybrid_client.app).state, "trace_writer", writer)
    control_plane_transport(_platform([_attempt(0, "openai", "gpt-4o-mini"), _attempt(1, "anthropic", "claude-x")]))

    async def first_fails(**kwargs: Any) -> ChatCompletion:
        if kwargs["model"].startswith("openai"):
            raise httpx.ConnectError("down")
        return _completion()

    monkeypatch.setattr("gateway.api.routes.chat.acompletion", first_fails)

    response = _send(hybrid_client)

    assert response.status_code == 200, response.text
    [write] = writer.writes
    attempts = sorted((span for span in write.spans if span.kind == "llm"), key=lambda span: span.provider or "")
    assert [(span.provider, span.outcome, span.recovered) for span in attempts] == [
        ("anthropic", "ok", False),
        ("openai", "error", True),
    ]


def test_a_resolve_answer_with_no_workspace_records_nothing(
    hybrid_client: TestClient, monkeypatch: pytest.MonkeyPatch, control_plane_transport: InstallControlPlane
) -> None:
    writer = _RecordingWriter()
    monkeypatch.setattr(cast(FastAPI, hybrid_client.app).state, "trace_writer", writer)
    control_plane_transport(_platform([_attempt(0, "openai", "gpt-4o-mini")], workspace_id=None))
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", lambda **kwargs: _async(_completion()))

    assert _send(hybrid_client).status_code == 200
    assert writer.writes == []


async def _async(value: Any) -> Any:
    return value
