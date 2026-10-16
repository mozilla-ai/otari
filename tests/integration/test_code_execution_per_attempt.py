"""Who runs a provider's code-execution keyword when a request's candidates disagree about it.

Under ``auto``, each attempt is decided on its own: a candidate that runs the keyword
natively is sent it, and one that cannot runs the code on Otari's sandbox and answers
in the same blocks. A fallback that cannot be served that way is skipped rather than
refusing the request. The pure decision is in ``tests/unit/test_code_executor.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.messages import (
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    MessageUsage,
    TextBlock,
)
from fastapi.testclient import TestClient
from openai.types.responses import Response, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME

_SANDBOX_URL = "http://127.0.0.1:9999/sandbox"
_DATED = {"type": "code_execution_20250825", "name": "code_execution"}
_INTERPRETER = {"type": "code_interpreter", "container": {"type": "auto"}}
_HEADER = "Otari-Code-Execution"
_CLAUDE = "claude-haiku-4-5"


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "claude-then-vertex": {
                        "select": [{"default": f"anthropic:{_CLAUDE}"}],
                        "on_failure": [f"vertexai:{_CLAUDE}"],
                    },
                    "vertex-then-claude": {
                        "select": [{"default": f"vertexai:{_CLAUDE}"}],
                        "on_failure": [f"anthropic:{_CLAUDE}"],
                    },
                    "claude-only": {
                        "select": [{"default": f"anthropic:{_CLAUDE}"}],
                        "on_failure": ["anthropic:claude-sonnet-4-5"],
                    },
                    "openai-then-mistral": {
                        "select": [{"default": "openai:gpt-4o-mini"}],
                        "on_failure": ["mistral:mistral-small-latest"],
                    },
                }
            }
        ),
    )


@pytest.fixture(autouse=True)
def _sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)


def _unavailable() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))


def _reply(text: str) -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model=_CLAUDE,
        content=[TextBlock(type="text", text=text, citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=5, output_tokens=2),
    )


def _responses_reply() -> Response:
    return Response(
        id="resp_test",
        created_at=0.0,
        model="fake",
        object="response",
        status=cast(Any, "completed"),
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=ResponseUsage(
            input_tokens=5,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=2,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=7,
        ),
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
    )


async def _events(text: str) -> AsyncIterator[MessageStreamEvent]:
    yield MessageStartEvent(type="message_start", message=cast(Any, _reply(text)))
    yield MessageDeltaEvent(
        type="message_delta",
        delta=MessageDelta(stop_reason=cast(Any, "end_turn"), stop_sequence=None),
        usage=MessageDeltaUsage(
            input_tokens=None,
            output_tokens=1,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            server_tool_use=None,
        ),
    )
    yield MessageStopEvent(type="message_stop")


@dataclass
class _Seen:
    """Every provider call, sandbox loop and sandbox one request made, in order."""

    provider_calls: list[dict[str, Any]] = field(default_factory=list)
    loop_calls: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)
    sandboxes: list[dict[str, Any]] = field(default_factory=list)

    def tool_types(self, kwargs: dict[str, Any]) -> set[str]:
        return {tool["type"] for tool in kwargs.get("tools") or []}


def _fake_sandbox(seen: _Seen) -> Any:
    def _build(**kwargs: Any) -> Any:
        seen.sandboxes.append(kwargs)
        backend = AsyncMock()
        backend.purpose_hints = lambda: []
        backend.__aenter__ = AsyncMock(return_value=backend)
        backend.__aexit__ = AsyncMock(return_value=None)
        return backend

    return _build


def _post_messages(
    client: TestClient,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    provider_fails: bool = False,
    loop_fails: bool = False,
) -> tuple[Any, _Seen]:
    seen = _Seen()

    async def fake_amessages(**kwargs: Any) -> Any:
        seen.provider_calls.append(kwargs)
        if provider_fails and len(seen.provider_calls) == 1:
            raise _unavailable()
        if kwargs.get("stream"):
            return _events("via-provider")
        return _reply("via-provider")

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any) -> MessageResponse:
        seen.loop_calls.append((completion_kwargs, extra))
        if loop_fails and len(seen.loop_calls) == 1:
            raise _unavailable()
        return _reply("via-sandbox")

    def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any
    ) -> AsyncIterator[MessageStreamEvent]:
        seen.loop_calls.append((completion_kwargs, extra))
        return _events("via-sandbox")

    with (
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        patch("gateway.api.routes._pipeline.SandboxBackend", new=_fake_sandbox(seen)),
    ):
        response = client.post(f"{API_ROOT}/messages", json=body, headers=headers)
    return response, seen


def _messages_body(policy: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": policy,
        "messages": [{"role": "user", "content": "compute"}],
        "max_tokens": 100,
        "tools": [_DATED],
        **extra,
    }


def _rows(db_session: Session) -> list[UsageLog]:
    return list(db_session.scalars(select(UsageLog)).all())


# --- Messages ---------------------------------------------------------------------------------


def test_a_native_head_runs_the_code_itself_and_opens_no_sandbox(
    client: TestClient, api_key_header: dict[str, str], db_session: Session
) -> None:
    response, seen = _post_messages(client, api_key_header, _messages_body("claude-then-vertex"))

    assert response.status_code == 200, response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{"code_execution_20250825"}]
    assert seen.loop_calls == []
    assert seen.sandboxes == [], "a sandbox was opened for a request the provider answered"
    assert "Otari-Container-Id" not in response.headers
    served = [row for row in _rows(db_session) if row.status == "success"]
    assert [row.provider for row in served] == ["anthropic"]


def test_a_fallback_that_cannot_run_the_code_runs_it_on_the_sandbox(
    client: TestClient, api_key_header: dict[str, str], db_session: Session
) -> None:
    response, seen = _post_messages(client, api_key_header, _messages_body("claude-then-vertex"), provider_fails=True)

    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["text"] == "via-sandbox"
    # The head was sent Anthropic's tool, and only the fallback brought the code here.
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{"code_execution_20250825"}]
    assert len(seen.loop_calls) == 1
    loop_kwargs, loop_extra = seen.loop_calls[0]
    assert seen.tool_types(loop_kwargs) == set(), "the claimed declaration was forwarded to the fallback"
    # Answered in the blocks the caller's own vocabulary expects, as Anthropic would have.
    assert loop_extra.get("native_tools") == frozenset({CODE_EXECUTION_TOOL_NAME})
    assert len(seen.sandboxes) == 1
    assert {(row.provider, row.status) for row in _rows(db_session)} >= {
        ("anthropic", "absorbed"),
        ("vertexai", "success"),
    }


def test_a_head_that_cannot_run_the_code_falls_back_to_one_that_runs_it_itself(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    response, seen = _post_messages(client, api_key_header, _messages_body("vertex-then-claude"), loop_fails=True)

    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["text"] == "via-provider"
    assert len(seen.loop_calls) == 1
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{"code_execution_20250825"}]


def test_a_sandbox_refusal_skips_only_the_fallback_that_needs_it(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """A workspace with code execution off still has Anthropic run its own."""
    workspace_id = client.get(f"{API_ROOT}/workspaces", headers=master_key_header).json()["data"][0]["id"]
    stored = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/code-execution-policy",
        json={"enabled": False},
        headers=master_key_header,
    )
    assert stored.status_code == 200, stored.text

    served, served_seen = _post_messages(client, api_key_header, _messages_body("claude-then-vertex"))
    assert served.status_code == 200, served.text
    assert served_seen.sandboxes == []

    failed, seen = _post_messages(client, api_key_header, _messages_body("claude-then-vertex"), provider_fails=True)

    assert failed.status_code >= 500, failed.text
    assert len(seen.provider_calls) == 1
    assert seen.loop_calls == [], "the fallback ran the code the workspace switched off"
    skipped = [row for row in _rows(db_session) if row.status == "absorbed" and row.provider == "vertexai"]
    assert len(skipped) == 1
    assert (skipped[0].error_message or "").startswith("Skipped: ")


def test_a_head_whose_sandbox_is_refused_is_skipped_for_a_fallback_that_runs_the_code_itself(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    workspace_id = client.get(f"{API_ROOT}/workspaces", headers=master_key_header).json()["data"][0]["id"]
    stored = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/code-execution-policy",
        json={"enabled": False},
        headers=master_key_header,
    )
    assert stored.status_code == 200, stored.text

    response, seen = _post_messages(client, api_key_header, _messages_body("vertex-then-claude"))

    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["text"] == "via-provider"
    assert seen.loop_calls == []
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{"code_execution_20250825"}]
    skipped = [row for row in _rows(db_session) if row.status == "absorbed" and row.provider == "vertexai"]
    assert len(skipped) == 1
    assert (skipped[0].error_message or "").startswith("Skipped: ")


def test_a_request_none_of_whose_candidates_can_run_the_code_is_refused(
    client: TestClient, api_key_header: dict[str, str], master_key_header: dict[str, str]
) -> None:
    workspace_id = client.get(f"{API_ROOT}/workspaces", headers=master_key_header).json()["data"][0]["id"]
    stored = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/code-execution-policy",
        json={"enabled": False},
        headers=master_key_header,
    )
    assert stored.status_code == 200, stored.text

    response, seen = _post_messages(client, {**api_key_header, _HEADER: "otari"}, _messages_body("claude-then-vertex"))

    assert response.status_code == 403, response.text
    assert "not enabled" in response.json()["detail"]["error"]["message"]
    assert seen.provider_calls == []
    assert seen.loop_calls == []


def test_a_plan_whose_candidates_all_run_the_code_never_uses_the_sandbox(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    response, seen = _post_messages(client, api_key_header, _messages_body("claude-only"), provider_fails=True)

    assert response.status_code == 200, response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{"code_execution_20250825"}] * 2
    assert seen.loop_calls == []
    assert seen.sandboxes == []


@pytest.mark.parametrize(("executor", "sandboxed"), [("otari", True), ("provider", False)])
def test_an_executor_asked_for_decides_every_candidate(
    client: TestClient, api_key_header: dict[str, str], executor: str, sandboxed: bool
) -> None:
    response, seen = _post_messages(
        client, {**api_key_header, _HEADER: executor}, _messages_body("claude-then-vertex"), provider_fails=True
    )

    assert response.status_code == 200, response.text
    if sandboxed:
        assert seen.provider_calls == []
        assert len(seen.loop_calls) == 1
    else:
        assert [seen.tool_types(call) for call in seen.provider_calls] == [{"code_execution_20250825"}] * 2
        assert seen.loop_calls == []


def test_otaris_container_spelling_is_kept_for_the_sandbox_and_left_out_for_the_provider(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    body = _messages_body("claude-then-vertex", container="auto")

    response, seen = _post_messages(client, api_key_header, body, provider_fails=True)

    assert response.status_code == 200, response.text
    assert "container" not in seen.provider_calls[0]
    loop_kwargs, _ = seen.loop_calls[0]
    assert "container" not in loop_kwargs
    # ``auto`` asks the sandbox to outlive the request, which only the fallback's could.
    assert seen.sandboxes[0]["containers"] is not None


def test_a_streamed_fallback_that_cannot_run_the_code_runs_it_on_the_sandbox(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    response, seen = _post_messages(
        client, api_key_header, _messages_body("claude-then-vertex", stream=True), provider_fails=True
    )

    assert response.status_code == 200, response.text
    assert "via-sandbox" in response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{"code_execution_20250825"}]
    loop_kwargs, loop_extra = seen.loop_calls[0]
    assert seen.tool_types(loop_kwargs) == set()
    assert loop_extra.get("native_tools") == frozenset({CODE_EXECUTION_TOOL_NAME})
    assert len(seen.sandboxes) == 1


def test_a_streamed_native_head_opens_no_sandbox(client: TestClient, api_key_header: dict[str, str]) -> None:
    response, seen = _post_messages(client, api_key_header, _messages_body("claude-then-vertex", stream=True))

    assert response.status_code == 200, response.text
    assert "via-provider" in response.text
    assert seen.loop_calls == []
    assert seen.sandboxes == []


# --- Responses --------------------------------------------------------------------------------


def _post_responses(
    client: TestClient, headers: dict[str, str], body: dict[str, Any], *, provider_fails: bool = False
) -> tuple[Any, _Seen]:
    seen = _Seen()

    async def fake_aresponses(**kwargs: Any) -> Response:
        seen.provider_calls.append(kwargs)
        if provider_fails and len(seen.provider_calls) == 1:
            raise _unavailable()
        return _responses_reply()

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any) -> Response:
        seen.loop_calls.append((completion_kwargs, extra))
        return _responses_reply()

    with (
        patch("gateway.api.routes.responses.aresponses", new=fake_aresponses),
        patch("gateway.api.routes.responses.responses_tool_loop", new=fake_loop),
        patch("gateway.api.routes._pipeline.SandboxBackend", new=_fake_sandbox(seen)),
    ):
        response = client.post(f"{API_ROOT}/responses", json=body, headers=headers)
    return response, seen


def _responses_body() -> dict[str, Any]:
    return {"model": "openai-then-mistral", "input": "compute", "tools": [_INTERPRETER]}


def test_responses_leaves_the_interpreter_with_an_openai_head(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    response, seen = _post_responses(client, api_key_header, _responses_body())

    assert response.status_code == 200, response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{"code_interpreter"}]
    assert seen.loop_calls == []
    assert seen.sandboxes == []


def test_responses_runs_the_interpreter_on_the_sandbox_for_a_fallback_without_one(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    response, seen = _post_responses(client, api_key_header, _responses_body(), provider_fails=True)

    assert response.status_code == 200, response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{"code_interpreter"}]
    loop_kwargs, loop_extra = seen.loop_calls[0]
    assert seen.tool_types(loop_kwargs) == set()
    # A ``code_interpreter_call`` item back, as OpenAI would have answered.
    assert loop_extra.get("native_tools") == frozenset({CODE_EXECUTION_TOOL_NAME})
    assert len(seen.sandboxes) == 1
