"""A streaming ``/api/v1/messages`` response that never completes a message fails visibly.

A provider without a native Messages API (Bedrock's Kimi, say) reaches the gateway
through any-llm's Messages bridge, which only emits ``message_start`` on the first
upstream chunk and nothing at all for an upstream that produced none. The gateway
used to close such a stream as a success: a 200 whose body held no Messages event,
which a client can only report as "the stream did not return a message_start". It
must end in an SSE ``error`` event instead, and settle the request as failed.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from any_llm.providers.bedrock.bedrock import BedrockProvider
from any_llm.types.messages import MessageStreamEvent
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from conftest import InstallControlPlane
from gateway.api.deps import reset_config
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.models.usage import UsageLog
from gateway.models.users import User
from gateway.services.tools._native import SERVER_TOOL_USE_ID_PREFIX

from .conftest import MODEL_NAME, app_for
from .test_messages_streaming_usage import _configure_pricing, _seed_budgeted_user

_INCOMPLETE_MESSAGE = "The upstream provider ended the stream before completing the message."
_ATTEMPT_ID = "3f1b6a1e-0000-4000-8000-0000000000b1"
_CODE_USE_ID = f"{SERVER_TOOL_USE_ID_PREFIX}0123456789abcdef"


@pytest.fixture
def platform_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    app = app_for(GatewayConfig(mode="hybrid", platform={"base_url": "http://platform.test/api/v1"}))
    with TestClient(app) as client:
        yield client
    reset_config()
    reset_db()


class _FakeSandboxBackend:
    """SandboxBackend duck-type: owns ``code_execution`` and runs it without a sandbox."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeSandboxBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "code_execution", "description": "", "parameters": {}}}]

    def owns_tool(self, name: str) -> bool:
        return name == "code_execution"

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return "ok"


class _FakeBedrockClient:
    """A boto3 bedrock-runtime client whose ConverseStream replays one scripted event list per call."""

    def __init__(self, rounds: list[list[dict[str, Any]]]) -> None:
        self._rounds = rounds
        self.requests: list[dict[str, Any]] = []

    def converse_stream(self, **kwargs: Any) -> dict[str, Iterator[dict[str, Any]]]:
        self.requests.append(kwargs)
        return {"stream": iter(self._rounds[len(self.requests) - 1])}


_CODE_EXECUTION_ROUND: list[dict[str, Any]] = [
    {"messageStart": {"role": "assistant"}},
    {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {"toolUse": {"toolUseId": "tooluse_code", "name": "code_execution"}},
        }
    },
    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"code": "print(42)"}'}}}},
    {"contentBlockStop": {"contentBlockIndex": 0}},
    {"messageStop": {"stopReason": "tool_use"}},
    {"metadata": {"usage": {"inputTokens": 50, "outputTokens": 9, "totalTokens": 59}}},
]


def _install_hybrid_bedrock(
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
    rounds: list[list[dict[str, Any]]],
) -> tuple[_FakeBedrockClient, list[dict[str, Any]]]:
    """Route the request to Bedrock's Kimi with the gateway's sandbox, through the real any-llm bridge."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://sandbox:8080")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            attempt = {
                "attempt_id": _ATTEMPT_ID,
                "position": 0,
                "provider": "bedrock",
                "model": "us.moonshotai.kimi-k3",
                "api_key": "bedrock-api-key-test",
                "api_base": None,
                "managed": True,
            }
            return httpx.Response(200, json={"request_id": "req-kimi", "fallback_enabled": True, "attempts": [attempt]})
        if url.endswith("/gateway/code-execution/resolve"):
            return httpx.Response(200, json={"enabled": True})
        usage_reports.append(body)
        return httpx.Response(204)

    bedrock = _FakeBedrockClient(rounds)
    control_plane_transport(fake_post_platform)
    monkeypatch.setattr("gateway.api.routes._pipeline.SandboxBackend", _FakeSandboxBackend)
    monkeypatch.setattr(BedrockProvider, "_client_for_timeout", lambda _self, _timeout: bedrock)
    return bedrock, usage_reports


def _follow_up_after_code_execution() -> dict[str, Any]:
    """The incident's request: the turn after one that ran code and then called the caller's own tool."""
    return {
        "model": "kimi-k3",
        "max_tokens": 256,
        "stream": True,
        "tools": [
            {"type": "code_execution_20250825", "name": "code_execution"},
            {"name": "lookup", "description": "Look something up", "input_schema": {"type": "object"}},
        ],
        "messages": [
            {"role": "user", "content": "Summarize the uploaded file."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": _CODE_USE_ID,
                        "name": "code_execution",
                        "input": {"code": "print(42)"},
                    },
                    {
                        "type": "code_execution_tool_result",
                        "tool_use_id": _CODE_USE_ID,
                        "content": {
                            "type": "code_execution_result",
                            "stdout": "42\n",
                            "stderr": "",
                            "return_code": 0,
                            "content": [],
                        },
                    },
                    {"type": "tool_use", "id": "toolu_lookup", "name": "lookup", "input": {}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_lookup", "content": "found"}]},
        ],
    }


def _sse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for frame in body.split("\n\n"):
        if not frame.startswith("event: "):
            continue
        name, data = frame.split("\ndata: ", 1)
        events.append((name.removeprefix("event: "), json.loads(data)))
    return events


def _assert_ends_in_incomplete_error(events: list[tuple[str, dict[str, Any]]]) -> None:
    names = [name for name, _ in events if name != "ping"]
    assert "message_stop" not in names
    assert "done" not in names, "a failed Anthropic stream has no done marker"
    assert names[-1] == "error"
    assert events[-1][1] == {"type": "error", "error": {"type": "api_error", "message": _INCOMPLETE_MESSAGE}}


def test_a_bridged_provider_that_streams_nothing_ends_in_an_error_event(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    """The incident: Bedrock returned an event stream with no events, so the bridge emitted none."""
    bedrock, usage_reports = _install_hybrid_bedrock(monkeypatch, control_plane_transport, [[]])

    response = platform_client.post(
        f"{API_ROOT}/messages",
        json=_follow_up_after_code_execution(),
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    events = _sse_events(response.text)
    assert [name for name, _ in events if name != "ping"] == ["error"]
    _assert_ends_in_incomplete_error(events)

    reports = [(report["correlation_id"], report["status"]) for report in usage_reports]
    assert reports == [(_ATTEMPT_ID, "error")], "the attempt settles as failed, never as a success"

    # The history reached Bedrock whole: the gateway's code-execution pair folded
    # into text the bridge can carry, beside the caller's own tool call.
    [request] = bedrock.requests
    assistant = next(message for message in request["messages"] if message["role"] == "assistant")
    assert any("[code executed]" in block.get("text", "") and "42" in block["text"] for block in assistant["content"])
    assert any(block.get("toolUse", {}).get("toolUseId") == "toolu_lookup" for block in assistant["content"])


def test_a_tool_loop_round_that_streams_nothing_ends_the_started_message_in_an_error_event(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    """``message_start`` went out with the first round; the round after the gateway ran code came back empty."""
    bedrock, usage_reports = _install_hybrid_bedrock(monkeypatch, control_plane_transport, [_CODE_EXECUTION_ROUND, []])

    response = platform_client.post(
        f"{API_ROOT}/messages",
        json=_follow_up_after_code_execution(),
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200, response.text
    events = _sse_events(response.text)
    assert events[0][0] == "message_start"
    _assert_ends_in_incomplete_error(events)
    assert len(bedrock.requests) == 2, "the gateway ran the code and asked for the next round"
    assert [report["status"] for report in usage_reports] == ["error"]


async def _empty_stream(**_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
    async def _gen() -> AsyncIterator[MessageStreamEvent]:
        return
        yield  # pragma: no cover

    return _gen()


def _poll_usage_row(make_session: Callable[[], Session], user_id: str, *, timeout: float = 3.0) -> UsageLog | None:
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


def test_a_standalone_stream_with_no_events_fails_and_releases_its_reservation(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    user_id = "stream-without-message"
    _seed_budgeted_user(client, master_key_header, user_id)
    _configure_pricing(client, master_key_header, MODEL_NAME)

    with patch("gateway.api.routes.messages.amessages", new=_empty_stream):
        response = client.post(
            f"{API_ROOT}/messages",
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
    _assert_ends_in_incomplete_error(_sse_events(response.text))

    row = _poll_usage_row(db_session_factory, user_id)
    assert row is not None, "the failed stream must leave a usage row"
    assert row.status == "error"
    db = db_session_factory()
    try:
        user = db.query(User).filter(User.user_id == user_id).one()
        assert float(user.reserved) == 0.0, "the hold must be released"
        assert float(user.spend) == 0.0, "nothing was reported, so nothing is charged"
    finally:
        db.close()
