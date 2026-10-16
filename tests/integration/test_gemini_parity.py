"""Gemini on the request path: native code execution, native tools and attachments past the inline limit.

The pure logic is covered by ``tests/unit/test_code_executor.py`` and
``tests/unit/test_content_normalizer.py``. These pin what reaches the provider
call through the chat and messages routes.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT

_GEMINI = "gemini:gemini-2.5-flash"
_SANDBOX_URL = "http://127.0.0.1:9999/sandbox"
_DATED = {"type": "code_execution_20250825", "name": "code_execution"}
_BARE = {"type": "code_execution"}
_NATIVE: dict[str, Any] = {"code_execution": {}}


def _chat_response() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model="gemini-2.5-flash",
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="ok"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )


def _messages_response() -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model="gemini-2.5-flash",
        content=[TextBlock(type="text", text="ok", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=5, output_tokens=2),
    )


def _fake_sandbox(**_: Any) -> Any:
    backend = AsyncMock()
    backend.purpose_hints = lambda: []
    return AsyncMock(__aenter__=AsyncMock(return_value=backend), __aexit__=AsyncMock(return_value=None))


def _post_chat(client: TestClient, headers: dict[str, str], body: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    seen: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        seen.update(kwargs)
        return _chat_response()

    with (
        patch("gateway.api.routes.chat.acompletion", new=fake_acompletion),
        patch("gateway.api.routes._pipeline.SandboxBackend", new=_fake_sandbox),
    ):
        response = client.post(f"{API_ROOT}/chat/completions", json=body, headers=headers)
    return response, seen


def _post_messages(client: TestClient, headers: dict[str, str], body: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    seen: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        seen.update(kwargs)
        return _messages_response()

    with (
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
        patch("gateway.api.routes._pipeline.SandboxBackend", new=_fake_sandbox),
    ):
        response = client.post(f"{API_ROOT}/messages", json=body, headers=headers)
    return response, seen


def _chat_body(*tools: dict[str, Any], content: Any = "compute") -> dict[str, Any]:
    body: dict[str, Any] = {"model": _GEMINI, "messages": [{"role": "user", "content": content}]}
    if tools:
        body["tools"] = list(tools)
    return body


def _messages_body(*tools: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": _GEMINI,
        "messages": [{"role": "user", "content": "compute"}],
        "max_tokens": 100,
        "tools": list(tools),
    }


@pytest.mark.parametrize("sandbox", [True, False])
@pytest.mark.parametrize("declared", [_BARE, _NATIVE])
def test_gemini_runs_code_itself_on_chat(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    sandbox: bool,
    declared: dict[str, Any],
) -> None:
    if sandbox:
        monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    else:
        monkeypatch.delenv("OTARI_SANDBOX_URL", raising=False)

    response, seen = _post_chat(client, api_key_header, _chat_body(declared))

    assert response.status_code == 200, response.text
    assert seen["tools"] == [_NATIVE]


def test_gemini_runs_anthropics_dated_keyword_itself_on_messages(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_messages(client, api_key_header, _messages_body(_DATED))

    assert response.status_code == 200, response.text
    assert seen["tools"] == [_NATIVE]


def test_geminis_own_declaration_runs_on_the_sandbox_when_the_request_asks_for_otari(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemini's spelling carries no ``type``, and is still decided by the executor rather than forwarded."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    headers = {**api_key_header, "Otari-Code-Execution": "otari"}
    seen: list[dict[str, Any]] = []

    async def fake_loop_acompletion(**kwargs: Any) -> ChatCompletion:
        seen.append(kwargs)
        return _chat_response()

    with patch("gateway.services.mcp_loop.acompletion", new=fake_loop_acompletion):
        response, direct = _post_chat(client, headers, _chat_body(_NATIVE))

    assert response.status_code == 200, response.text
    sent = [*seen, direct] if direct else seen
    assert sent
    for kwargs in sent:
        assert _NATIVE not in (kwargs.get("tools") or [])
        assert _BARE not in (kwargs.get("tools") or [])


def test_a_gemini_native_tool_reaches_the_provider_untouched_on_messages(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    function = {"name": "lookup", "description": "", "input_schema": {"type": "object", "properties": {}}}

    response, seen = _post_messages(client, api_key_header, _messages_body({"google_search": {}}, function))

    assert response.status_code == 200, response.text
    assert seen["tools"] == [{"google_search": {}}, function]
