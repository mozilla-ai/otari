"""Gemini on the request path: native code execution, native tools and attachments past the inline limit.

The pure logic is covered by ``tests/unit/test_code_executor.py`` and
``tests/unit/test_content_normalizer.py``. These pin what reaches the provider
call through the chat and messages routes.
"""

from __future__ import annotations

import base64
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig

from .conftest import build_test_client

_GEMINI = "gemini:gemini-2.5-flash"
_SANDBOX_URL = "http://127.0.0.1:9999/sandbox"
_DATED = {"type": "code_execution_20250825", "name": "code_execution"}
_BARE = {"type": "code_execution"}
_NATIVE: dict[str, Any] = {"code_execution": {}}
_URI = "https://generativelanguage.googleapis.com/v1beta/files/abc"


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
def test_gemini_runs_the_bare_keyword_itself_on_chat(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch, sandbox: bool
) -> None:
    if sandbox:
        monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    else:
        monkeypatch.delenv("OTARI_SANDBOX_URL", raising=False)

    response, seen = _post_chat(client, api_key_header, _chat_body(_BARE))

    assert response.status_code == 200, response.text
    assert seen["tools"] == [_NATIVE]


def test_gemini_runs_anthropics_dated_keyword_itself_on_messages(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_messages(client, api_key_header, _messages_body(_DATED))

    assert response.status_code == 200, response.text
    assert seen["tools"] == [_NATIVE]


def test_a_gemini_native_tool_reaches_the_provider_untouched_on_messages(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    function = {"name": "lookup", "description": "", "input_schema": {"type": "object", "properties": {}}}

    response, seen = _post_messages(client, api_key_header, _messages_body({"google_search": {}}, function))

    assert response.status_code == 200, response.text
    assert seen["tools"] == [{"google_search": {}}, function]


@pytest.fixture
def small_inline_client(test_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    yield from build_test_client(test_config.model_copy(update={"files_gemini_inline_max_bytes": 100}))


def _key_header(client: TestClient, test_config: GatewayConfig) -> dict[str, str]:
    created = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "gemini-inline"},
        headers={API_KEY_HEADER: f"Bearer {test_config.master_key}"},
    )
    assert created.status_code in (200, 201), created.text
    return {API_KEY_HEADER: f"Bearer {created.json()['key']}"}


def _pdf_part(size: int) -> dict[str, Any]:
    data = base64.b64encode(b"x" * size).decode("ascii")
    return {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{data}", "filename": "big.pdf"}}


def test_an_attachment_past_geminis_inline_limit_reaches_it_by_uri(
    small_inline_client: TestClient, test_config: GatewayConfig
) -> None:
    headers = _key_header(small_inline_client, test_config)
    upload = AsyncMock(return_value=_URI)

    with patch("gateway.api.routes._normalize.upload_attachment", new=upload):
        response, seen = _post_chat(small_inline_client, headers, _chat_body(content=[_pdf_part(500)]))

    assert response.status_code == 200, response.text
    assert seen["messages"][0]["content"] == [{"type": "file", "file": {"file_data": _URI, "filename": "big.pdf"}}]
    assert upload.await_args is not None
    assert upload.await_args.kwargs["filename"] == "big.pdf"


def test_an_attachment_that_cannot_be_moved_is_refused_before_the_provider(
    small_inline_client: TestClient, test_config: GatewayConfig
) -> None:
    headers = _key_header(small_inline_client, test_config)

    with patch("gateway.api.routes._normalize.upload_attachment", new=AsyncMock(side_effect=RuntimeError("refused"))):
        response, seen = _post_chat(small_inline_client, headers, _chat_body(content=[_pdf_part(500)]))

    assert response.status_code == 400, response.text
    assert "inline" in response.text
    assert seen == {}
