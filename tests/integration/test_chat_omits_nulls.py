"""A chat completion leaves out its null fields, which strict OpenAI clients reject.

The nulls the OpenAI spec requires as keys stay: ``finish_reason``, a completion
choice's ``logprobs``, and its message's ``content`` and ``refusal``.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT

from .conftest import MODEL_NAME
from .test_chat_otari_tool_calls import _completion, _sse_payloads


def _null_paths(value: Any, path: str = "") -> set[str]:
    if value is None:
        return {path}
    if isinstance(value, dict):
        return {found for key, item in value.items() for found in _null_paths(item, f"{path}.{key}".lstrip("."))}
    if isinstance(value, list):
        return {found for index, item in enumerate(value) for found in _null_paths(item, f"{path}[{index}]")}
    return set()


def _post(client: TestClient, headers: dict[str, str], provider_result: Any, **extra: Any) -> Any:
    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], **extra}
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=provider_result)):
        return client.post(f"{API_ROOT}/chat/completions", json=body, headers=headers)


def test_an_answer_keeps_only_the_required_nulls(client: TestClient, api_key_header: dict[str, str]) -> None:
    completion = ChatCompletion.model_validate(
        {**_completion(None).model_dump(), "service_tier": None, "system_fingerprint": None}
    )
    completion.choices[0].message.citations = ["https://example.com"]  # type: ignore[attr-defined]

    response = _post(client, api_key_header, completion)

    assert response.status_code == 200, response.text
    body = response.json()
    assert _null_paths(body) == {"choices[0].logprobs", "choices[0].message.refusal"}
    assert body["choices"][0]["message"]["provider_specific_fields"] == {"citations": ["https://example.com"]}


def test_a_tool_call_answer_keeps_its_null_content(client: TestClient, api_key_header: dict[str, str]) -> None:
    response = _post(client, api_key_header, _completion(["web_search"]))

    assert response.status_code == 200, response.text
    body = response.json()
    assert _null_paths(body) == {
        "choices[0].logprobs",
        "choices[0].message.content",
        "choices[0].message.refusal",
    }
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "web_search"


def _chunk(delta: dict[str, Any], finish_reason: str | None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None}],
            "usage": None,
        }
    )


async def _stream() -> AsyncIterator[ChatCompletionChunk]:
    yield _chunk({"role": "assistant", "content": "hi", "tool_calls": None}, None)
    yield _chunk({}, "stop")


def test_a_stream_keeps_only_finish_reason_null(client: TestClient, api_key_header: dict[str, str]) -> None:
    response = _post(client, api_key_header, _stream(), stream=True)

    assert response.status_code == 200, response.text
    first, last = _sse_payloads(response.text)[:2]
    assert _null_paths(first) == {"choices[0].finish_reason"}
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "hi"}
    assert last["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}
