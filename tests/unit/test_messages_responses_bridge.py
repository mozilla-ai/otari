"""The Messages-to-Responses bridge for provider instances that set ``messages_api: responses``."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from any_llm.exceptions import UnsupportedParameterError
from any_llm.types.messages import MessageDeltaEvent, MessageResponse
from starlette.datastructures import Headers

from gateway.api.routes.messages import _ADAPTER, _messages_stream_usage
from gateway.core.config import GatewayConfig
from gateway.services.inference import (
    REASONING_SIGNATURE_PREFIX,
    call_messages,
    conversation_cache_key,
    reasoning_signature,
    usage_unreported,
    with_responses_instance,
)
from gateway.services.inference._messages_responses_bridge import (
    CONVERSATION_CACHE_KEY_FIELD,
    RESPONSES_INSTANCE_FIELD,
    ResponsesStreamFailedError,
    responses_kwargs_for,
)
from gateway.services.mcp_loop_messages import anthropic_tool_loop, anthropic_tool_loop_stream

_ARESPONSES = "gateway.services.inference._messages_responses_bridge.aresponses"
_MODEL = "openai:gpt-5"
_INSTANCE = "openai"
_USAGE = {
    "input_tokens": 100,
    "input_tokens_details": {"cached_tokens": 40},
    "output_tokens": 30,
    "output_tokens_details": {"reasoning_tokens": 12},
    "total_tokens": 130,
}


def _response(output: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    return {"id": "resp_1", "model": "gpt-5", "status": "completed", "output": output, "usage": _USAGE, **fields}


def _message(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _reasoning(encrypted: str, summary: str = "") -> dict[str, Any]:
    parts = [{"type": "summary_text", "text": summary}] if summary else []
    return {"type": "reasoning", "id": "rs_1", "summary": parts, "encrypted_content": encrypted}


def _call(name: str, arguments: str, call_id: str = "call_1") -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _kwargs(**fields: Any) -> dict[str, Any]:
    return {
        "model": _MODEL,
        "api_key": "sk",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
        **fields,
    }


def _translate(kwargs: dict[str, Any]) -> dict[str, Any]:
    return responses_kwargs_for(kwargs, instance=_INSTANCE)


async def _aiter(events: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for event in events:
        yield event


async def _via_responses(result: Any, captured: list[dict[str, Any]] | None = None, **fields: Any) -> Any:
    async def fake_aresponses(**kwargs: Any) -> Any:
        if captured is not None:
            captured.append(kwargs)
        return result

    async def native(**_: Any) -> Any:
        raise AssertionError("the native Messages call must not run")

    with patch(_ARESPONSES, new=fake_aresponses):
        return await call_messages(native, with_responses_instance(_kwargs(**fields), _INSTANCE))


# ---------- dispatch ----------


@pytest.mark.asyncio
async def test_unmarked_call_stays_native_and_drops_reasoning_signed_here() -> None:
    sent: dict[str, Any] = {}

    async def native(**kwargs: Any) -> str:
        sent.update(kwargs)
        return "native"

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t", "signature": f"{REASONING_SIGNATURE_PREFIX}abc"},
                {"type": "thinking", "thinking": "a", "signature": "anthropic-sig"},
                {"type": "text", "text": "hello"},
            ],
        },
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "otari-rsn1.x"}]},
    ]
    assert await call_messages(native, {"model": "anthropic:claude", "messages": messages}) == "native"
    assert sent["messages"] == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "a", "signature": "anthropic-sig"},
                {"type": "text", "text": "hello"},
            ],
        },
    ]


@pytest.mark.asyncio
async def test_marked_call_to_a_provider_without_a_responses_api_stays_native() -> None:
    sent: dict[str, Any] = {}

    async def native(**kwargs: Any) -> str:
        sent.update(kwargs)
        return "native"

    kwargs = with_responses_instance({"model": "mistral:mistral-small-latest", "messages": []}, _INSTANCE)
    assert await call_messages(native, kwargs) == "native"
    assert RESPONSES_INSTANCE_FIELD not in sent


def test_only_an_opted_in_instance_marks_the_kwargs() -> None:
    assert with_responses_instance({"model": _MODEL}, None) == {"model": _MODEL}
    assert with_responses_instance({"model": _MODEL}, _INSTANCE)[RESPONSES_INSTANCE_FIELD] == _INSTANCE


@pytest.mark.asyncio
async def test_native_call_continuing_a_tool_turn_whose_thinking_it_drops_leaves_thinking_off() -> None:
    sent: list[dict[str, Any]] = []

    async def native(**kwargs: Any) -> None:
        sent.append(kwargs)

    tool_turn = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "", "signature": reasoning_signature(_INSTANCE, "enc")},
                {"type": "tool_use", "id": "c", "name": "read", "input": {}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": "x"}]},
    ]
    thinking = {"type": "enabled", "budget_tokens": 2048}
    await call_messages(native, {"model": "anthropic:claude", "messages": tool_turn, "thinking": thinking})
    # A new user turn may open without thinking, so thinking stays on.
    new_turn = [*tool_turn, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
    await call_messages(native, {"model": "anthropic:claude", "messages": new_turn, "thinking": thinking})

    assert "thinking" not in sent[0]
    assert sent[0]["messages"][1]["content"] == [{"type": "tool_use", "id": "c", "name": "read", "input": {}}]
    assert sent[1]["thinking"] == thinking


def test_config_refuses_an_unknown_messages_api() -> None:
    with pytest.raises(ValueError, match=r"providers\.openai\.messages_api"):
        GatewayConfig(providers={"openai": {"api_key": "sk", "messages_api": "chat"}}).validate_provider_instances()
    GatewayConfig(providers={"openai": {"api_key": "sk", "messages_api": "responses"}}).validate_provider_instances()


# ---------- request ----------


def test_request_fields_translate() -> None:
    out = _translate(
        _kwargs(
            system=[{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
            temperature=0.2,
            prompt_cache_key="k",
            user="end-user-digest",
            tools=[
                {"name": "read", "description": "Read a file", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
            tool_choice={"type": "any", "disable_parallel_tool_use": True},
            thinking={"type": "enabled", "budget_tokens": 10000},
            metadata={"user_id": "u"},
            cache_control={"type": "ephemeral"},
            top_k=5,
            stream=True,
        )
    )
    assert out == {
        "model": _MODEL,
        "api_key": "sk",
        "temperature": 0.2,
        "prompt_cache_key": "k",
        "input_data": [{"role": "user", "content": "hi"}],
        "store": False,
        "instructions": "be brief",
        "max_output_tokens": 1024,
        "safety_identifier": "end-user-digest",
        "tools": [
            {
                "type": "function",
                "name": "read",
                "parameters": {"type": "object"},
                "strict": False,
                "description": "Read a file",
            }
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "stream": True,
    }


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ({"type": "auto"}, "auto"),
        ({"type": "none"}, "none"),
        ({"type": "tool", "name": "read"}, {"type": "function", "name": "read"}),
    ],
)
def test_tool_choice_translates(tool_choice: dict[str, Any], expected: Any) -> None:
    assert _translate(_kwargs(tool_choice=tool_choice))["tool_choice"] == expected


@pytest.mark.parametrize(
    ("budget", "effort"),
    [(1024, "low"), (2048, "low"), (8000, "medium"), (16384, "medium"), (31999, "high")],
)
def test_thinking_budget_maps_to_an_effort(budget: int, effort: str) -> None:
    out = _translate(_kwargs(thinking={"type": "enabled", "budget_tokens": budget}))
    assert out["reasoning"] == {"effort": effort, "summary": "auto"}


def test_output_config_effort_wins_over_the_budget() -> None:
    out = _translate(_kwargs(thinking={"type": "enabled", "budget_tokens": 1024}, output_format={"effort": "max"}))
    assert out["reasoning"] == {"effort": "high", "summary": "auto"}


def test_reasoning_model_gets_encrypted_reasoning_without_asking() -> None:
    out = _translate(_kwargs(model="openai:gpt-5.1-codex"))
    assert out["reasoning"] == {"summary": "auto"}
    assert out["include"] == ["reasoning.encrypted_content"]
    disabled = _translate(_kwargs(thinking={"type": "disabled"}))
    assert "reasoning" not in disabled
    assert disabled["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize("model", ["openai:gpt-4.1", "openai:gpt-5-chat-latest"])
@pytest.mark.parametrize(
    "asked",
    [{}, {"thinking": {"type": "enabled", "budget_tokens": 4096}}, {"output_format": {"effort": "high"}}],
)
def test_non_reasoning_model_gets_no_reasoning_fields(model: str, asked: dict[str, Any]) -> None:
    out = _translate(_kwargs(model=model, **asked))
    assert "reasoning" not in out
    assert "include" not in out


def test_a_routed_reasoning_model_is_recognized() -> None:
    assert _translate(_kwargs(model="openrouter:openai/gpt-5"))["include"] == ["reasoning.encrypted_content"]


def test_json_schema_output_config_becomes_a_text_format() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    out = _translate(_kwargs(output_format={"format": {"type": "json_schema", "schema": schema}}))
    assert out["text"] == {"format": {"type": "json_schema", "name": "output", "schema": schema, "strict": False}}


def test_stop_sequences_are_refused() -> None:
    with pytest.raises(UnsupportedParameterError, match="stop_sequences"):
        _translate(_kwargs(stop_sequences=["END"]))


def test_conversation_translates_to_input_items() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAA"}},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "PDF"}},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": reasoning_signature(_INSTANCE, "enc-1")},
                {"type": "thinking", "thinking": "other", "signature": "anthropic-sig"},
                # Encrypted for another instance's account, which this one would refuse.
                {"type": "thinking", "thinking": "elsewhere", "signature": reasoning_signature("azure", "enc-2")},
                {"type": "redacted_thinking", "data": "x"},
                {"type": "text", "text": "Reading "},
                {"type": "text", "text": "both."},
                {"type": "tool_use", "id": "call_a", "name": "read", "input": {"path": "a"}},
                {"type": "tool_use", "id": "call_b", "name": "read", "input": {"path": "b"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_a", "content": "A"},
                {
                    "type": "tool_result",
                    "tool_use_id": "call_b",
                    "content": [{"type": "text", "text": "no"}],
                    "is_error": True,
                },
                {"type": "text", "text": "go on"},
            ],
        },
    ]
    assert _translate(_kwargs(messages=messages))["input_data"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,AAA", "detail": "auto"},
                {"type": "input_file", "filename": "document.pdf", "file_data": "data:application/pdf;base64,PDF"},
            ],
        },
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "plan"}], "encrypted_content": "enc-1"},
        {"role": "assistant", "content": "Reading both."},
        {"type": "function_call", "call_id": "call_a", "name": "read", "arguments": '{"path": "a"}'},
        {"type": "function_call", "call_id": "call_b", "name": "read", "arguments": '{"path": "b"}'},
        {"type": "function_call_output", "call_id": "call_a", "output": "A"},
        {"type": "function_call_output", "call_id": "call_b", "output": "Error: no"},
        {"role": "user", "content": [{"type": "input_text", "text": "go on"}]},
    ]


def test_tool_result_with_an_image_keeps_its_parts() -> None:
    image = {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
    messages = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": [image]}]}]
    assert _translate(_kwargs(messages=messages))["input_data"] == [
        {
            "type": "function_call_output",
            "call_id": "c",
            "output": [{"type": "input_image", "image_url": "https://example.com/a.png", "detail": "auto"}],
        }
    ]


def test_unknown_user_block_is_refused() -> None:
    messages = [{"role": "user", "content": [{"type": "container_upload", "file_id": "f"}]}]
    with pytest.raises(UnsupportedParameterError, match="container_upload"):
        _translate(_kwargs(messages=messages))


# ---------- response ----------


@pytest.mark.asyncio
async def test_response_translates_to_a_message() -> None:
    captured: list[dict[str, Any]] = []
    output = [_reasoning("enc-1", "Thinking it over"), _message("Hello"), _call("read", '{"path": "a"}')]
    result = await _via_responses(_response(output), captured)

    assert captured[0]["store"] is False
    assert RESPONSES_INSTANCE_FIELD not in captured[0]
    assert isinstance(result, MessageResponse)
    assert result.model_dump(exclude_none=True) == {
        "id": "msg_resp_1",
        "type": "message",
        "role": "assistant",
        "model": "gpt-5",
        "content": [
            {"type": "thinking", "thinking": "Thinking it over", "signature": reasoning_signature(_INSTANCE, "enc-1")},
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "a"}},
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 60,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 0,
            "output_tokens": 30,
            "output_tokens_details": {"thinking_tokens": 12},
        },
    }


@pytest.mark.parametrize(
    ("fields", "output", "stop_reason"),
    [
        ({}, [_message("done")], "end_turn"),
        ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, [], "max_tokens"),
        ({"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}, [], "refusal"),
        ({}, [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}], "refusal"),
    ],
)
@pytest.mark.asyncio
async def test_stop_reasons(fields: dict[str, Any], output: list[dict[str, Any]], stop_reason: str) -> None:
    result = await _via_responses(_response(output, **fields))
    assert result.stop_reason == stop_reason


@pytest.mark.asyncio
async def test_reasoning_round_trips_into_the_next_request() -> None:
    first = await _via_responses(_response([_reasoning("enc-1"), _call("read", "{}")]))
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [block.model_dump(exclude_none=True) for block in first.content]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "x"}]},
    ]
    captured: list[dict[str, Any]] = []
    await _via_responses(_response([_message("ok")]), captured, messages=history)
    assert captured[0]["input_data"][1:] == [
        {"type": "reasoning", "summary": [], "encrypted_content": "enc-1"},
        {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "x"},
    ]


# ---------- streaming ----------


def _stream(*output_events: dict[str, Any], final: str = "response.completed") -> list[dict[str, Any]]:
    created = {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5", "output": []}}
    done = {"type": final, "response": _response([])}
    return [created, *output_events, done]


def _wire(events: list[Any]) -> list[dict[str, Any]]:
    return [event.model_dump(exclude_none=True) for event in events]


async def _collect(stream: AsyncIterator[Any]) -> list[dict[str, Any]]:
    return _wire([event async for event in stream])


@pytest.mark.asyncio
async def test_stream_translates_to_messages_events() -> None:
    events = _stream(
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}},
        {"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 0},
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "Plan"},
        {"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 1},
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 1, "delta": "More"},
        {"type": "response.output_item.done", "output_index": 0, "item": _reasoning("enc-1")},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "m"}},
        {"type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": "Hi"},
        {"type": "response.output_item.done", "output_index": 1, "item": _message("Hi")},
        {"type": "response.output_item.added", "output_index": 2, "item": {**_call("read", ""), "id": "fc"}},
        {"type": "response.function_call_arguments.delta", "output_index": 2, "delta": '{"path":'},
        {"type": "response.function_call_arguments.delta", "output_index": 2, "delta": '"a"}'},
        {"type": "response.output_item.done", "output_index": 2, "item": _call("read", '{"path":"a"}')},
    )
    stream = await _via_responses(_aiter(events), stream=True)
    sig = reasoning_signature(_INSTANCE, "enc-1")
    assert await _collect(stream) == [
        {
            "type": "message_start",
            "message": {
                "id": "msg_resp_1",
                "type": "message",
                "role": "assistant",
                "model": "gpt-5",
                "content": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Plan"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "\n\n"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "More"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": sig}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "call_1", "name": "read", "input": {}},
        },
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"path":'}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '"a"}'}},
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {
                "input_tokens": 60,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 0,
                "output_tokens": 30,
                "output_tokens_details": {"thinking_tokens": 12},
            },
        },
        {"type": "message_stop"},
    ]


@pytest.mark.asyncio
async def test_stream_arguments_sent_only_on_done_still_reach_the_client() -> None:
    events = _stream(
        {"type": "response.output_item.added", "output_index": 0, "item": _call("read", "")},
        {"type": "response.output_item.done", "output_index": 0, "item": _call("read", '{"path":"a"}')},
    )
    wire = await _collect(await _via_responses(_aiter(events), stream=True))
    assert {"type": "input_json_delta", "partial_json": '{"path":"a"}'} in [e.get("delta") for e in wire]


@pytest.mark.asyncio
async def test_incomplete_stream_reports_max_tokens() -> None:
    incomplete: dict[str, Any] = {
        "type": "response.incomplete",
        "response": _response([], status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
    }
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5"}},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "par"},
        incomplete,
    ]
    wire = await _collect(await _via_responses(_aiter(events), stream=True))
    assert wire[-3] == {"type": "content_block_stop", "index": 0}
    assert wire[-2]["delta"] == {"stop_reason": "max_tokens"}
    assert wire[-1] == {"type": "message_stop"}


@pytest.mark.asyncio
async def test_failed_stream_raises_with_the_provider_code() -> None:
    error = {"code": "context_length_exceeded", "message": "too long"}
    events = [
        {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5"}},
        {"type": "response.failed", "response": {"id": "resp_1", "status": "failed", "error": error}},
    ]
    stream = await _via_responses(_aiter(events), stream=True)
    with pytest.raises(ResponsesStreamFailedError) as raised:
        await _collect(stream)
    assert raised.value.code == "context_length_exceeded"


@pytest.mark.asyncio
async def test_stream_opens_a_thinking_block_only_for_reasoning_with_something_to_show() -> None:
    events = _stream(
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning"}},
        {"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 0},
        {"type": "response.output_item.done", "output_index": 0, "item": _reasoning("")},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "reasoning"}},
        {"type": "response.output_item.done", "output_index": 1, "item": _reasoning("enc-1")},
    )
    wire = await _collect(await _via_responses(_aiter(events), stream=True))
    blocks = [e for e in wire if e["type"].startswith("content_block")]
    assert blocks == [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": reasoning_signature(_INSTANCE, "enc-1")},
        },
        {"type": "content_block_stop", "index": 0},
    ]


# ---------- tool loops ----------


class _Pool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "fetch_url", "description": "", "parameters": {}}}]

    def owns_tool(self, name: str) -> bool:
        return name == "fetch_url"

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return "page text"


@pytest.mark.asyncio
async def test_tool_loop_sends_its_reasoning_back_through_the_bridge() -> None:
    replies = iter([_response([_reasoning("enc-1"), _call("fetch_url", '{"u": "x"}')]), _response([_message("done")])])
    sent: list[list[dict[str, Any]]] = []

    async def fake_aresponses(**kwargs: Any) -> Any:
        sent.append(kwargs["input_data"])
        return next(replies)

    pool = _Pool()
    with patch(_ARESPONSES, new=fake_aresponses):
        result = await anthropic_tool_loop(
            completion_kwargs=with_responses_instance(_kwargs(), _INSTANCE),
            pool=cast(Any, pool),
            max_iterations=3,
        )

    assert pool.calls == [("fetch_url", {"u": "x"})]
    assert sent[1][1:] == [
        {"type": "reasoning", "summary": [], "encrypted_content": "enc-1"},
        {"type": "function_call", "call_id": "call_1", "name": "fetch_url", "arguments": '{"u": "x"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "page text"},
    ]
    assert result.content[-1].text == "done"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_streaming_tool_loop_sends_its_reasoning_back_through_the_bridge() -> None:
    turns: Iterator[list[dict[str, Any]]] = iter(
        [
            _stream(
                {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning"}},
                {"type": "response.output_item.done", "output_index": 0, "item": _reasoning("enc-1")},
                {"type": "response.output_item.added", "output_index": 1, "item": _call("fetch_url", "")},
                {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": '{"u": "x"}'},
                {"type": "response.output_item.done", "output_index": 1, "item": _call("fetch_url", '{"u": "x"}')},
            ),
            _stream(
                {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "done"},
                {"type": "response.output_item.done", "output_index": 0, "item": _message("done")},
            ),
        ]
    )
    sent: list[list[dict[str, Any]]] = []

    async def fake_aresponses(**kwargs: Any) -> Any:
        sent.append(kwargs["input_data"])
        return _aiter(next(turns))

    pool = _Pool()
    with patch(_ARESPONSES, new=fake_aresponses):
        wire = _wire(
            [
                event
                async for event in anthropic_tool_loop_stream(
                    completion_kwargs=with_responses_instance(_kwargs(stream=True), _INSTANCE),
                    pool=cast(Any, pool),
                    max_iterations=3,
                )
            ]
        )

    assert pool.calls == [("fetch_url", {"u": "x"})]
    assert sent[1][1] == {"type": "reasoning", "summary": [], "encrypted_content": "enc-1"}
    assert wire[-1] == {"type": "message_stop"}
    assert {"type": "text_delta", "text": "done"} in [event.get("delta") for event in wire]


@pytest.mark.asyncio
async def test_tool_loop_bills_every_round_input_and_cached_input() -> None:
    replies = iter([_response([_call("fetch_url", "{}")]), _response([_message("done")])])

    async def fake_aresponses(**_: Any) -> Any:
        return next(replies)

    with patch(_ARESPONSES, new=fake_aresponses):
        result = await anthropic_tool_loop(
            completion_kwargs=with_responses_instance(_kwargs(), _INSTANCE), pool=cast(Any, _Pool()), max_iterations=3
        )

    assert (result.usage.input_tokens, result.usage.cache_read_input_tokens, result.usage.output_tokens) == (
        120,
        80,
        60,
    )


@pytest.mark.asyncio
async def test_streaming_tool_loop_bills_every_round_input_and_cached_input() -> None:
    turns = iter(
        [
            _stream(
                {"type": "response.output_item.added", "output_index": 0, "item": _call("fetch_url", "")},
                {"type": "response.output_item.done", "output_index": 0, "item": _call("fetch_url", "{}")},
            ),
            _stream({"type": "response.output_item.done", "output_index": 0, "item": _message("done")}),
        ]
    )

    async def fake_aresponses(**_: Any) -> Any:
        return _aiter(next(turns))

    with patch(_ARESPONSES, new=fake_aresponses):
        events = [
            event
            async for event in anthropic_tool_loop_stream(
                completion_kwargs=with_responses_instance(_kwargs(stream=True), _INSTANCE),
                pool=cast(Any, _Pool()),
                max_iterations=3,
            )
        ]

    deltas = [event for event in events if isinstance(event, MessageDeltaEvent)]
    assert len(deltas) == 1
    usage = _messages_stream_usage(deltas[0])
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens) == (120, 60)
    assert deltas[0].usage.cache_read_input_tokens == 80


# ---------- conversation cache key ----------


def test_the_session_supplies_the_conversation_cache_key() -> None:
    conversation = Headers({"otari-conversation-id": "c1"})
    assert conversation_cache_key(conversation, session_label=None) == {CONVERSATION_CACHE_KEY_FIELD: "client:c1"}
    harness = Headers({"x-claude-code-session-id": "s1"})
    assert conversation_cache_key(harness, session_label=None) == {CONVERSATION_CACHE_KEY_FIELD: "harness:s1"}
    assert conversation_cache_key(Headers({}), session_label=None) == {}


@pytest.mark.asyncio
async def test_conversation_cache_key_fills_an_absent_prompt_cache_key() -> None:
    captured: list[dict[str, Any]] = []
    await _via_responses(_response([_message("ok")]), captured, **{CONVERSATION_CACHE_KEY_FIELD: "conv"})
    await _via_responses(
        _response([_message("ok")]), captured, prompt_cache_key="caller", **{CONVERSATION_CACHE_KEY_FIELD: "conv"}
    )
    assert [call["prompt_cache_key"] for call in captured] == ["conv", "caller"]
    assert all(CONVERSATION_CACHE_KEY_FIELD not in call for call in captured)


@pytest.mark.asyncio
async def test_conversation_cache_key_never_reaches_a_native_call() -> None:
    sent: dict[str, Any] = {}

    async def native(**kwargs: Any) -> None:
        sent.update(kwargs)

    await call_messages(native, {"model": "anthropic:claude", "messages": [], CONVERSATION_CACHE_KEY_FIELD: "conv"})
    assert CONVERSATION_CACHE_KEY_FIELD not in sent
    assert "prompt_cache_key" not in sent


# ---------- missing usage ----------


@pytest.mark.asyncio
async def test_usage_the_provider_did_not_report_is_missing_to_metering_not_zero() -> None:
    response = _response([_message("ok")])
    del response["usage"]
    result = await _via_responses(response)
    assert result.usage.output_tokens == 0
    assert usage_unreported(result.usage)
    assert _ADAPTER.extract_usage(result) is None

    done = {"type": "response.completed", "response": response}
    stream = await _via_responses(_aiter([{"type": "response.created", "response": response}, done]), stream=True)
    events = [event async for event in stream]
    delta = next(event for event in events if event.type == "message_delta")
    assert usage_unreported(delta.usage)
    assert _messages_stream_usage(delta) is None

    # A tool loop that folds earlier turns' tokens in bills them.
    folded = result.usage.model_copy(update={"output_tokens": 7})
    assert not usage_unreported(folded)


@pytest.mark.asyncio
async def test_reported_zero_usage_is_still_metered() -> None:
    response = _response([_message("")])
    response["usage"] = {"input_tokens": 0, "output_tokens": 0}
    result = await _via_responses(response)
    assert not usage_unreported(result.usage)
    assert _ADAPTER.extract_usage(result) is not None
