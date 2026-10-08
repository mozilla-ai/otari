"""The Responses-to-chat-completions bridge for providers without a Responses API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from any_llm import LLMProvider
from any_llm.exceptions import AnyLLMError, ModelNotFoundError, UnsupportedParameterError
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.types.responses import Response

from gateway.api.routes.responses import _strip_gateway_minted_items
from gateway.services.inference import (
    REASONING_ITEM_ID_PREFIX,
    aresponses_via_chat_completions,
    call_responses,
    uses_chat_completions_bridge,
)
from gateway.services.mcp_loop_responses import responses_tool_loop, responses_tool_loop_stream

_ACOMPLETION = "gateway.services.inference._responses_bridge.acompletion"
_USAGE = {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "total_tokens": 18,
    "prompt_tokens_details": {"cached_tokens": 3},
    "completion_tokens_details": {"reasoning_tokens": 2},
}


def _completion(message: dict[str, Any], finish_reason: str = "stop") -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "mistral-small-latest",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish_reason}],
            "usage": _USAGE,
        }
    )


def _chunk(
    delta: dict[str, Any] | None = None, finish_reason: str | None = None, usage: Any = None
) -> ChatCompletionChunk:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    return ChatCompletionChunk.model_validate(
        {
            "id": "cmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "mistral-small-latest",
            "choices": choices,
            "usage": usage,
        }
    )


async def _call(captured: dict[str, Any], result: Any, **kwargs: Any) -> Any:
    async def fake_acompletion(**call_kwargs: Any) -> Any:
        captured.update(call_kwargs)
        return result

    with patch(_ACOMPLETION, new=fake_acompletion):
        return await aresponses_via_chat_completions(
            **{"provider": LLMProvider.MISTRAL, "model": "mistral-small-latest", "api_key": "sk", **kwargs}
        )


async def _aiter(chunks: list[ChatCompletionChunk]) -> AsyncIterator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


# ---------- which providers bridge ----------


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (LLMProvider.MISTRAL, True),
        ("anthropic", True),
        (LLMProvider.OPENAI, False),
        ("not-a-provider", False),
        (None, False),
    ],
)
def test_uses_chat_completions_bridge(provider: Any, expected: bool) -> None:
    assert uses_chat_completions_bridge(provider) is expected


# ---------- request ----------


@pytest.mark.asyncio
async def test_string_input_and_instructions_become_system_and_user_messages() -> None:
    captured: dict[str, Any] = {}
    await _call(captured, _completion({"content": "ok"}), input_data="hi", instructions="be brief")

    assert captured["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert captured["provider"] == LLMProvider.MISTRAL
    assert captured["model"] == "mistral-small-latest"
    assert captured["api_key"] == "sk"
    assert "stream" not in captured


@pytest.mark.asyncio
async def test_input_items_translate_to_chat_messages() -> None:
    captured: dict[str, Any] = {}
    input_data = [
        {"role": "developer", "content": "rules"},
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "low"},
                {"type": "input_file", "file_data": "data:application/pdf;base64,AA", "filename": "a.pdf"},
            ],
        },
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "calling"}]},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": '{"q": 1}'},
        {"type": "function_call", "call_id": "call_2", "name": "lookup", "arguments": '{"q": 2}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "one"},
        {"type": "function_call_output", "call_id": "call_2", "output": [{"type": "input_text", "text": "two"}]},
    ]
    await _call(captured, _completion({"content": "ok"}), input_data=input_data)

    assert captured["messages"] == [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "detail": "low"}},
                {"type": "file", "file": {"file_data": "data:application/pdf;base64,AA", "filename": "a.pdf"}},
            ],
        },
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": 1}'}},
                {"id": "call_2", "type": "function", "function": {"name": "lookup", "arguments": '{"q": 2}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "one"},
        {"role": "tool", "tool_call_id": "call_2", "content": "two"},
    ]


@pytest.mark.asyncio
async def test_function_call_without_a_preceding_assistant_message_opens_one() -> None:
    captured: dict[str, Any] = {}
    input_data = [
        {"role": "user", "content": "go"},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
    ]
    await _call(captured, _completion({"content": "ok"}), input_data=input_data)

    assert captured["messages"][1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    }


@pytest.mark.asyncio
async def test_options_translate_and_responses_only_fields_are_dropped() -> None:
    captured: dict[str, Any] = {}
    await _call(
        captured,
        _completion({"content": "{}"}),
        input_data="hi",
        tools=[{"type": "function", "name": "lookup", "description": "d", "parameters": {"type": "object"}}],
        tool_choice={"type": "function", "name": "lookup"},
        max_output_tokens=64,
        reasoning={"effort": "low", "summary": "auto"},
        text={"format": {"type": "json_schema", "name": "out", "schema": {"type": "object"}, "strict": True}},
        temperature=0.2,
        store=True,
        metadata={"k": "v"},
        include=["reasoning.encrypted_content"],
        truncation="auto",
    )

    assert captured["tools"] == [
        {"type": "function", "function": {"name": "lookup", "description": "d", "parameters": {"type": "object"}}}
    ]
    assert captured["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert captured["max_tokens"] == 64
    assert captured["reasoning_effort"] == "low"
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "out", "schema": {"type": "object"}, "strict": True},
    }
    assert captured["temperature"] == 0.2
    for dropped in ("store", "metadata", "include", "truncation", "text", "reasoning", "max_output_tokens"):
        assert dropped not in captured


@pytest.mark.asyncio
async def test_prompt_cache_key_is_dropped_for_a_provider_without_one() -> None:
    captured: dict[str, Any] = {}
    await _call(captured, _completion({"content": "ok"}), input_data="hi", prompt_cache_key="k")

    assert "prompt_cache_key" not in captured


@pytest.mark.parametrize("field", ["previous_response_id", "conversation", "background"])
@pytest.mark.asyncio
async def test_server_state_fields_are_refused(field: str) -> None:
    with pytest.raises(UnsupportedParameterError, match=field):
        await _call(
            {}, _completion({"content": "ok"}), input_data="hi", **{field: "resp_1" if field != "background" else True}
        )


@pytest.mark.asyncio
async def test_hosted_tool_is_refused() -> None:
    with pytest.raises(UnsupportedParameterError, match="file_search"):
        await _call({}, _completion({"content": "ok"}), input_data="hi", tools=[{"type": "file_search"}])


@pytest.mark.asyncio
async def test_input_item_without_chat_equivalent_is_refused() -> None:
    with pytest.raises(UnsupportedParameterError, match="item_reference"):
        await _call({}, _completion({"content": "ok"}), input_data=[{"type": "item_reference", "id": "msg_1"}])


_IMAGE_PART = {"type": "input_image", "image_url": "data:image/png;base64,AA"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item",
    [
        {"role": "developer", "content": [{"type": "input_text", "text": "rules"}, _IMAGE_PART]},
        {"type": "function_call_output", "call_id": "call_1", "output": [_IMAGE_PART]},
    ],
    ids=["system", "tool-output"],
)
async def test_non_text_part_in_a_string_only_message_is_refused(item: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedParameterError, match="input_image"):
        await _call({}, _completion({"content": "ok"}), input_data=[item])


# ---------- non-streaming response ----------


@pytest.mark.asyncio
async def test_completion_translates_to_a_response() -> None:
    message = {
        "content": "Let me check.",
        "reasoning": "thinking",
        "tool_calls": [{"id": "call_9", "type": "function", "function": {"name": "lookup", "arguments": '{"q":1}'}}],
    }
    result = await _call(
        {},
        _completion(message, finish_reason="tool_calls"),
        input_data="hi",
        tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
    )

    assert isinstance(result, Response)
    assert result.status == "completed"
    assert result.model == "mistral-small-latest"
    assert [item.type for item in result.output] == ["reasoning", "message", "function_call"]
    assert result.output_text == "Let me check."
    call = result.output[2]
    assert call.call_id == "call_9"  # type: ignore[union-attr]
    assert call.name == "lookup"  # type: ignore[union-attr]
    assert call.arguments == '{"q":1}'  # type: ignore[union-attr]
    assert result.tools[0].name == "lookup"  # type: ignore[union-attr]
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.total_tokens == 18
    assert result.usage.input_tokens_details.cached_tokens == 3
    assert result.usage.output_tokens_details.reasoning_tokens == 2


@pytest.mark.asyncio
async def test_reasoning_item_id_carries_the_gateway_prefix() -> None:
    result = await _call({}, _completion({"content": "ok", "reasoning": "thinking"}), input_data="hi")

    assert result.output[0].type == "reasoning"
    assert result.output[0].id.startswith(REASONING_ITEM_ID_PREFIX)


def test_echoed_bridge_reasoning_is_stripped_but_a_providers_own_survives() -> None:
    own = {"type": "reasoning", "id": "rs_68a1b2c3", "summary": []}
    minted = {"type": "reasoning", "id": f"{REASONING_ITEM_ID_PREFIX}abc", "summary": []}
    minted_reference = {"type": "item_reference", "id": f"{REASONING_ITEM_ID_PREFIX}abc"}
    own_reference = {"type": "item_reference", "id": "rs_68a1b2c3"}
    message = {"role": "user", "content": "hi"}

    result = _strip_gateway_minted_items([message, minted, minted_reference, own, own_reference])

    assert result == [message, own, own_reference]


@pytest.mark.asyncio
async def test_length_finish_is_an_incomplete_response() -> None:
    result = await _call({}, _completion({"content": "cut"}, finish_reason="length"), input_data="hi")

    assert result.status == "incomplete"
    assert result.incomplete_details.reason == "max_output_tokens"


# ---------- streaming ----------


async def _stream(captured: dict[str, Any], chunks: list[ChatCompletionChunk], **kwargs: Any) -> list[Any]:
    stream = await _call(captured, _aiter(chunks), input_data="hi", stream=True, **kwargs)
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_text_stream_emits_the_responses_event_sequence() -> None:
    captured: dict[str, Any] = {}
    events = await _stream(
        captured,
        [
            _chunk({"role": "assistant", "content": "Hel"}),
            _chunk({"content": "lo"}),
            _chunk({}, finish_reason="stop"),
            _chunk(None, usage=_USAGE),
        ],
    )

    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert events[6].text == "Hello"
    completed = events[-1].response
    assert completed.status == "completed"
    assert completed.output_text == "Hello"
    assert completed.usage.input_tokens == 11
    assert completed.id == events[0].response.id


@pytest.mark.asyncio
async def test_streamed_tool_call_fragments_accumulate_into_a_function_call() -> None:
    events = await _stream(
        {},
        [
            _chunk({"reasoning": "plan"}),
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q"'},
                        }
                    ]
                }
            ),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}),
            _chunk({}, finish_reason="tool_calls"),
            _chunk(None, usage=_USAGE),
        ],
    )

    types = [event.type for event in events]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    added = events[8]
    assert added.item.type == "function_call"
    assert added.item.name == "lookup"
    assert added.output_index == 1
    assert events[11].arguments == '{"q": 1}'
    final = events[-1].response.output
    assert [item.type for item in final] == ["reasoning", "function_call"]
    assert final[1].call_id == "call_1"
    assert final[1].arguments == '{"q": 1}'


@pytest.mark.asyncio
async def test_streamed_length_finish_ends_with_response_incomplete() -> None:
    events = await _stream({}, [_chunk({"content": "cut"}, finish_reason="length")])

    assert events[-1].type == "response.incomplete"
    assert events[-1].response.incomplete_details.reason == "max_output_tokens"


@pytest.mark.asyncio
async def test_failure_on_first_chunk_surfaces_before_any_event() -> None:
    async def failing() -> AsyncIterator[ChatCompletionChunk]:
        raise RuntimeError("upstream down")
        yield  # pragma: no cover

    stream = await _call({}, failing(), input_data="hi", stream=True)
    with pytest.raises(RuntimeError, match="upstream down"):
        await anext(stream)


# ---------- gateway tool loop over the bridge ----------


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


_TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "fetch_url", "arguments": '{"u": "x"}'}}


@pytest.mark.asyncio
async def test_tool_loop_replays_its_own_turn_through_the_bridge() -> None:
    replies = iter(
        [_completion({"content": None, "tool_calls": [_TOOL_CALL]}, "tool_calls"), _completion({"content": "done"})]
    )
    sent: list[list[dict[str, Any]]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        sent.append(kwargs["messages"])
        return next(replies)

    pool = _Pool()
    with patch(_ACOMPLETION, new=fake_acompletion):
        result = await responses_tool_loop(
            completion_kwargs={
                "provider": LLMProvider.MISTRAL,
                "model": "mistral-small-latest",
                "input_data": [{"role": "user", "content": "fetch x"}],
            },
            pool=cast(Any, pool),
            max_iterations=3,
        )

    assert pool.calls == [("fetch_url", {"u": "x"})]
    assert sent[1][1:] == [
        {"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "page text"},
    ]
    assert result.output_text == "done"


@pytest.mark.asyncio
async def test_streaming_tool_loop_runs_a_bridged_tool_call() -> None:
    turns = iter(
        [
            [
                _chunk({"tool_calls": [{"index": 0, **_TOOL_CALL}]}),
                _chunk({}, finish_reason="tool_calls"),
                _chunk(None, usage=_USAGE),
            ],
            [_chunk({"content": "done"}, finish_reason="stop"), _chunk(None, usage=_USAGE)],
        ]
    )
    sent: list[list[dict[str, Any]]] = []

    async def fake_acompletion(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        sent.append(kwargs["messages"])
        return _aiter(next(turns))

    pool = _Pool()
    with patch(_ACOMPLETION, new=fake_acompletion):
        events: list[Any] = [
            event
            async for event in responses_tool_loop_stream(
                completion_kwargs={
                    "provider": LLMProvider.MISTRAL,
                    "model": "mistral-small-latest",
                    "input_data": [{"role": "user", "content": "fetch x"}],
                    "stream": True,
                },
                pool=cast(Any, pool),
                max_iterations=3,
            )
        ]

    assert pool.calls == [("fetch_url", {"u": "x"})]
    assert sent[1][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "page text"}
    types = [event.type for event in events]
    assert types.count("response.created") == 1
    assert types[-1] == "response.completed"
    assert "function_call" not in [getattr(getattr(event, "item", None), "type", None) for event in events]
    assert events[-1].response.output_text == "done"


# ---------- /responses missing on a custom api_base ----------


def _missing_route(status_code: int = 404, message: str = "Not Found") -> AnyLLMError:
    return AnyLLMError(message, provider_name="openai", status_code=status_code)


async def _call_responses(native_error: AnyLLMError, **kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
    chat_calls: list[dict[str, Any]] = []

    async def native(**_: Any) -> Any:
        raise native_error

    async def fake_acompletion(**call_kwargs: Any) -> Any:
        chat_calls.append(call_kwargs)
        return _completion({"content": "pong"})

    request = {
        "provider": LLMProvider.OPENAI,
        "model": "local-model",
        "api_key": "sk",
        "api_base": "http://llm.internal/v1",
        "input_data": "ping",
        **kwargs,
    }
    with patch(_ACOMPLETION, new=fake_acompletion):
        return await call_responses(native, request), chat_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 405, 501])
async def test_missing_responses_route_on_a_custom_api_base_falls_back_to_chat(status_code: int) -> None:
    result, chat_calls = await _call_responses(_missing_route(status_code))

    assert isinstance(result, Response)
    assert result.output_text == "pong"
    assert chat_calls[0]["api_base"] == "http://llm.internal/v1"
    assert chat_calls[0]["messages"] == [{"role": "user", "content": "ping"}]


@pytest.mark.asyncio
async def test_not_found_without_a_custom_api_base_is_not_retried() -> None:
    with pytest.raises(AnyLLMError, match="Not Found"):
        await _call_responses(_missing_route(), api_base=None)


@pytest.mark.asyncio
async def test_other_native_failures_are_not_retried() -> None:
    with pytest.raises(AnyLLMError):
        await _call_responses(AnyLLMError("bad request", status_code=400))


@pytest.mark.asyncio
async def test_request_the_bridge_cannot_translate_reraises_the_native_error() -> None:
    error = ModelNotFoundError("no such item", status_code=404)

    with pytest.raises(ModelNotFoundError) as raised:
        await _call_responses(error, previous_response_id="resp_1")

    assert raised.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["Item with id 'rs_1' not found", "The model `x` does not exist"])
async def test_not_found_naming_an_item_or_model_is_not_taken_for_a_missing_route(message: str) -> None:
    with pytest.raises(AnyLLMError, match="not found|does not exist"):
        await _call_responses(_missing_route(404, message))


@pytest.mark.asyncio
async def test_item_or_model_wording_does_not_hold_back_405_or_501() -> None:
    result, _ = await _call_responses(_missing_route(405, "Method not allowed for this model"))

    assert result.output_text == "pong"


@pytest.mark.asyncio
async def test_chat_side_unsupported_parameter_is_not_replaced_by_the_native_error() -> None:
    async def native(**_: Any) -> Any:
        raise _missing_route()

    async def refusing_acompletion(**_: Any) -> Any:
        raise UnsupportedParameterError("reasoning_effort", "openai")

    request = {"provider": LLMProvider.OPENAI, "model": "m", "api_base": "http://llm.internal/v1", "input_data": "hi"}
    with patch(_ACOMPLETION, new=refusing_acompletion), pytest.raises(UnsupportedParameterError):
        await call_responses(native, request)


@pytest.mark.asyncio
async def test_responses_only_extra_body_keys_stay_out_of_the_chat_request() -> None:
    _, chat_calls = await _call_responses(
        _missing_route(),
        extra_body={"input": [{"role": "user", "content": "ping"}], "client_metadata": {"a": 1}, "top_k": 4},
    )

    assert chat_calls[0]["extra_body"] == {"top_k": 4}


@pytest.mark.asyncio
async def test_extra_body_holding_only_responses_keys_is_dropped() -> None:
    _, chat_calls = await _call_responses(_missing_route(), extra_body={"input": []})

    assert "extra_body" not in chat_calls[0]


@pytest.mark.asyncio
async def test_streaming_request_falls_back_to_a_bridged_event_stream() -> None:
    async def native(**_: Any) -> Any:
        raise _missing_route()

    async def fake_acompletion(**call_kwargs: Any) -> Any:
        assert call_kwargs["stream"] is True
        return _aiter([_chunk({"role": "assistant", "content": "po"}), _chunk({"content": "ng"}, "stop")])

    request = {
        "provider": LLMProvider.OPENAI,
        "model": "m",
        "api_base": "http://llm.internal/v1",
        "input_data": "hi",
        "stream": True,
    }
    with patch(_ACOMPLETION, new=fake_acompletion):
        stream = await call_responses(native, request)
        types = [event.type async for event in stream]

    assert types[0] == "response.created"
    assert types[-1] == "response.completed"
