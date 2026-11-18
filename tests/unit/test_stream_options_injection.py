"""Unit tests for automatic stream_options injection behavior."""

from typing import Any

import pytest
from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice, CompletionUsage

from gateway.api.routes import chat
from gateway.api.routes.chat import ChatCompletionRequest


def test_stream_options_accepted_by_request_model() -> None:
    request = ChatCompletionRequest(
        model="openai:gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    assert request.stream_options == {"include_usage": True}


def test_stream_options_defaults_to_none() -> None:
    request = ChatCompletionRequest(
        model="openai:gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert request.stream_options is None


def test_stream_options_excluded_when_unset() -> None:
    request = ChatCompletionRequest(
        model="openai:gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    dumped = request.model_dump(exclude_unset=True)
    assert "stream_options" not in dumped


def test_stream_options_included_when_set() -> None:
    request = ChatCompletionRequest(
        model="openai:gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": False},
    )
    dumped = request.model_dump(exclude_unset=True)
    assert dumped["stream_options"] == {"include_usage": False}


def _build_completion_kwargs(
    request_body: dict[str, Any],
    provider_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = ChatCompletionRequest(**request_body)
    request_fields = request.model_dump(exclude_unset=True)
    completion_kwargs: dict[str, Any] = {**(provider_kwargs or {}), **request_fields}

    if request.stream and completion_kwargs.get("stream_options") is None:
        completion_kwargs["stream_options"] = {"include_usage": True}

    return completion_kwargs


def test_auto_injects_stream_options_for_streaming() -> None:
    kwargs = _build_completion_kwargs(
        {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
    )
    assert kwargs["stream_options"] == {"include_usage": True}


def test_no_injection_for_non_streaming() -> None:
    kwargs = _build_completion_kwargs(
        {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    assert "stream_options" not in kwargs


def test_stream_asks_for_usage_when_the_caller_turned_it_off() -> None:
    """The usage chunk is what a stream is billed on, so the provider is asked for it either way."""
    kwargs = chat._ADAPTER.prepare_stream_kwargs(
        {"stream_options": {"include_usage": False}},
    )
    assert kwargs["stream_options"]["include_usage"] is True


def test_stream_asks_for_usage_on_empty_options() -> None:
    kwargs = chat._ADAPTER.prepare_stream_kwargs(
        {"stream_options": {}},
    )
    assert kwargs["stream_options"] == {"include_usage": True}


def test_stream_keeps_custom_options_and_asks_for_usage() -> None:
    kwargs = chat._ADAPTER.prepare_stream_kwargs(
        {"stream_options": {"custom_field": "value"}},
    )
    assert kwargs["stream_options"] == {"custom_field": "value", "include_usage": True}


def test_preserves_client_stream_options() -> None:
    custom = {"include_usage": False, "custom_field": "value"}
    kwargs = _build_completion_kwargs(
        {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": custom,
        }
    )
    assert kwargs["stream_options"] == custom


def _chunk(*, text: str | None = None, usage: bool = False) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="c1",
        choices=[] if text is None else [ChunkChoice(delta=ChoiceDelta(content=text), index=0)],
        created=0,
        model="m",
        object="chat.completion.chunk",
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2) if usage else None,
    )


@pytest.mark.parametrize(
    ("stream_options", "forwarded"),
    [(None, True), ({"include_usage": True}, True), ({"include_usage": False}, False), ({}, False)],
)
def test_the_usage_chunk_reaches_the_caller_only_when_it_asked_for_usage(
    stream_options: dict[str, Any] | None, forwarded: bool
) -> None:
    """The provider is asked for usage either way; a caller that turned it off still gets a stream without it."""
    adapter = chat._ADAPTER.for_stream_options(stream_options)
    assert bool(adapter.format_chunk(_chunk(usage=True))) is forwarded
    assert adapter.format_chunk(_chunk(text="hi")).startswith("data: ")
