"""``mock_response`` stands in for the provider call, and never reaches a provider."""

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException

from gateway.api.routes._pipeline import _refuse_mock_in_tool_loop
from gateway.api.routes.chat import _ADAPTER

MESSAGES = [{"role": "user", "content": "x" * 40}]


def test_a_mocked_call_answers_with_the_text_and_estimated_usage() -> None:
    kwargs: dict[str, Any] = {"model": "openai:gpt-5-mini", "messages": MESSAGES, "mock_response": "hello world!"}

    completion = asyncio.run(_ADAPTER.call_provider(kwargs))

    assert completion.choices[0].message.content == "hello world!"
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 3
    # Taken out, so nothing downstream could forward it.
    assert "mock_response" not in kwargs


def test_a_mocked_stream_ends_with_its_usage() -> None:
    kwargs: dict[str, Any] = {"model": "openai:gpt-5-mini", "messages": MESSAGES, "mock_response": "hi"}

    async def collect() -> list[Any]:
        return [chunk async for chunk in await _ADAPTER.open_provider_stream(kwargs)]

    first, last = asyncio.run(collect())

    assert first.choices[0].delta.content == "hi"
    assert first.choices[0].finish_reason == "stop"
    assert last.choices == []
    assert last.usage is not None
    assert last.usage.total_tokens == 11


def test_the_tool_loop_refuses_a_mock() -> None:
    with pytest.raises(HTTPException) as refused:
        _refuse_mock_in_tool_loop(_ADAPTER, {"mock_response": "hi"})

    assert refused.value.status_code == 400


def test_the_tool_loop_takes_a_request_without_a_mock() -> None:
    _refuse_mock_in_tool_loop(_ADAPTER, {"model": "openai:gpt-5-mini"})
