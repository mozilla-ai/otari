"""The two Messages conversions agree: any-llm's to chat completions, and Otari's to the Responses API.

Otari serves a Messages request through the Responses API for an instance that
sets ``messages_api: responses`` (``services/inference/_messages_responses_bridge``),
and through any-llm's ``messages_params_to_completion_params`` otherwise. The two
target different wire formats, so each output is reduced here to the same
neutral shape: what the model is told, by whom, and with which tools. A fix to
one conversion that the other lacks then fails this test, rather than making the
same request behave differently depending on an instance setting.

Deliberate differences are not compared, and are listed in ``_KNOWN_DIFFERENCES``
so a reader sees them in one place.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from any_llm.types.messages import MessagesParams
from any_llm.utils.messages_compat import messages_params_to_completion_params

from gateway.services.inference._messages_responses_bridge import responses_kwargs_for

_KNOWN_DIFFERENCES = {
    "thinking": "any-llm replays thinking text as reasoning_content; the Responses path sends back only "
    "reasoning it signed itself, as encrypted reasoning items, and drops the rest.",
    "effort": "any-llm maps a thinking budget onto minimal..xhigh; the Responses path uses low..high, "
    "which every OpenAI reasoning model accepts.",
    "tool result media": "any-llm moves a tool result's images into a following user message, since a chat "
    "tool message takes text only; a Responses function output carries them itself.",
    "block order": "any-llm puts an assistant turn's text before its tool calls; the Responses path keeps "
    "the original order. Both are compared per turn, as text plus calls.",
    "stop_sequences": "any-llm forwards them as stop; the Responses API has no equivalent, so it refuses them.",
    "user documents": "any-llm passes a document block in a user message through unconverted (it converts "
    "one only inside a tool result); the Responses path sends it as input_file or text. Pinned by "
    "test_a_user_document_converts_the_same_way, which fails strict once any-llm converts it.",
}

_DOCUMENTS: dict[str, Any] = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "PDF"}},
                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "notes"}},
            ],
        }
    ]
}

_TOOLS = [
    {"name": "read", "description": "Read a file", "input_schema": {"type": "object", "properties": {}}},
    {"name": "bare", "input_schema": {"type": "object"}},
]

_FIXTURES: dict[str, dict[str, Any]] = {
    "plain": {"messages": [{"role": "user", "content": "hi"}]},
    "system string": {"system": "be brief", "messages": [{"role": "user", "content": "hi"}]},
    "system blocks": {
        "system": [{"type": "text", "text": "be brief"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    },
    "sampling": {"temperature": 0.3, "top_p": 0.9, "messages": [{"role": "user", "content": "hi"}]},
    "images": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                ],
            }
        ]
    },
    "tool loop": {
        "tools": _TOOLS,
        "tool_choice": {"type": "auto"},
        "messages": [
            {"role": "user", "content": "read a and b"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading."},
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
                        "content": [{"type": "text", "text": "missing"}],
                        "is_error": True,
                    },
                    {"type": "tool_result", "tool_use_id": "call_c", "content": "", "is_error": True},
                    {"type": "text", "text": "go on"},
                ],
            },
            {"role": "assistant", "content": "Done."},
        ],
    },
    "forced tool, sequential": {
        "tools": _TOOLS,
        "tool_choice": {"type": "tool", "name": "read", "disable_parallel_tool_use": True},
        "messages": [{"role": "user", "content": "read"}],
    },
    "any tool": {"tools": _TOOLS, "tool_choice": {"type": "any"}, "messages": [{"role": "user", "content": "x"}]},
    "no tool": {"tools": _TOOLS, "tool_choice": {"type": "none"}, "messages": [{"role": "user", "content": "x"}]},
}


def _chat_view(params: dict[str, Any]) -> dict[str, Any]:
    turns: list[Any] = []
    for message in params["messages"]:
        role = message["role"]
        content = message.get("content")
        if role == "system":
            turns.append(("system", content))
        elif role == "assistant":
            calls = [
                (call["id"], call["function"]["name"], json.loads(call["function"]["arguments"]))
                for call in message.get("tool_calls") or []
            ]
            turns.append(("assistant", content or "", calls))
        elif role == "tool":
            turns.append(("result", message["tool_call_id"], message["content"]))
        elif isinstance(content, str):
            turns.append(("user", [("text", content)]))
        else:
            turns.append(("user", [_chat_part(part) for part in content]))
    tool_choice = params.get("tool_choice")
    if isinstance(tool_choice, dict):
        tool_choice = ("function", tool_choice["function"]["name"])
    return {
        "turns": turns,
        "max_tokens": params.get("max_tokens"),
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "tools": [
            (tool["function"]["name"], tool["function"].get("description") or "", tool["function"]["parameters"])
            for tool in params.get("tools") or []
        ],
        "tool_choice": tool_choice,
        "parallel_tool_calls": params.get("parallel_tool_calls"),
    }


def _chat_part(part: dict[str, Any]) -> tuple[str, Any]:
    if part["type"] == "text":
        return ("text", part["text"])
    if part["type"] == "image_url":
        return ("image", part["image_url"]["url"])
    if part["type"] == "file":
        return ("file", part["file"]["file_data"])
    return (part["type"], part)


def _responses_view(kwargs: dict[str, Any]) -> dict[str, Any]:
    turns: list[Any] = [("system", kwargs["instructions"])] if kwargs.get("instructions") else []
    for item in kwargs["input_data"]:
        kind = item.get("type") or "message"
        if kind == "reasoning":
            continue
        if kind == "function_call":
            call = (item["call_id"], item["name"], json.loads(item["arguments"]))
            if not turns or turns[-1][0] != "assistant":
                turns.append(("assistant", "", []))
            turns[-1][2].append(call)
        elif kind == "function_call_output":
            turns.append(("result", item["call_id"], item["output"]))
        elif item["role"] == "assistant":
            turns.append(("assistant", item["content"], []))
        elif isinstance(item["content"], str):
            turns.append(("user", [("text", item["content"])]))
        else:
            turns.append(("user", [_responses_part(part) for part in item["content"]]))
    tool_choice = kwargs.get("tool_choice")
    if isinstance(tool_choice, dict):
        tool_choice = ("function", tool_choice["name"])
    return {
        "turns": turns,
        "max_tokens": kwargs.get("max_output_tokens"),
        "temperature": kwargs.get("temperature"),
        "top_p": kwargs.get("top_p"),
        "tools": [
            (tool["name"], tool.get("description") or "", tool["parameters"]) for tool in kwargs.get("tools") or []
        ],
        "tool_choice": tool_choice,
        "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
    }


def _responses_part(part: dict[str, Any]) -> tuple[str, Any]:
    if part["type"] == "input_text":
        return ("text", part["text"])
    if part["type"] == "input_image":
        return ("image", part["image_url"])
    return ("file", part["file_data"])


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_both_conversions_tell_the_model_the_same_thing(name: str) -> None:
    request = {"max_tokens": 512, **_FIXTURES[name]}
    chat = messages_params_to_completion_params(MessagesParams(model="gpt-5", **request))
    responses = responses_kwargs_for({"model": "openai:gpt-5", **request}, instance="openai")
    assert _responses_view(responses) == _chat_view(chat)


@pytest.mark.xfail(strict=True, reason=_KNOWN_DIFFERENCES["user documents"])
def test_a_user_document_converts_the_same_way() -> None:
    request = {"max_tokens": 512, **_DOCUMENTS}
    chat = messages_params_to_completion_params(MessagesParams(model="gpt-5", **request))
    responses = responses_kwargs_for({"model": "openai:gpt-5", **request}, instance="openai")
    assert _responses_view(responses) == _chat_view(chat)


def test_known_differences_are_documented() -> None:
    # A reminder, not a check: removing a difference from one side means deleting its entry here.
    assert all(_KNOWN_DIFFERENCES.values())
