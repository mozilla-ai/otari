"""Per-format tool-set normalization for the Reprise v0 fingerprint (otari-ai#1647).

The loop merges the caller's tools with the gateway's converted ones and the
merged list is format-shaped, so hashing it as built never groups across
formats. The normalizer unwraps each shape into one triple, which then feeds two
different values: ``tool_set_hash`` (names only, a fingerprint input) and
``tool_definitions_hash`` (names, descriptions, schemas; payload only). They are
separate because they change for unrelated reasons. A tool's description and
schema arrive in the same live ``list_tools()`` response and the pool is rebuilt
per request, so an upstream server adding one optional property, backward
compatibly and breaking nothing, would split every pile in the workspace if the
key covered them.
"""

from typing import Any

import pytest

from gateway.core.observation import NormalizedTool, tool_definitions_hash, tool_set_hash
from gateway.services._tool_loop import ToolLoopStrategy
from gateway.services.mcp_loop import _CHAT_STRATEGY
from gateway.services.mcp_loop_messages import _MESSAGES_STRATEGY
from gateway.services.mcp_loop_responses import _RESPONSES_STRATEGY

_LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"repo": {"type": "string"}},
    "required": ["repo"],
}
_GET_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"number": {"type": "integer"}}}

_LIST_DESCRIPTION = "List open issues."
_GET_DESCRIPTION = "Fetch one issue."


def _chat_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_issues",
                "description": _LIST_DESCRIPTION,
                "parameters": _LIST_SCHEMA,
            },
        },
        {
            "type": "function",
            "function": {"name": "get_issue", "description": _GET_DESCRIPTION, "parameters": _GET_SCHEMA},
        },
    ]


def _messages_tools() -> list[dict[str, Any]]:
    return [
        {"name": "list_issues", "description": _LIST_DESCRIPTION, "input_schema": _LIST_SCHEMA},
        {"name": "get_issue", "description": _GET_DESCRIPTION, "input_schema": _GET_SCHEMA},
    ]


def _responses_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "list_issues",
            "description": _LIST_DESCRIPTION,
            "parameters": _LIST_SCHEMA,
        },
        {"type": "function", "name": "get_issue", "description": _GET_DESCRIPTION, "parameters": _GET_SCHEMA},
    ]


_FORMATS: list[tuple[ToolLoopStrategy[Any, Any], list[dict[str, Any]]]] = [
    (_CHAT_STRATEGY, _chat_tools()),
    (_MESSAGES_STRATEGY, _messages_tools()),
    (_RESPONSES_STRATEGY, _responses_tools()),
]


def test_one_tool_set_hash_across_the_three_formats() -> None:
    hashes = {tool_set_hash(strategy.normalize_tools(tools)) for strategy, tools in _FORMATS}

    assert len(hashes) == 1


def test_one_tool_definitions_hash_across_the_three_formats() -> None:
    """The three converters only rewrap the same JSON Schema object."""
    hashes = {tool_definitions_hash(strategy.normalize_tools(tools)) for strategy, tools in _FORMATS}

    assert len(hashes) == 1


@pytest.mark.parametrize(
    ("strategy", "tools"),
    [
        pytest.param(_CHAT_STRATEGY, _chat_tools(), id="chat"),
        pytest.param(_MESSAGES_STRATEGY, _messages_tools(), id="messages"),
        pytest.param(_RESPONSES_STRATEGY, _responses_tools(), id="responses"),
    ],
)
def test_normalizer_recovers_the_same_triple_from_every_shape(
    strategy: ToolLoopStrategy[Any, Any], tools: list[dict[str, Any]]
) -> None:
    assert strategy.normalize_tools(tools) == [
        NormalizedTool("list_issues", _LIST_DESCRIPTION, _LIST_SCHEMA),
        NormalizedTool("get_issue", _GET_DESCRIPTION, _GET_SCHEMA),
    ]


def test_reordering_the_tools_changes_neither_hash() -> None:
    forward = _CHAT_STRATEGY.normalize_tools(_chat_tools())
    reverse = _CHAT_STRATEGY.normalize_tools(list(reversed(_chat_tools())))

    assert tool_set_hash(forward) == tool_set_hash(reverse)
    assert tool_definitions_hash(forward) == tool_definitions_hash(reverse)


def test_an_added_optional_schema_property_moves_only_the_definitions_hash() -> None:
    """The upstream server broke nothing, so the workspace's piles must survive it."""
    widened = _chat_tools()
    widened[0]["function"]["parameters"] = {
        **_LIST_SCHEMA,
        "properties": {**_LIST_SCHEMA["properties"], "state": {"type": "string"}},
    }

    before = _CHAT_STRATEGY.normalize_tools(_chat_tools())
    after = _CHAT_STRATEGY.normalize_tools(widened)

    assert tool_set_hash(before) == tool_set_hash(after)
    assert tool_definitions_hash(before) != tool_definitions_hash(after)


def test_a_reworded_description_moves_only_the_definitions_hash() -> None:
    reworded = _chat_tools()
    reworded[0]["function"]["description"] = "List the issues that are still open."

    before = _CHAT_STRATEGY.normalize_tools(_chat_tools())
    after = _CHAT_STRATEGY.normalize_tools(reworded)

    assert tool_set_hash(before) == tool_set_hash(after)
    assert tool_definitions_hash(before) != tool_definitions_hash(after)


def test_reserializing_an_unchanged_schema_reads_as_the_same_tool() -> None:
    """Canonical JSON, so key order and whitespace are not a tool change."""
    reordered = _chat_tools()
    reordered[0]["function"]["parameters"] = {
        "required": ["repo"],
        "properties": {"repo": {"type": "string"}},
        "type": "object",
    }

    assert tool_definitions_hash(_CHAT_STRATEGY.normalize_tools(_chat_tools())) == tool_definitions_hash(
        _CHAT_STRATEGY.normalize_tools(reordered)
    )


def test_adding_a_tool_moves_both_hashes() -> None:
    extended = [
        *_chat_tools(),
        {
            "type": "function",
            "function": {"name": "close_issue", "description": "Close it.", "parameters": _GET_SCHEMA},
        },
    ]

    before = _CHAT_STRATEGY.normalize_tools(_chat_tools())
    after = _CHAT_STRATEGY.normalize_tools(extended)

    assert tool_set_hash(before) != tool_set_hash(after)
    assert tool_definitions_hash(before) != tool_definitions_hash(after)


def test_removing_a_tool_moves_both_hashes() -> None:
    before = _CHAT_STRATEGY.normalize_tools(_chat_tools())
    after = _CHAT_STRATEGY.normalize_tools(_chat_tools()[:1])

    assert tool_set_hash(before) != tool_set_hash(after)
    assert tool_definitions_hash(before) != tool_definitions_hash(after)


def test_responses_native_tools_stay_distinct_without_a_name_key() -> None:
    """A Responses server tool carries its identity in ``type``, with no ``name`` at all.

    Reading ``name`` alone collapses every one of them to the empty string, so a
    workspace swapping web search for file search is fingerprinted as an unchanged
    tool set and the two automations land in one pile.
    """
    native: list[dict[str, Any]] = [
        {"type": "web_search"},
        {"type": "file_search", "vector_store_ids": ["vs_1"]},
        {"type": "mcp", "server_label": "github", "server_url": "https://example.invalid"},
    ]

    normalized = _RESPONSES_STRATEGY.normalize_tools(native)

    assert [tool.name for tool in normalized] == ["web_search", "file_search", "mcp"]
    assert len({tool_set_hash([tool]) for tool in normalized}) == 3


def test_a_named_tool_is_not_renamed_by_its_type() -> None:
    """``name`` wins wherever a format supplies both; only its absence falls back."""
    assert _RESPONSES_STRATEGY.normalize_tools(_responses_tools())[0].name == "list_issues"
    assert _CHAT_STRATEGY.normalize_tools(_chat_tools())[0].name == "list_issues"
    assert _MESSAGES_STRATEGY.normalize_tools(
        [{"type": "web_search_20250305", "name": "web_search"}]
    ) == [NormalizedTool("web_search", "", {})]


def test_a_provider_native_tool_still_counts_by_name() -> None:
    """A server-side tool the gateway never converts is still part of the set."""
    native = [*_messages_tools(), {"type": "web_search_20250305", "name": "web_search"}]

    normalized = _MESSAGES_STRATEGY.normalize_tools(native)

    assert normalized[-1] == NormalizedTool("web_search", "", {})
    assert tool_set_hash(normalized) != tool_set_hash(_MESSAGES_STRATEGY.normalize_tools(_messages_tools()))
