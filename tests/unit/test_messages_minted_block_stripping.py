"""Inbound stripping of the server-tool blocks the gateway mints itself.

Continuing an Anthropic conversation means echoing the previous assistant turn
back. A gateway-minted ``web_search_tool_result`` carries an
``encrypted_content`` the gateway cannot sign, so it must not reach a provider.
Mirrors ``responses._strip_gateway_minted_items``.
"""

from __future__ import annotations

from typing import Any

from gateway.api.routes.messages import _strip_gateway_minted_blocks


def test_strips_the_minted_pair_but_keeps_the_text() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "what is the latest python?"},
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "python"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
                {"type": "text", "text": "Python 3.14."},
            ],
        },
        {"role": "user", "content": "and the one before?"},
    ]

    out = _strip_gateway_minted_blocks(messages)

    assert out[0] == messages[0]
    assert out[1]["content"] == [{"type": "text", "text": "Python 3.14."}]
    assert out[2] == messages[2]


def test_drops_a_message_left_with_no_content() -> None:
    """An empty content array is rejected by the API, and a turn that held nothing
    but the minted pair has nothing left to say."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
            ],
        },
    ]

    out = _strip_gateway_minted_blocks(messages)

    assert out == [{"role": "user", "content": "hi"}]


def test_leaves_a_transcript_without_minted_blocks_untouched() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]

    assert _strip_gateway_minted_blocks(messages) == messages


def test_leaves_string_content_untouched() -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "plain string"}]
    assert _strip_gateway_minted_blocks(messages) == messages


def test_keeps_real_tool_use_and_tool_result_blocks() -> None:
    """A caller's own client-side tool round-trip is not a gateway-minted block."""
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}]},
    ]

    assert _strip_gateway_minted_blocks(messages) == messages


def test_non_list_input_passes_through() -> None:
    assert _strip_gateway_minted_blocks(None) is None
    assert _strip_gateway_minted_blocks("not a list") == "not a list"
