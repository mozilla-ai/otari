"""Inbound stripping of the server-tool blocks the gateway mints itself.

Continuing an Anthropic conversation means echoing the previous assistant turn
back. A gateway-minted ``web_search_tool_result`` carries an
``encrypted_content`` the gateway cannot sign, so it must not reach a provider.
Mirrors ``responses._strip_gateway_minted_items``.
"""

from __future__ import annotations

from typing import Any

import pytest

from gateway.api.routes._pipeline import ToolContext
from gateway.api.routes.messages import _strip_gateway_minted_blocks
from gateway.core.config import GatewayConfig


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


# --- the gate that decides whether stripping runs at all ----------------------


def _tool_ctx(config: GatewayConfig) -> ToolContext:
    """A ToolContext carrying nothing but the two inputs the gate reads."""
    return ToolContext(
        config=config,
        mcp_server_configs=None,
        use_sandbox=False,
        sandbox_tool_entry=None,
        sandbox_url=None,
        sandbox_auth_token=None,
        use_web_search=False,
        web_search_tool_entry=None,
        web_search_url=config.web_search_url,
        web_search_auth_token=None,
        remaining_user_tools=None,
        max_tool_iterations=10,
        tools_header=None,
    )


def test_gate_is_off_without_the_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_WEB_SEARCH_INTERCEPT", raising=False)
    config = GatewayConfig(web_search_url="http://searxng:8080")
    assert _tool_ctx(config).intercepts_web_search is False


def test_gate_is_on_when_opted_in_with_a_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_WEB_SEARCH_INTERCEPT", raising=False)
    config = GatewayConfig(web_search_intercept=True, web_search_url="http://searxng:8080")
    assert _tool_ctx(config).intercepts_web_search is True


def test_gate_is_off_when_opted_in_without_a_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing to intercept to, the keyword was forwarded and the provider ran
    the search, so the blocks in the transcript are its own signed ones. Stripping
    them would break the citations round-trip Anthropic itself established."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_INTERCEPT", raising=False)
    config = GatewayConfig(web_search_intercept=True)
    assert config.web_search_url is None
    assert _tool_ctx(config).intercepts_web_search is False


# --- provenance: only our own blocks are stripped -----------------------------


def _provider_pair() -> list[dict[str, Any]]:
    """What Anthropic returns when it ran the search itself: signed content."""
    return [
        {"type": "server_tool_use", "id": "srvtoolu_prov", "name": "web_search", "input": {"query": "x"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_prov",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://a",
                    "title": "A",
                    "encrypted_content": "ErcBCioIAxgCIiQ4ZDhkOGQ4ZC1hYmNk",
                }
            ],
        },
    ]


def _gateway_pair(tool_use_id: str = "srvtoolu_gw") -> list[dict[str, Any]]:
    """What the gateway mints: the same shape with encrypted_content empty."""
    return [
        {"type": "server_tool_use", "id": tool_use_id, "name": "web_search", "input": {"query": "y"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": [{"type": "web_search_result", "url": "https://b", "title": "B", "encrypted_content": ""}],
        },
    ]


def test_provider_signed_blocks_survive() -> None:
    """A search Anthropic ran and signed must round-trip untouched, even with
    interception on: stripping it would break the citations chain Anthropic owns."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [*_provider_pair(), {"type": "text", "text": "answer"}]},
    ]

    assert _strip_gateway_minted_blocks(messages) == messages


def test_gateway_pair_is_stripped_and_provider_pair_kept_in_one_turn() -> None:
    """A transcript can hold both: one search the provider ran, one the gateway did."""
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [*_provider_pair(), *_gateway_pair(), {"type": "text", "text": "answer"}],
        }
    ]

    kept = _strip_gateway_minted_blocks(messages)[0]["content"]

    assert kept == [*_provider_pair(), {"type": "text", "text": "answer"}]


def test_a_providers_server_tool_use_is_never_orphaned() -> None:
    """The server_tool_use dropped is the one our result answers, matched by id, so a
    provider's pair is never split into an orphan the API would reject."""
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": [*_provider_pair(), *_gateway_pair()]}
    ]

    kept = _strip_gateway_minted_blocks(messages)[0]["content"]

    ids = [b.get("id") for b in kept if b.get("type") == "server_tool_use"]
    result_ids = [b.get("tool_use_id") for b in kept if b.get("type") == "web_search_tool_result"]
    assert ids == ["srvtoolu_prov"]
    assert result_ids == ["srvtoolu_prov"]


def test_a_result_with_no_hits_is_treated_as_ours() -> None:
    """A gateway search that found nothing usable produces an empty content list."""
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_gw", "name": "web_search", "input": {}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_gw", "content": []},
                {"type": "text", "text": "nothing found"},
            ],
        }
    ]

    kept = _strip_gateway_minted_blocks(messages)[0]["content"]
    assert kept == [{"type": "text", "text": "nothing found"}]


def test_a_provider_error_result_is_kept() -> None:
    """An error code the gateway never mints came from the provider and survives."""
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_prov", "name": "web_search", "input": {}},
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_prov",
                    "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"},
                },
            ],
        }
    ]

    assert _strip_gateway_minted_blocks(messages) == messages


def test_a_max_uses_error_result_and_its_call_are_stripped() -> None:
    """A capped gateway search must not be echoed back to the provider."""
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_gw", "name": "web_search", "input": {}},
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_gw",
                    "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
                },
                {"type": "text", "text": "done"},
            ],
        }
    ]

    assert _strip_gateway_minted_blocks(messages) == [
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
    ]
