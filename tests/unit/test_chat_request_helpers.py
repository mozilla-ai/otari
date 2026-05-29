"""Unit tests for the tool-extraction helpers in `gateway.api.routes._tools`.

Only the explicit gateway-managed types (`otari_code_execution` /
`otari_web_search`) are extracted and run by the gateway. Provider-named
keywords (OpenAI `code_interpreter`, Anthropic versioned `code_execution_*` /
`web_search_*`, and the bare `code_execution` / `web_search`) are *not*
extracted — they stay in `tools[]` and pass through to the upstream provider,
which executes them server-side.
"""

from __future__ import annotations

from gateway.api.routes._tools import _extract_code_execution_tool, _extract_web_search_tool


def test_extracts_otari_code_execution() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "otari_code_execution"}])
    assert entry == {"type": "otari_code_execution"}
    assert remaining is None


def test_passes_through_gateway_native_short_form() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "code_execution"}])
    assert entry is None
    assert remaining == [{"type": "code_execution"}]


def test_passes_through_openai_code_interpreter() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "code_interpreter"}])
    assert entry is None
    assert remaining == [{"type": "code_interpreter"}]


def test_passes_through_anthropic_versioned_type() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "code_execution_20250825"}])
    assert entry is None
    assert remaining == [{"type": "code_execution_20250825"}]


def test_passes_through_future_anthropic_version() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "code_execution_20991231"}])
    assert entry is None
    assert remaining == [{"type": "code_execution_20991231"}]


def test_passes_through_unrelated_tools_alongside_otari() -> None:
    user_tool = {"type": "function", "function": {"name": "get_weather"}}
    entry, remaining = _extract_code_execution_tool([user_tool, {"type": "otari_code_execution"}])
    assert entry == {"type": "otari_code_execution"}
    assert remaining == [user_tool]


def test_provider_keywords_stay_in_remaining_for_passthrough() -> None:
    # A request mixing the gateway-managed type with a provider-named one:
    # the gateway runs the otari_* entry, the provider-named entry passes
    # through untouched.
    entry, remaining = _extract_code_execution_tool(
        [
            {"type": "otari_code_execution", "purpose_hint": "first"},
            {"type": "code_interpreter"},
        ]
    )
    assert entry == {"type": "otari_code_execution", "purpose_hint": "first"}
    assert remaining == [{"type": "code_interpreter"}]


def test_takes_only_the_first_otari_entry() -> None:
    entry, remaining = _extract_code_execution_tool(
        [
            {"type": "otari_code_execution", "purpose_hint": "first"},
            {"type": "otari_code_execution", "purpose_hint": "second"},
        ]
    )
    assert entry == {"type": "otari_code_execution", "purpose_hint": "first"}
    assert remaining == [{"type": "otari_code_execution", "purpose_hint": "second"}]


def test_returns_no_entry_when_absent() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "function", "function": {"name": "f"}}])
    assert entry is None
    assert remaining == [{"type": "function", "function": {"name": "f"}}]


def test_empty_tools_returns_no_entry() -> None:
    entry, remaining = _extract_code_execution_tool(None)
    assert entry is None
    assert remaining is None


def test_does_not_match_unrelated_types_starting_with_otari() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "otari_code_review"}])
    assert entry is None
    assert remaining == [{"type": "otari_code_review"}]


def test_non_string_type_does_not_match() -> None:
    entry, _ = _extract_code_execution_tool([{"type": None}, {"type": 42}])
    assert entry is None


# --- web_search extraction ---------------------------------------------------


def test_web_search_extracts_otari_web_search() -> None:
    entry, remaining = _extract_web_search_tool([{"type": "otari_web_search"}])
    assert entry == {"type": "otari_web_search"}
    assert remaining is None


def test_web_search_passes_through_gateway_native_short_form() -> None:
    entry, remaining = _extract_web_search_tool([{"type": "web_search"}])
    assert entry is None
    assert remaining == [{"type": "web_search"}]


def test_web_search_passes_through_anthropic_versioned_type() -> None:
    entry, remaining = _extract_web_search_tool([{"type": "web_search_20250305"}])
    assert entry is None
    assert remaining == [{"type": "web_search_20250305"}]


def test_web_search_passes_through_future_anthropic_version() -> None:
    entry, remaining = _extract_web_search_tool([{"type": "web_search_20991231"}])
    assert entry is None
    assert remaining == [{"type": "web_search_20991231"}]


def test_web_search_passes_through_unrelated_tools_alongside_otari() -> None:
    user_tool = {"type": "function", "function": {"name": "get_weather"}}
    entry, remaining = _extract_web_search_tool([user_tool, {"type": "otari_web_search"}])
    assert entry == {"type": "otari_web_search"}
    assert remaining == [user_tool]


def test_web_search_does_not_match_code_execution() -> None:
    entry, _ = _extract_web_search_tool([{"type": "otari_code_execution"}])
    assert entry is None


def test_web_search_carries_per_tool_config_through() -> None:
    entry, _ = _extract_web_search_tool(
        [{"type": "otari_web_search", "max_results": 3, "allowed_domains": ["docs.python.org"]}]
    )
    assert entry is not None
    assert entry["max_results"] == 3
    assert entry["allowed_domains"] == ["docs.python.org"]
