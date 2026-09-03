"""Unit tests for the tool-extraction helpers in `gateway.api.routes._tools`.

By default only the explicit gateway-managed types (`otari_code_execution` /
`otari_web_search`) are extracted and run by the gateway. Provider-named
keywords (OpenAI `code_interpreter`, Anthropic versioned `code_execution_*` /
`web_search_*`, and the bare `code_execution` / `web_search`) are *not*
extracted — they stay in `tools[]` and pass through to the upstream provider,
which executes them server-side.

Web search has one opt-in exception: with `intercept=True` the provider-named
web-search keywords are claimed too, so a client that can only speak a
provider's vocabulary reaches a configured gateway backend. Code execution has
no such mode, and an OpenAI `function` named `web_search` is never claimed.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException, status

from gateway.api.routes._pipeline import ToolContext
from gateway.api.routes._tools import (
    _extract_code_execution_tool,
    _extract_web_search_tool,
    _retargeted_tool_choice,
    _strip_gateway_fields,
    _web_search_intercept_enabled,
    declares_native_web_search,
)
from gateway.core.config import GatewayConfig


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


# --- web_search interception (opt-in) ----------------------------------------


def test_intercept_claims_bare_web_search() -> None:
    entry, remaining = _extract_web_search_tool([{"type": "web_search"}], intercept=True)
    assert entry == {"type": "web_search"}
    assert remaining is None


def test_intercept_claims_anthropic_versioned_type() -> None:
    entry, remaining = _extract_web_search_tool([{"type": "web_search_20250305"}], intercept=True)
    assert entry == {"type": "web_search_20250305"}
    assert remaining is None


def test_intercept_claims_future_anthropic_version() -> None:
    entry, _ = _extract_web_search_tool([{"type": "web_search_20991231"}], intercept=True)
    assert entry is not None


def test_intercept_claims_openai_responses_preview_type() -> None:
    entry, _ = _extract_web_search_tool([{"type": "web_search_preview"}], intercept=True)
    assert entry is not None


def test_intercept_claims_claude_code_shape_with_name_and_max_uses() -> None:
    claude_code = {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
    entry, remaining = _extract_web_search_tool([claude_code], intercept=True)
    assert entry == claude_code
    assert remaining is None


def test_intercept_still_claims_the_canonical_otari_type() -> None:
    entry, _ = _extract_web_search_tool([{"type": "otari_web_search"}], intercept=True)
    assert entry == {"type": "otari_web_search"}


def test_intercept_never_claims_a_function_named_web_search() -> None:
    """A caller's own tool stays theirs to dispatch, even under interception.

    Claiming it would mean their handler never fires and they never get back a
    tool_call they can execute. LiteLLM excludes this case for the same reason.
    """
    own_tool = {"type": "function", "function": {"name": "web_search", "parameters": {}}}
    entry, remaining = _extract_web_search_tool([own_tool], intercept=True)
    assert entry is None
    assert remaining == [own_tool]


def test_intercept_does_not_claim_code_execution_or_unrelated_tools() -> None:
    tools: list[dict[str, Any]] = [
        {"type": "otari_code_execution"},
        {"type": "web_fetch_20250910"},
        {"type": "function", "function": {"name": "get_weather"}},
    ]
    entry, remaining = _extract_web_search_tool(tools, intercept=True)
    assert entry is None
    assert remaining == tools


def test_intercept_off_is_the_default_and_passes_provider_keywords_through() -> None:
    entry, remaining = _extract_web_search_tool([{"type": "web_search_20250305"}])
    assert entry is None
    assert remaining == [{"type": "web_search_20250305"}]


# --- native-declaration discrimination ---------------------------------------


def test_versioned_declaration_is_native() -> None:
    assert declares_native_web_search({"type": "web_search_20250305"}) is True


def test_bare_and_canonical_declarations_are_not_native() -> None:
    """Neither shape implies the caller expects native server-tool blocks back."""
    assert declares_native_web_search({"type": "web_search"}) is False
    assert declares_native_web_search({"type": "otari_web_search"}) is False


def test_missing_declaration_is_not_native() -> None:
    assert declares_native_web_search(None) is False
    assert declares_native_web_search({}) is False


# --- intercept toggle resolution ---------------------------------------------


def test_intercept_defaults_off(monkeypatch: Any) -> None:
    monkeypatch.delenv("OTARI_WEB_SEARCH_INTERCEPT", raising=False)
    assert _web_search_intercept_enabled(GatewayConfig()) is False


def test_intercept_reads_the_config_field(monkeypatch: Any) -> None:
    monkeypatch.delenv("OTARI_WEB_SEARCH_INTERCEPT", raising=False)
    assert _web_search_intercept_enabled(GatewayConfig(web_search_intercept=True)) is True


def test_intercept_falls_back_to_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_INTERCEPT", "true")
    assert _web_search_intercept_enabled(GatewayConfig()) is True


def test_intercept_env_falsey_values_stay_off(monkeypatch: Any) -> None:
    for raw in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("OTARI_WEB_SEARCH_INTERCEPT", raw)
        assert _web_search_intercept_enabled(GatewayConfig()) is False


def test_config_false_wins_over_a_truthy_env(monkeypatch: Any) -> None:
    """An explicit off (dashboard override / YAML) is not overridden by the env."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_INTERCEPT", "true")
    assert _web_search_intercept_enabled(GatewayConfig(web_search_intercept=False)) is False


# --- tool_choice retargeting -------------------------------------------------


def test_retargets_anthropic_tool_choice_to_the_canonical_name() -> None:
    assert _retargeted_tool_choice({"type": "tool", "name": "my_search"}, "my_search") == {
        "type": "tool",
        "name": "web_search",
    }


def test_retargets_chat_completions_tool_choice() -> None:
    choice = {"type": "function", "function": {"name": "my_search"}}
    assert _retargeted_tool_choice(choice, "my_search") == {
        "type": "function",
        "function": {"name": "web_search"},
    }


def test_retargets_responses_flat_function_tool_choice() -> None:
    assert _retargeted_tool_choice({"type": "function", "name": "my_search"}, "my_search") == {
        "type": "function",
        "name": "web_search",
    }


def test_retarget_is_a_noop_when_the_declared_name_is_already_canonical() -> None:
    choice = {"type": "tool", "name": "web_search"}
    assert _retargeted_tool_choice(choice, "web_search") is choice


def test_retarget_leaves_auto_and_any_untouched() -> None:
    for choice in ({"type": "auto"}, {"type": "any"}, {"type": "none"}):
        assert _retargeted_tool_choice(choice, "my_search") == choice


def test_retarget_leaves_a_choice_naming_a_different_tool_untouched() -> None:
    choice = {"type": "tool", "name": "get_weather"}
    assert _retargeted_tool_choice(choice, "my_search") is choice


def test_retarget_leaves_string_tool_choice_untouched() -> None:
    assert _retargeted_tool_choice("required", "my_search") == "required"


def test_retarget_does_not_mutate_the_callers_dict() -> None:
    choice = {"type": "function", "function": {"name": "my_search"}}
    _retargeted_tool_choice(choice, "my_search")
    assert choice == {"type": "function", "function": {"name": "my_search"}}


def test_strip_gateway_fields_retargets_tool_choice() -> None:
    fields = _strip_gateway_fields(
        {
            "tools": [{"type": "web_search_20250305", "name": "my_search"}],
            "tool_choice": {"type": "tool", "name": "my_search"},
        },
        tools_extracted=True,
        remaining_user_tools=None,
        web_search_declared_name="my_search",
    )
    assert fields["tool_choice"] == {"type": "tool", "name": "web_search"}
    assert "tools" not in fields


def test_strip_gateway_fields_leaves_tool_choice_alone_without_a_declared_name() -> None:
    fields = _strip_gateway_fields(
        {"tool_choice": {"type": "tool", "name": "my_search"}},
        tools_extracted=True,
        remaining_user_tools=None,
    )
    assert fields["tool_choice"] == {"type": "tool", "name": "my_search"}


def _capped_context(entry: dict[str, Any] | None) -> ToolContext:
    """A web-search tool context carrying ``entry``, with nothing else turned on."""
    return ToolContext(
        config=GatewayConfig(),
        mcp_server_configs=None,
        use_sandbox=False,
        sandbox_tool_entry=None,
        sandbox_url=None,
        sandbox_auth_token=None,
        use_web_search=True,
        web_search_tool_entry=entry,
        web_search_url="http://search.invalid",
        web_search_auth_token=None,
        remaining_user_tools=None,
        max_tool_iterations=10,
        tools_header=None,
    )


def test_max_uses_is_read_from_a_native_declaration() -> None:
    entry = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
    assert _capped_context(entry).max_web_search_uses == 3


def test_max_uses_is_honored_on_a_declaration_with_no_native_response_shape() -> None:
    """The cap bounds spend, so it does not depend on being able to describe the refusal.

    ``otari_web_search`` and the bare ``web_search`` short form get a plain tool
    error instead of a ``web_search_tool_result``, but the same number of searches.
    """
    for type_value in ("otari_web_search", "web_search"):
        entry = {"type": type_value, "max_uses": 2}
        ctx = _capped_context(entry)
        assert ctx.emit_native_web_search is False
        assert ctx.max_web_search_uses == 2, type_value


def test_no_max_uses_leaves_the_searches_uncapped() -> None:
    assert _capped_context({"type": "web_search_20250305"}).max_web_search_uses is None
    assert _capped_context(None).max_web_search_uses is None
    assert _capped_context({"type": "web_search_20250305", "max_uses": None}).max_web_search_uses is None


def test_a_zero_max_uses_caps_the_searches_at_none_rather_than_at_no_limit() -> None:
    """A spend control must not read a limit of zero as permission to spend freely."""
    entry = {"type": "web_search_20250305", "max_uses": 0}
    ctx = _capped_context(entry)
    assert ctx.max_web_search_uses == 0
    assert ctx.web_search_budget is not None
    assert ctx.web_search_budget.exhausted(), "the first search must already be over the cap"


def test_a_nonsensical_max_uses_is_rejected_instead_of_becoming_uncapped() -> None:
    """Malformed spend controls fail closed instead of allowing unlimited searches."""
    for value in (-1, True, False, "2", 1.5):
        entry = {"type": "web_search_20250305", "max_uses": value}
        with pytest.raises(HTTPException) as exc_info:
            _capped_context(entry)
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST, value
        assert exc_info.value.detail == "web_search max_uses must be a non-negative integer"
