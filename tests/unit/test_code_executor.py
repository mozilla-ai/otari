"""Who runs a provider-native code-execution declaration: the executor decision.

Pure logic in ``gateway.api.routes._tools`` plus the settings that feed it. The
request-path wiring (claiming the keyword, the policy pin, the header) is
covered by ``tests/integration/test_code_execution_executor.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from any_llm import LLMProvider

from gateway.api.routes._normalize import sandbox_requested
from gateway.api.routes._tools import (
    CODE_EXECUTION_HEADER,
    _extract_code_execution_tool,
    code_execution_declaration_forms,
    decide_code_executor,
    first_provider_code_execution_tool,
    native_code_execution_dialect,
    parse_code_execution_header,
    provider_runs_code_natively,
    resolve_code_executor_preference,
)
from gateway.core.config import GatewayConfig
from gateway.services.tool_settings_service import get_field_options, validate_value
from gateway.types.code_execution import CodeExecutor

ANTHROPIC_DATED = {"type": "code_execution_20250825", "name": "code_execution"}
OPENAI_INTERPRETER = {"type": "code_interpreter", "container": {"type": "auto"}}
BARE = {"type": "code_execution"}


# --- the vocabulary -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("auto", CodeExecutor.AUTO),
        ("OTARI", CodeExecutor.OTARI),
        (" provider ", CodeExecutor.PROVIDER),
        (CodeExecutor.OTARI, CodeExecutor.OTARI),
        ("anthropic", None),
        ("", None),
        (None, None),
        (3, None),
    ],
)
def test_parse_accepts_the_three_values_in_any_case_and_nothing_else(
    raw: object, expected: CodeExecutor | None
) -> None:
    assert CodeExecutor.parse(raw) is expected


def test_header_absent_or_blank_means_no_request_preference() -> None:
    assert parse_code_execution_header(None) is None
    assert parse_code_execution_header("   ") is None


def test_header_value_is_parsed_case_insensitively() -> None:
    assert parse_code_execution_header("Otari") is CodeExecutor.OTARI


def test_header_outside_the_vocabulary_is_a_caller_error() -> None:
    with pytest.raises(ValueError, match=CODE_EXECUTION_HEADER):
        parse_code_execution_header("anthropic")


# --- composing the three layers ---------------------------------------------------


def test_deployment_default_applies_when_nobody_closer_said_otherwise() -> None:
    assert resolve_code_executor_preference(requested=None, workspace=None, deployment=CodeExecutor.AUTO) == (
        CodeExecutor.AUTO,
        False,
    )


def test_request_header_wins_over_the_deployment_default() -> None:
    preference, conflict = resolve_code_executor_preference(
        requested=CodeExecutor.OTARI, workspace=None, deployment=CodeExecutor.PROVIDER
    )
    assert preference is CodeExecutor.OTARI
    assert conflict is False


def test_workspace_pin_wins_over_both_and_a_disagreeing_header_is_a_conflict() -> None:
    preference, conflict = resolve_code_executor_preference(
        requested=CodeExecutor.PROVIDER, workspace=CodeExecutor.OTARI, deployment=CodeExecutor.AUTO
    )
    assert preference is CodeExecutor.OTARI
    assert conflict is True


def test_a_header_that_agrees_with_the_pin_is_not_a_conflict() -> None:
    preference, conflict = resolve_code_executor_preference(
        requested=CodeExecutor.OTARI, workspace=CodeExecutor.OTARI, deployment=CodeExecutor.PROVIDER
    )
    assert preference is CodeExecutor.OTARI
    assert conflict is False


# --- turning a preference into a decision -----------------------------------------


@pytest.mark.parametrize("native_available", [True, False])
@pytest.mark.parametrize("sandbox_configured", [True, False])
def test_an_explicit_preference_is_returned_as_asked(native_available: bool, sandbox_configured: bool) -> None:
    for explicit in (CodeExecutor.OTARI, CodeExecutor.PROVIDER):
        assert (
            decide_code_executor(explicit, sandbox_configured=sandbox_configured, native_available=native_available)
            is explicit
        )


def test_auto_prefers_a_provider_that_runs_the_tool_natively() -> None:
    assert (
        decide_code_executor(CodeExecutor.AUTO, sandbox_configured=True, native_available=True) is CodeExecutor.PROVIDER
    )


def test_auto_brings_the_code_here_when_the_provider_cannot_run_it() -> None:
    assert (
        decide_code_executor(CodeExecutor.AUTO, sandbox_configured=True, native_available=False) is CodeExecutor.OTARI
    )


def test_auto_leaves_the_provider_in_charge_when_there_is_no_sandbox() -> None:
    """Nothing to bring the code to, so the declaration is forwarded as it always was."""
    assert (
        decide_code_executor(CodeExecutor.AUTO, sandbox_configured=False, native_available=False)
        is CodeExecutor.PROVIDER
    )


# --- which declarations a provider runs natively -----------------------------------


def test_anthropics_dated_keyword_is_native_on_messages_against_anthropic() -> None:
    assert provider_runs_code_natively(ANTHROPIC_DATED, provider="anthropic", dialect="messages") is True


def test_openais_interpreter_is_native_on_responses_against_openai() -> None:
    assert provider_runs_code_natively(OPENAI_INTERPRETER, provider="openai", dialect="responses") is True


@pytest.mark.parametrize(
    ("entry", "provider", "dialect"),
    [
        (ANTHROPIC_DATED, "mistral", "messages"),  # the model swap the executor exists for
        (ANTHROPIC_DATED, "anthropic", "chat"),  # no native form on Chat Completions
        (ANTHROPIC_DATED, "anthropic", "responses"),  # Anthropic's words in OpenAI's format
        (OPENAI_INTERPRETER, "openai", "chat"),
        (OPENAI_INTERPRETER, "anthropic", "responses"),
        (BARE, "anthropic", "messages"),  # the bare short form is nobody's
        (BARE, "openai", "responses"),
        (ANTHROPIC_DATED, None, "messages"),  # provider unknown
        (None, "anthropic", "messages"),
    ],
)
def test_everything_else_is_not_natively_served(
    entry: dict[str, str] | None, provider: str | None, dialect: str
) -> None:
    assert provider_runs_code_natively(entry, provider=provider, dialect=dialect) is False


def test_provider_name_is_matched_case_insensitively() -> None:
    assert provider_runs_code_natively(ANTHROPIC_DATED, provider="Anthropic", dialect="messages") is True


# --- which native result shape a caller expects back -----------------------------------


def test_dated_anthropic_keyword_expects_messages_blocks() -> None:
    assert native_code_execution_dialect(ANTHROPIC_DATED) == "messages"


def test_openai_interpreter_expects_a_responses_item() -> None:
    assert native_code_execution_dialect(OPENAI_INTERPRETER) == "responses"


@pytest.mark.parametrize("entry", [BARE, {"type": "otari_code_execution"}, None, {}])
def test_other_declarations_expect_the_plain_result(entry: dict[str, str] | None) -> None:
    assert native_code_execution_dialect(entry) is None


# --- finding and claiming the keyword --------------------------------------------------


def test_the_first_provider_keyword_is_found_without_being_removed() -> None:
    tools: list[dict[str, Any]] = [{"type": "function", "function": {"name": "f"}}, OPENAI_INTERPRETER, BARE]
    assert first_provider_code_execution_tool(tools) is OPENAI_INTERPRETER
    assert len(tools) == 3


def test_a_function_named_code_execution_is_the_callers_own() -> None:
    assert first_provider_code_execution_tool([{"type": "function", "function": {"name": "code_execution"}}]) is None


def test_intercept_claims_the_provider_keyword_and_leaves_the_rest() -> None:
    user_tool = {"type": "function", "function": {"name": "get_weather"}}
    entry, remaining = _extract_code_execution_tool([user_tool, ANTHROPIC_DATED], intercept=True)
    assert entry == ANTHROPIC_DATED
    assert remaining == [user_tool]


def test_intercept_off_still_leaves_the_provider_keyword_alone() -> None:
    entry, remaining = _extract_code_execution_tool([ANTHROPIC_DATED])
    assert entry is None
    assert remaining == [ANTHROPIC_DATED]


# --- the deployment setting ----------------------------------------------------------


def test_unset_means_auto() -> None:
    assert GatewayConfig().effective_code_executor() is CodeExecutor.AUTO


def test_the_config_field_is_normalized_and_validated() -> None:
    assert GatewayConfig(code_execution_executor="Provider").effective_code_executor() is CodeExecutor.PROVIDER
    with pytest.raises(ValueError, match="code_execution_executor"):
        GatewayConfig(code_execution_executor="anthropic")


def test_the_env_var_fills_in_when_the_override_is_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_CODE_EXECUTION_EXECUTOR", "otari")
    config = GatewayConfig()
    config.code_execution_executor = None
    assert config.effective_code_executor() is CodeExecutor.OTARI


def test_the_dashboard_setting_is_a_closed_vocabulary() -> None:
    assert get_field_options("code_execution_executor") == ["auto", "otari", "provider"]
    assert get_field_options("sandbox_url") is None
    assert validate_value("code_execution_executor", "OTARI") == "otari"
    assert validate_value("code_execution_executor", "") is None
    with pytest.raises(ValueError, match="must be one of"):
        validate_value("code_execution_executor", "anthropic")


def test_declaration_forms_include_the_provider_keywords_unless_the_provider_owns_them() -> None:
    assert code_execution_declaration_forms(GatewayConfig()) == [
        "otari_code_execution",
        "code_execution",
        "code_interpreter",
        "code_execution_<date>",
    ]
    assert code_execution_declaration_forms(GatewayConfig(code_execution_executor="provider")) == [
        "otari_code_execution"
    ]


# --- staging attachments follows the executor ---------------------------------------------


def _requested(
    tools: list[dict[str, Any]],
    *,
    provider: str | None,
    dialect: str = "messages",
    header: str | None = None,
    pin: CodeExecutor | None = None,
    **config: Any,
) -> bool:
    return sandbox_requested(
        tools,
        config=GatewayConfig(sandbox_url="http://sandbox:8080", **config),
        provider=LLMProvider(provider) if provider else None,
        dialect=dialect,
        code_execution_header=header,
        workspace_executor=pin,
    )


def test_the_explicit_type_always_stages() -> None:
    assert _requested([{"type": "otari_code_execution"}], provider="anthropic") is True


def test_a_natively_served_declaration_does_not_stage() -> None:
    assert _requested([ANTHROPIC_DATED], provider="anthropic") is False


def test_a_declaration_the_provider_cannot_run_stages() -> None:
    assert _requested([ANTHROPIC_DATED], provider="mistral") is True
    assert _requested([BARE], provider="anthropic") is True


def test_the_deployment_default_decides_staging_too() -> None:
    assert _requested([ANTHROPIC_DATED], provider="anthropic", code_execution_executor="otari") is True
    assert _requested([BARE], provider="mistral", code_execution_executor="provider") is False


def test_a_workspace_pin_decides_staging_over_the_header_and_the_default() -> None:
    # The same three layers admission composes: a pin pulling a natively served
    # declaration here stages, and one pushing it to the provider does not.
    assert _requested([ANTHROPIC_DATED], provider="anthropic", pin=CodeExecutor.OTARI) is True
    assert _requested([ANTHROPIC_DATED], provider="mistral", pin=CodeExecutor.PROVIDER) is False
    assert _requested([BARE], provider="mistral", header="otari", pin=CodeExecutor.PROVIDER) is False


def test_no_sandbox_means_nothing_is_staged() -> None:
    assert (
        sandbox_requested(
            [BARE],
            config=GatewayConfig(),
            provider=LLMProvider("mistral"),
            dialect="messages",
            code_execution_header=None,
        )
        is False
    )
