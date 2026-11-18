"""Which session a request joins, whether it opens a turn, and which client tools it answers.

The request shapes here are structure only, shaped like what each harness sends:
roles, block types and ids, never real content.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from gateway.services.traces import (
    AnsweredCall,
    RequestClient,
    RequestTrace,
    SessionRef,
    TurnFacts,
    is_subagent,
    read_turn,
)
from gateway.services.traces import resolve_session as resolve
from gateway.services.traces._collector import call_id_digest, client_tool_span_id
from gateway.services.traces._identity import harness_of

_WORKSPACE = uuid.uuid4()
_CLAUDE_CODE = {"user-agent": "claude-cli/2.1.291 (external, sdk-cli)", "x-claude-code-session-id": "b57732d4"}
_OPENCODE_SESSION = "ses_e17323cf5ffesvwVHeHZmSK26l"
_OPENCODE = {
    "user-agent": "opencode/1.19.5 ai-sdk/provider-utils/4.0.23 runtime/bun/1.4.4",
    "x-opencode-session-id": _OPENCODE_SESSION,
    "x-session-id": _OPENCODE_SESSION,
    "x-session-affinity": _OPENCODE_SESSION,
}
_CODEX_THREAD = "0199e5a1-7c2b-7d4e-9f10-2a3b4c5d6e7f"
_CODEX_SUBAGENT_THREAD = "0199e5a1-8d3c-7e5f-a021-3b4c5d6e7f80"
_CODEX = {
    "originator": "codex_exec",
    "user-agent": "codex_exec/0.154.0 (Linux 6.8.0; x86_64) unknown (codex_exec; 0.154.0)",
    "session-id": _CODEX_THREAD,
    "thread-id": _CODEX_THREAD,
}
# Harnesses that send their session id under a generic name, each with the mark that
# makes the name count, and the header the id rides in.
_KILO_CODE = {"user-agent": "Kilo-Code/7.2.4", "x-title": "Kilo Code", "x-session-id": "ses_4f1a"}
_CRUSH = {"user-agent": "Charm-Crush/0.31.0 (https://charm.land/crush)", "x-session-id": "9c1e0f2ab47d3e55"}
_PI = {"user-agent": "pi/0.112.0 (darwin; node/24.11.0; arm64)", "session-id": "019a7c1e-3b2d-7f44"}
_ROO_CODE = {
    "originator": "roo-code",
    "user-agent": "roo-code/3.36.0 (darwin 25.1.0; arm64) node/22",
    "session_id": "t-1",
}
_MISTRAL_VIBE = {"user-agent": "Mistral-Vibe/2.14.0", "x-affinity": "a8f3c2d1-5e6b-4c7d"}
_OPENCLAW = {"user-agent": "openclaw/2026.11.3", "session_id": "7d2e9b10", "x-session-affinity": "7d2e9b10"}
_MARKED = {
    "kilo-code": (_KILO_CODE, "x-session-id"),
    "crush": (_CRUSH, "x-session-id"),
    "pi": (_PI, "session-id"),
    "roo-code": (_ROO_CODE, "session_id"),
    "mistral-vibe": (_MISTRAL_VIBE, "x-affinity"),
    "openclaw": (_OPENCLAW, "session_id"),
}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_claude_codes_own_session_header_names_the_session() -> None:
    assert resolve(_CLAUDE_CODE, session_label=None, tags=None) == SessionRef(source="harness", key="b57732d4")


def test_opencodes_own_session_header_names_the_session() -> None:
    assert resolve(_OPENCODE, session_label=None, tags=None) == SessionRef(source="harness", key=_OPENCODE_SESSION)


def test_the_generic_session_headers_opencode_also_sends_name_no_session() -> None:
    headers = {key: value for key, value in _OPENCODE.items() if key != "x-opencode-session-id"}

    assert resolve(headers, session_label=None, tags=None) is None


def test_codexs_own_session_header_names_the_session() -> None:
    assert resolve(_CODEX, session_label=None, tags=None) == SessionRef(source="harness", key=_CODEX_THREAD)


def test_a_codex_subagent_is_a_session_of_its_own_thread() -> None:
    subagent = _CODEX | {"thread-id": _CODEX_SUBAGENT_THREAD}

    assert resolve(subagent, session_label=None, tags=None) == SessionRef(source="harness", key=_CODEX_SUBAGENT_THREAD)


def test_a_codex_request_without_a_thread_id_is_keyed_on_its_session_id() -> None:
    headers = {key: value for key, value in _CODEX.items() if key != "thread-id"}

    assert resolve(headers, session_label=None, tags=None) == SessionRef(source="harness", key=_CODEX_THREAD)


@pytest.mark.parametrize("marker", ["originator", "user-agent"])
def test_either_header_codex_marks_its_requests_with_is_enough(marker: str) -> None:
    headers = {marker: _CODEX[marker], "session-id": _CODEX_THREAD}

    assert resolve(headers, session_label=None, tags=None) == SessionRef(source="harness", key=_CODEX_THREAD)
    assert harness_of(headers) == "codex"


def test_a_session_id_header_from_another_client_names_no_session() -> None:
    headers = {"user-agent": "python-httpx/0.28", "session-id": _CODEX_THREAD, "thread-id": _CODEX_THREAD}

    assert resolve(headers, session_label=None, tags=None) is None
    assert harness_of(headers) == "python-httpx"


@pytest.mark.parametrize(("header", "harness"), [("x-amp-thread-id", "amp"), ("agent-session-id", "goose")])
def test_a_harnesss_own_session_header_names_the_session_from_any_client(header: str, harness: str) -> None:
    headers = {header: "T-7f3e"}

    assert resolve(headers, session_label=None, tags=None) == SessionRef(source="harness", key="T-7f3e")
    assert harness_of(headers) == harness


@pytest.mark.parametrize("harness", list(_MARKED))
def test_a_generic_session_header_names_the_session_beside_its_harnesss_mark(harness: str) -> None:
    headers, header = _MARKED[harness]

    assert resolve(headers, session_label=None, tags=None) == SessionRef(source="harness", key=headers[header])
    assert harness_of(headers) == harness


@pytest.mark.parametrize("harness", list(_MARKED))
def test_a_generic_session_header_without_its_harnesss_mark_names_no_session(harness: str) -> None:
    headers, header = _MARKED[harness]
    unmarked = {"user-agent": "python-httpx/0.28", header: headers[header]}

    assert resolve(unmarked, session_label=None, tags=None) is None


@pytest.mark.parametrize(
    "headers",
    [
        {"user-agent": "kilo/stable/7.2.4/cli", "x-session-id": "ses_4f1a"},
        {"user-agent": "pi/0.112.0", "session_id": "ses_4f1a"},
        {"user-agent": "pi/0.112.0", "x-session-affinity": "ses_4f1a"},
        {"user-agent": "roo-code/3.36.0", "session_id": "ses_4f1a"},
        {"user-agent": "mistral-client-python/Mistral-Vibe/2.14.0", "x-affinity": "ses_4f1a"},
        {"user-agent": "openclaw/2026.11.3", "x-session-affinity": "ses_4f1a"},
    ],
)
def test_each_header_and_mark_a_harness_sends_its_session_with_is_enough(headers: dict[str, str]) -> None:
    assert resolve(headers, session_label=None, tags=None) == SessionRef(source="harness", key="ses_4f1a")


def test_a_kilo_code_subagent_is_a_session_of_its_own() -> None:
    subagent = _KILO_CODE | {"x-session-id": "ses_77b0", "x-parent-session-id": _KILO_CODE["x-session-id"]}

    assert resolve(subagent, session_label=None, tags=None) == SessionRef(source="harness", key="ses_77b0")
    assert not is_subagent(subagent)


def test_litellms_session_header_names_the_session_and_leaves_the_harness_to_the_user_agent() -> None:
    headers = {"user-agent": "litellm/1.80.0", "x-litellm-session-id": "conv-81d2"}

    assert resolve(headers, session_label=None, tags=None) == SessionRef(source="harness", key="conv-81d2")
    assert harness_of(headers) == "litellm"


def test_oh_my_pi_is_named_by_its_mark_while_claude_codes_header_names_its_session() -> None:
    headers = {"user-agent": "omp/17.2.12", "x-claude-code-session-id": "019a7c1e-3b2d"}

    assert resolve(headers, session_label=None, tags=None) == SessionRef(source="harness", key="019a7c1e-3b2d")
    assert harness_of(headers) == "oh-my-pi"


def test_a_harnesss_own_session_header_wins_over_a_generic_one() -> None:
    headers = _CRUSH | {"x-amp-thread-id": "T-7f3e", "x-litellm-session-id": "conv-81d2"}

    assert resolve(headers, session_label=None, tags=None) == SessionRef(source="harness", key="T-7f3e")


@pytest.mark.parametrize(
    ("headers", "label", "tags"),
    [
        ({"Otari-Conversation-Id": "mine"}, None, None),
        ({}, "mine", None),
        ({}, None, {"session_id": "mine"}),
    ],
)
def test_a_session_the_client_names_on_purpose(
    headers: dict[str, str], label: str | None, tags: dict[str, str] | None
) -> None:
    assert resolve(headers, session_label=label, tags=tags) == SessionRef(source="client", key="mine")


def test_the_clients_own_session_wins_over_its_harness() -> None:
    ref = resolve(_CLAUDE_CODE | {"Otari-Conversation-Id": "mine"}, session_label=None, tags=None)

    assert ref == SessionRef(source="client", key="mine")


def test_a_request_that_names_no_session_has_none() -> None:
    assert resolve({"user-agent": "python-httpx/0.28"}, session_label="  ", tags={"purpose": "x"}) is None


def test_the_same_label_from_two_users_is_two_sessions() -> None:
    ref = SessionRef(source="client", key="chat-1")

    assert ref.trace_id(_WORKSPACE, "alice") != ref.trace_id(_WORKSPACE, "bob")
    assert ref.trace_id(_WORKSPACE, "alice") != ref.trace_id(uuid.uuid4(), "alice")
    assert ref.trace_id(_WORKSPACE, "alice") == ref.trace_id(_WORKSPACE, "alice")


@pytest.mark.parametrize(
    ("ref", "name"),
    [
        (SessionRef(source="client", key="hackbot-1234"), "hackbot-1234"),
        (SessionRef(source="client", key="the run Alice started"), None),
        (SessionRef(source="client", key="alice@example.com"), None),
        (SessionRef(source="client", key="api-key-123456"), None),
        (SessionRef(source="client", key="API_KEY_0123456789abcdef0123456789abcdef"), None),
        (SessionRef(source="client", key="ASIAIOSFODNN7EXAMPLE"), None),
        (SessionRef(source="client", key="alice@example.com:42"), None),
        (SessionRef(source="client", key="src/secret/notes.txt:42"), None),
        (SessionRef(source="harness", key="b57732d4"), None),
        (SessionRef(source="otlp", key="chat-1"), None),
    ],
)
def test_only_an_identifier_shaped_key_the_client_chose_names_the_session(ref: SessionRef, name: str | None) -> None:
    assert ref.name == name


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (_CLAUDE_CODE, "claude-code"),
        ({"user-agent": "claude-cli/2.1.291"}, "claude-code"),
        (_CODEX, "codex"),
        (_OPENCODE, "opencode"),
        ({"user-agent": "my-proxy/2.0", "x-opencode-session-id": _OPENCODE_SESSION}, "opencode"),
        ({"user-agent": "Charm-Crush/0.31.0 (https://charm.land/crush)"}, "crush"),
        ({"user-agent": "kilo/stable/7.2.4/cli"}, "kilo-code"),
        ({"user-agent": "opencode/0.15 ai-sdk"}, "opencode"),
        ({"user-agent": "omp/17.2.12"}, "oh-my-pi"),
        ({"user-agent": "mistral-client-python/Mistral-Vibe/2.14.0"}, "mistral-vibe"),
        ({"user-agent": "HermesAgent/1.0"}, "hermes-agent"),
        ({"user-agent": "kimi-code-cli/1.4.0"}, "kimi-code"),
        ({"user-agent": "QwenCode/0.9.1 (darwin; arm64)"}, "qwen-code"),
        ({"user-agent": "GitHubCopilotChat/0.40.0"}, "github-copilot"),
        ({"user-agent": "Zed/0.215.3 (macos; aarch64)"}, "zed"),
        ({"user-agent": "curl/8.7.1"}, "curl"),
        ({"user-agent": "OpenAI/Python 1.99"}, "openai"),
        ({"user-agent": "AsyncOpenAI/Python 2.30.0"}, "openai"),
        ({"user-agent": "AsyncAzureOpenAI/Python 2.30.0"}, "openai"),
        ({"user-agent": "AsyncAnthropicBedrock/Python 1.17.0"}, "anthropic"),
        ({"user-agent": "Anthropic/JS 0.71.0"}, "anthropic"),
        ({"user-agent": "AsyncGroq/Python 1.1.2"}, "groq"),
        ({"user-agent": "AsyncAnthropicVertex/Python 1.17.0"}, "anthropic"),
        ({"user-agent": "AcmeOpenAI/Python 1.2.3"}, "acmeopenai"),
        ({"user-agent": "speakeasy-sdk/python 3.1.0 2.935.1 1.0.0 mistralai"}, "mistralai"),
        ({"user-agent": "speakeasy-sdk/python 3.1.0 2.935.1 1.0.0 mistralai (integration)"}, "mistralai"),
        ({"user-agent": "google-genai-sdk/2.25.0 gl-python/3.13.1"}, "google-genai"),
        ({"user-agent": "OpenAPI-Generator/1.0.0/python"}, "openapi-generator"),
        ({"user-agent": "python-httpx/0.28.1"}, "python-httpx"),
        ({}, None),
    ],
)
def test_the_harness_comes_from_the_user_agents_product_name(headers: dict[str, str], expected: str | None) -> None:
    assert harness_of(headers) == expected


@pytest.mark.parametrize(
    ("headers", "language"),
    [
        ({"user-agent": "AsyncOpenAI/Python 2.30.0", "x-stainless-lang": "python"}, "python"),
        ({"user-agent": "OpenAI/JS 6.1.0", "x-stainless-lang": "js"}, "javascript"),
        ({"user-agent": "my-app/1.0", "x-stainless-lang": "go"}, "go"),
        ({"user-agent": "cohere/7.2.0", "x-fern-language": "Python"}, "python"),
        ({"user-agent": "OpenAPI-Generator/1.0.0/python"}, "python"),
        ({"user-agent": "OpenAPI-Generator/1.0.0/go"}, "go"),
        ({"user-agent": "speakeasy-sdk/python 3.1.0 2.935.1 1.0.0 mistralai"}, "python"),
        ({"user-agent": "google-genai-sdk/2.25.0 gl-python/3.13.1"}, "python"),
        ({"user-agent": "ollama-python/0.6.1 (arm64 darwin) Python/3.13.1"}, "python"),
        ({"user-agent": "python-httpx/0.28.1"}, "python"),
        ({"user-agent": "python-requests/2.32.3"}, "python"),
        ({"user-agent": "Python/3.12 aiohttp/3.11.0"}, "python"),
        ({"user-agent": "node"}, "javascript"),
        ({"user-agent": "axios/1.7.9"}, "javascript"),
        ({"user-agent": "Bun/1.2.0"}, "javascript"),
        ({"user-agent": "Go-http-client/2.0"}, "go"),
        ({"user-agent": "okhttp/4.12.0"}, "java"),
        ({"user-agent": "my-app/okhttp 2026"}, "java"),
        ({"user-agent": "Java-http-client/21.0.2"}, "java"),
        ({"user-agent": "Ruby"}, "ruby"),
        (_PI, "javascript"),
        ({"user-agent": "curl/8.7.1"}, None),
        ({"user-agent": "claude-cli/2.1.291 (external, sdk-cli)"}, None),
        ({"user-agent": "my-agent/1.4.0", "x-stainless-lang": "cobol"}, None),
        ({}, None),
    ],
)
def test_the_clients_language_comes_from_its_sdk_or_its_http_library(
    headers: dict[str, str], language: str | None
) -> None:
    assert RequestClient.of(headers).language == language


@pytest.mark.parametrize(
    ("headers", "version"),
    [
        ({"user-agent": "AsyncOpenAI/Python 2.30.0"}, "2.30.0"),
        ({"user-agent": "OpenAPI-Generator/1.0.0/python"}, "1.0.0"),
        (_CLAUDE_CODE | {"x-stainless-package-version": "0.71.0"}, "2.1.291"),
        (_CODEX, "0.154.0"),
        ({"user-agent": "mistral-client-python/Mistral-Vibe/2.14.0"}, "2.14.0"),
        ({"user-agent": "speakeasy-sdk/python 3.1.0 2.935.1 1.0.0 mistralai"}, "3.1.0"),
        ({"user-agent": "python-httpx/0.28.1"}, "0.28.1"),
        ({"user-agent": "my-agent/1.4.0-beta.2"}, "1.4.0-beta.2"),
        ({"user-agent": "AsyncOpenAI/Python 2.30.0rc1"}, "2.30.0rc1"),
        ({"user-agent": "my-agent/v2.0.0.dev3"}, "2.0.0.dev3"),
        ({"user-agent": "my-agent/2.30.0-rc.1+build.7"}, "2.30.0-rc.1+build.7"),
        # An SDK's package version is the SDK's, not the client built on it.
        ({"user-agent": "my-app", "x-stainless-package-version": "4.2.0"}, None),
        # A proxy's version is not the version of the harness its session header names.
        ({"user-agent": "my-proxy/2.0", "x-opencode-session-id": _OPENCODE_SESSION}, None),
        ({"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/131.0.0.0 Safari/537.36"}, None),
        ({"user-agent": "Go-http-client/1.1"}, None),
        ({"user-agent": "Python/3.12 aiohttp/3.11.0"}, None),
        ({"user-agent": "node"}, None),
        ({"user-agent": "my-agent/latest"}, None),
        # A second product is not the first one's version.
        ({"user-agent": "my-app 2026"}, None),
        ({"user-agent": "my-app/okhttp 2026"}, None),
        ({"user-agent": "speakeasy-sdk/python 3.1.0 2.935.1 1.0.0 mistralai (integration)"}, "3.1.0"),
        ({}, None),
    ],
)
def test_the_clients_version_comes_from_its_user_agent(headers: dict[str, str], version: str | None) -> None:
    assert RequestClient.of(headers).version == version


def test_a_claude_code_subagent_is_marked_by_its_agent_header() -> None:
    assert is_subagent(_CLAUDE_CODE | {"x-claude-code-agent-id": "a1b2c3"})
    assert not is_subagent(_CLAUDE_CODE)
    assert not is_subagent(_CLAUDE_CODE | {"x-claude-code-agent-id": " "})


# ---------------------------------------------------------------------------
# Turns, per dialect
# ---------------------------------------------------------------------------


def test_messages_a_new_prompt_opens_a_turn() -> None:
    assert read_turn("messages", [{"role": "user", "content": [{"type": "text", "text": "x"}]}]).opens_turn


def test_messages_tool_results_answer_the_calls_the_history_named() -> None:
    history = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}},
                {"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "x"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "x", "is_error": True},
            ],
        },
    ]

    assert read_turn("messages", history) == TurnFacts(
        opens_turn=False,
        answered=(
            AnsweredCall(call_id="toolu_1", name="Bash", is_error=False, position=2),
            AnsweredCall(call_id="toolu_2", name="Read", is_error=True, position=2),
        ),
    )


_OPERATOR_NOTE = {"role": "system", "content": [{"type": "text", "text": "x"}]}
_EFFORT_CHANGE = {"role": "system", "content": [], "output_config": {"effort": "low"}}


@pytest.mark.parametrize(
    ("prompt", "trailing"),
    [
        ([{"type": "text", "text": "x"}] * 3, [_OPERATOR_NOTE]),
        ("x", [_OPERATOR_NOTE]),
        ("x", [_OPERATOR_NOTE, _EFFORT_CHANGE]),
    ],
)
def test_messages_a_prompt_under_trailing_system_messages_opens_a_turn(
    prompt: Any, trailing: list[dict[str, Any]]
) -> None:
    """Claude Code follows the user's input with a mid-conversation system message, its own context."""
    assert read_turn("messages", [{"role": "user", "content": prompt}, *trailing]).opens_turn


def test_messages_tool_results_under_a_trailing_system_message_answer_their_calls() -> None:
    history = [
        {"role": "user", "content": [{"type": "text", "text": "x"}]},
        _OPERATOR_NOTE,
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "x"}]},
        _OPERATOR_NOTE,
    ]

    assert read_turn("messages", history) == TurnFacts(
        opens_turn=False, answered=(AnsweredCall(call_id="toolu_1", name="Bash", is_error=False, position=3),)
    )


def test_messages_a_system_message_after_a_server_tool_use_opens_no_turn() -> None:
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": [{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search"}]},
        _OPERATOR_NOTE,
    ]

    assert read_turn("messages", history) == TurnFacts(opens_turn=False)


def test_chat_trailing_tool_messages_answer_the_assistants_calls() -> None:
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "x"},
    ]

    assert read_turn("chat", history) == TurnFacts(
        opens_turn=False, answered=(AnsweredCall(call_id="call_1", name="bash", is_error=False, position=2),)
    )
    assert read_turn("chat", [{"role": "user", "content": "x"}]).opens_turn


def test_responses_function_call_outputs_answer_their_calls() -> None:
    items = [
        {"type": "message", "role": "user", "content": "x"},
        {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "x"},
    ]

    assert read_turn("responses", items).answered == (
        AnsweredCall(call_id="call_1", name="shell", is_error=False, position=2),
    )
    assert read_turn("responses", "a prompt").opens_turn


@pytest.mark.parametrize("dialect", ["chat", "messages", "responses"])
def test_an_empty_or_unexpected_input_says_nothing(dialect: str) -> None:
    assert read_turn(dialect, None) == TurnFacts(opens_turn=False)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------


def _trace(*, session: SessionRef | None, turn: TurnFacts | None, subagent: bool = False) -> RequestTrace:
    return RequestTrace(
        request_id="req-2",
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id=None,
        endpoint="/v1/messages",
        started_at=datetime(2026, 10, 7, 12, tzinfo=UTC),
        max_spans=256,
        session=session,
        harness="claude-code",
        turn=turn,
        subagent=subagent,
    )


def test_a_request_in_a_session_writes_into_the_sessions_trace() -> None:
    session = SessionRef(source="harness", key="b57732d4")

    write = _trace(session=session, turn=TurnFacts(opens_turn=True)).finish(status_code=200, completed=True)

    assert write.trace_id == session.trace_id(_WORKSPACE, "alice")
    assert (write.session_source, write.harness) == ("harness", "claude-code")
    assert write.spans[0].opens_turn is True


def test_a_session_the_client_named_is_written_under_that_name() -> None:
    named = _trace(session=SessionRef(source="client", key="hackbot-1234"), turn=None).finish(
        status_code=200, completed=True
    )
    harness = _trace(session=SessionRef(source="harness", key="b57732d4"), turn=None).finish(
        status_code=200, completed=True
    )

    assert (named.session_source, named.name, named.harness) == ("client", "hackbot-1234", "claude-code")
    assert harness.name is None


def test_a_subagents_request_opens_no_turn_but_records_the_tools_it_answers() -> None:
    answered = (AnsweredCall(call_id="toolu_1", name="Bash", is_error=False),)

    prompt = _trace(session=None, turn=TurnFacts(opens_turn=True), subagent=True).finish(
        status_code=200, completed=True
    )
    results = _trace(session=None, turn=TurnFacts(opens_turn=False, answered=answered), subagent=True).finish(
        status_code=200, completed=True
    )

    assert prompt.spans[0].opens_turn is False
    assert [span.tool_name for span in results.spans if span.kind == "tool"] == ["Bash"]


def test_answered_calls_become_client_tool_spans_ending_when_this_request_arrived() -> None:
    turn = TurnFacts(
        opens_turn=False,
        answered=(
            AnsweredCall(call_id="toolu_1", name="Bash", is_error=False),
            AnsweredCall(call_id="toolu_2", name="rm -rf /", is_error=True),
        ),
    )

    write = _trace(session=None, turn=turn).finish(status_code=200, completed=True)

    tools = [span for span in write.spans if span.kind == "tool"]
    assert [(span.span_id, span.tool_name, span.outcome, span.tool_type) for span in tools] == [
        (client_tool_span_id("toolu_1", 0), "Bash", "ok", "client"),
        (client_tool_span_id("toolu_2", 0), None, "ok", "client"),
    ]
    # An error result the agent read is marked, not counted as the session's error.
    assert [span.attributes.get("otari.tool.succeeded") for span in tools] == [None, False]
    assert all(span.start_time is None and span.end_time is not None for span in tools)


def test_a_call_id_is_kept_on_its_tool_span_as_a_digest() -> None:
    call_id = "toolu_01Vd8AQi5j3wv6NtfNJFHkdM"
    turn = TurnFacts(opens_turn=False, answered=(AnsweredCall(call_id=call_id, name="Read", is_error=False),))

    write = _trace(session=None, turn=turn).finish(status_code=200, completed=True)

    (tool,) = [span for span in write.spans if span.kind == "tool"]
    assert tool.tool_call_id == call_id_digest(_WORKSPACE, call_id)


def test_a_client_tool_span_id_never_carries_the_call_id() -> None:
    turn = TurnFacts(
        opens_turn=False, answered=(AnsweredCall(call_id="my private\u0000note", name=None, is_error=False),)
    )

    write = _trace(session=None, turn=turn).finish(status_code=200, completed=True)

    (tool,) = [span for span in write.spans if span.kind == "tool"]
    assert tool.span_id == client_tool_span_id("my private\u0000note", 0)
    assert "private" not in tool.span_id and "private" not in (tool.tool_call_id or "")


def test_a_call_id_reused_later_in_the_session_is_its_own_span_and_a_retry_is_not() -> None:
    """Some clients number calls per turn (``call_1`` every time); a retry repeats the request exactly."""

    def history(turns: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [{"role": "user", "content": "x"}]
        for _ in range(turns):
            items.append({"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "bash"}}]})
            items.append({"role": "tool", "tool_call_id": "call_1", "content": "x"})
        return items

    def tool_span_id(items: list[dict[str, Any]]) -> str:
        write = _trace(session=None, turn=read_turn("chat", items)).finish(status_code=200, completed=True)
        (tool,) = [span for span in write.spans if span.kind == "tool"]
        return tool.span_id

    assert tool_span_id(history(1)) == tool_span_id(history(1))
    assert tool_span_id(history(1)) != tool_span_id(history(2))


def test_client_tool_spans_count_against_the_span_budget() -> None:
    answered = tuple(AnsweredCall(call_id=f"c{i}", name="Bash", is_error=False) for i in range(20))
    trace = _trace(session=None, turn=TurnFacts(opens_turn=False, answered=answered))
    trace.max_spans = 2

    write = trace.finish(status_code=200, completed=True)

    assert len([span for span in write.spans if span.kind == "tool"]) == 2
    assert trace.dropped == 18
