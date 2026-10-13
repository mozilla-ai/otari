"""Which built-in tools Otari runs on each candidate a request may be sent to."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.models.tools import CodeExecutor
from gateway.services.tools import (
    BUILTIN_TOOLS,
    CODE_EXECUTION_TOOL,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    BuiltinTool,
    CandidateTools,
    Dialect,
    ProviderToolDeclaration,
    decide_candidate_tools,
)

ANTHROPIC_CODE = {"type": "code_execution_20250825", "name": "code_execution"}
ANTHROPIC_SEARCH = {"type": "web_search_20250305", "name": "web_search"}
ANTHROPIC_FETCH = {"type": "web_fetch_20250910", "name": "web_fetch"}
CODE = CODE_EXECUTION_TOOL.name


def _tool(name: str) -> BuiltinTool:
    return next(tool for tool in BUILTIN_TOOLS if tool.name == name)


def _declared(
    tool: BuiltinTool,
    entry: dict[str, Any],
    preference: CodeExecutor | None,
    *,
    backend: bool = True,
    uniform: bool = False,
    follow_head: bool = False,
) -> ProviderToolDeclaration:
    return ProviderToolDeclaration(
        tool=tool,
        entry=entry,
        preference=preference,
        backend_configured=backend,
        uniform=uniform,
        follow_head=follow_head,
    )


def _claimed(declarations: list[ProviderToolDeclaration], providers: list[str | None]) -> list[set[str]]:
    return [set(c.claimed) for c in decide_candidate_tools(declarations, providers=providers, dialect=Dialect.MESSAGES)]


@pytest.mark.parametrize(
    ("providers", "claimed"),
    [
        (["anthropic"], [set()]),
        (["anthropic", "anthropic"], [set(), set()]),
        # Claude falling back to Vertex or Bedrock: only the fallback runs on Otari.
        (["anthropic", "vertexai"], [set(), {CODE}]),
        (["anthropic", "bedrock"], [set(), {CODE}]),
        (["mistral", "anthropic"], [{CODE}, set()]),
        (["anthropic", None], [set(), {CODE}]),
    ],
)
def test_auto_decides_each_candidate_on_its_own(providers: list[str | None], claimed: list[set[str]]) -> None:
    assert _claimed([_declared(CODE_EXECUTION_TOOL, ANTHROPIC_CODE, CodeExecutor.AUTO)], providers) == claimed


@pytest.mark.parametrize(
    ("preference", "backend", "claimed"),
    [
        (CodeExecutor.OTARI, True, {CODE}),
        (CodeExecutor.PROVIDER, True, set()),
        (None, True, set()),
        # With nowhere to run it, nothing is claimed whatever was asked.
        (CodeExecutor.AUTO, False, set()),
        (CodeExecutor.OTARI, False, set()),
    ],
)
def test_every_other_preference_decides_every_candidate_alike(
    preference: CodeExecutor | None, backend: bool, claimed: set[str]
) -> None:
    declaration = _declared(CODE_EXECUTION_TOOL, ANTHROPIC_CODE, preference, backend=backend)
    assert _claimed([declaration], ["anthropic", "vertexai"]) == [claimed, claimed]


def test_a_uniform_declaration_is_claimed_for_every_candidate_where_any_needs_it() -> None:
    declaration = _declared(CODE_EXECUTION_TOOL, ANTHROPIC_CODE, CodeExecutor.AUTO, uniform=True)

    assert _claimed([declaration], ["anthropic", "vertexai"]) == [{CODE}, {CODE}]
    assert _claimed([declaration], ["anthropic", "anthropic"]) == [set(), set()]


def test_a_head_following_declaration_is_decided_for_every_candidate_as_the_head_is() -> None:
    declaration = _declared(CODE_EXECUTION_TOOL, ANTHROPIC_CODE, CodeExecutor.AUTO, follow_head=True)

    assert _claimed([declaration], ["anthropic", "vertexai"]) == [set(), set()]
    assert _claimed([declaration], ["vertexai", "anthropic"]) == [{CODE}, {CODE}]
    [head, fallback] = decide_candidate_tools(
        [declaration], providers=["anthropic", "vertexai"], dialect=Dialect.MESSAGES
    )
    assert head.native == {CODE}
    assert fallback.native == frozenset()


def test_each_tool_is_decided_on_its_own_answer() -> None:
    declarations = [
        _declared(CODE_EXECUTION_TOOL, ANTHROPIC_CODE, CodeExecutor.AUTO),
        _declared(_tool(WEB_SEARCH_TOOL_NAME), ANTHROPIC_SEARCH, CodeExecutor.OTARI),
        _declared(_tool(WEB_FETCH_TOOL_NAME), ANTHROPIC_FETCH, None),
    ]

    decided = decide_candidate_tools(declarations, providers=["anthropic", "vertexai"], dialect=Dialect.MESSAGES)

    assert decided == [
        CandidateTools(claimed=frozenset({WEB_SEARCH_TOOL_NAME}), native=frozenset({CODE, WEB_FETCH_TOOL_NAME})),
        CandidateTools(claimed=frozenset({CODE, WEB_SEARCH_TOOL_NAME})),
    ]


def test_the_dialect_is_part_of_the_question() -> None:
    declaration = _declared(_tool(WEB_SEARCH_TOOL_NAME), ANTHROPIC_SEARCH, CodeExecutor.AUTO)

    on_chat = decide_candidate_tools([declaration], providers=["anthropic"], dialect=Dialect.CHAT)

    assert on_chat == [CandidateTools(claimed=frozenset({WEB_SEARCH_TOOL_NAME}))]


@pytest.mark.parametrize("tool", BUILTIN_TOOLS, ids=lambda tool: tool.name)
def test_every_listed_tool_answers_the_native_question(tool: BuiltinTool) -> None:
    assert tool.runs_natively(None, provider="anthropic", dialect=Dialect.MESSAGES) is False
    assert tool.runs_natively({"type": f"otari_{tool.name}"}, provider="anthropic", dialect=Dialect.MESSAGES) is False
