"""`message` is optional on every gate type, and each type falls back to its own line.

Requiring it only ever forced an author to restate what the gate's other
fields already say. It was worst on a `judge` gate, whose `rubric` states the
rule at length one field above, which is why `otari guardrails generate` kept
proposing judge gates without one and bouncing them off the validator.
"""

import pytest

from otari_agent.domain.check import run_policy_check
from otari_agent.domain.policy import PolicyError, parse_policy
from otari_agent.domain.types import Outcome

_HEADER = 'schema_version: "1.0"\npolicy:\n  id: test\ngates:\n'

_GATES_WITHOUT_MESSAGE = {
    "path": (
        "  - id: g\n    type: path\n    runs: [stop.working_tree]\n"
        "    enforcement: required\n    forbidden: ['CHANGELOG.md']\n"
    ),
    "command": (
        "  - id: g\n    type: command\n    runs: [pre_tool_use.command]\n"
        "    enforcement: required\n    forbidden: ['git push --force']\n"
    ),
    "command_if_changed": (
        "  - id: g\n    type: command_if_changed\n    runs: [stop.session]\n"
        "    enforcement: required\n    when_changed: ['spec.json']\n    require: ['make postman']\n"
    ),
    "verifier": (
        "  - id: g\n    type: verifier\n    runs: [stop.verifier]\n"
        "    enforcement: required\n    verifier: scripts/check.sh\n"
    ),
    "judge": (
        "  - id: g\n    type: judge\n    runs: [stop.session]\n"
        "    enforcement: advisory\n    rubric: Does this change follow the conventions?\n"
    ),
}

_EXPECTED_FALLBACK = {
    "path": "A path this gate forbids was matched.",
    "command": "A command this gate forbids was run.",
    "command_if_changed": "A watched path changed without this gate's required command.",
    "verifier": "This gate's verifier reported a failure.",
    "judge": "This turn did not meet this gate's rubric.",
}


@pytest.mark.parametrize("gate_type", sorted(_GATES_WITHOUT_MESSAGE))
def test_every_gate_type_parses_without_a_message(gate_type: str) -> None:
    spec = parse_policy(_HEADER + _GATES_WITHOUT_MESSAGE[gate_type], source="t")

    assert spec.gates[0].message == ""
    assert spec.gates[0].failure_message == _EXPECTED_FALLBACK[gate_type]


@pytest.mark.parametrize("gate_type", sorted(_GATES_WITHOUT_MESSAGE))
def test_an_author_supplied_message_still_wins(gate_type: str) -> None:
    """The fallback is a floor, not a replacement: otari's own 23 gates keep their wording."""
    spec = parse_policy(_HEADER + _GATES_WITHOUT_MESSAGE[gate_type] + "    message: use the tool instead\n", source="t")

    assert spec.gates[0].failure_message == "use the tool instead"


def test_a_whitespace_only_message_falls_back() -> None:
    """Otherwise a gate renders a blank line where its failure reason belongs."""
    spec = parse_policy(_HEADER + _GATES_WITHOUT_MESSAGE["path"] + '    message: "   "\n', source="t")

    assert spec.gates[0].failure_message == "A path this gate forbids was matched."


def test_a_non_string_message_is_still_rejected() -> None:
    """Optional is not untyped: a YAML scalar that is not a string is an author error."""
    with pytest.raises(PolicyError, match="'message' must be a string"):
        parse_policy(_HEADER + _GATES_WITHOUT_MESSAGE["path"] + "    message: 42\n", source="t")


def test_the_fallback_reaches_a_real_failing_result() -> None:
    """End to end through the evaluator both `otari hook` and the Hook Server call."""
    result = run_policy_check(
        _HEADER + _GATES_WITHOUT_MESSAGE["path"],
        source="t",
        paths=["CHANGELOG.md"],
        path_source="stop.working_tree",
        commands=None,
    )

    assert result.blocked
    assert result.results[0].outcome is Outcome.FAIL
    assert result.results[0].message == "A path this gate forbids was matched."
