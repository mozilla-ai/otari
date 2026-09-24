"""`runs`: when a gate runs, and what it can see there.

The field exists because a path gate used to mean two different things
depending on which tool the agent happened to reach for, with no way for the
author to say which they meant. These pin both halves: that a gate declares
its moments, and that evidence from a moment it did not declare resolves
`not_applicable` rather than being read as a clean result.
"""

from __future__ import annotations

import pytest

from otari_agent.domain.check import PolicyCheckError, run_policy_check
from otari_agent.domain.evaluators import evaluate_path
from otari_agent.domain.policy import PolicyError, parse_policy
from otari_agent.domain.types import ChangedPathEvidence, Outcome, PathGate, RunsAt


def _policy(runs: str, gate_type: str = "path", extra: str = '    forbidden: ["CHANGELOG.md"]\n') -> str:
    return (
        'schema_version: "1.0"\n'
        "policy:\n  id: test\n"
        "gates:\n"
        "  - id: g\n"
        f"    type: {gate_type}\n"
        f"    runs: {runs}\n"
        "    enforcement: required\n"
        f"{extra}"
        "    message: m\n"
    )


def _gate(*runs: str) -> PathGate:
    return PathGate(
        id="g",
        enforcement="required",
        runs=runs,  # type: ignore[arg-type]
        forbidden=("CHANGELOG.md",),
        message="m",
    )


def test_a_gate_declares_the_moments_it_runs_at() -> None:
    spec = parse_policy(_policy("[pre_tool_use.edit_target, stop.working_tree]"), source="t")
    assert spec.gates[0].runs == ("pre_tool_use.edit_target", "stop.working_tree")


def test_a_single_moment_may_be_written_as_a_bare_string() -> None:
    """The one-entry case, the same shape judge_cli already allows."""
    spec = parse_policy(_policy("stop.working_tree"), source="t")
    assert spec.gates[0].runs == ("stop.working_tree",)


def test_a_repeated_moment_is_deduplicated_first_seen() -> None:
    spec = parse_policy(_policy("[stop.working_tree, stop.working_tree]"), source="t")
    assert spec.gates[0].runs == ("stop.working_tree",)


def test_a_missing_runs_is_refused_and_names_what_is_legal_for_the_type() -> None:
    """The error is the whole migration story for an older policy, so it must be actionable."""
    policy = _policy("[stop.working_tree]").replace("    runs: [stop.working_tree]\n", "")
    with pytest.raises(PolicyError) as exc:
        parse_policy(policy, source="t")
    assert "needs a non-empty 'runs'" in str(exc.value)
    assert "pre_tool_use.edit_target, stop.working_tree" in str(exc.value)


def test_an_empty_runs_is_refused_rather_than_read_as_always() -> None:
    """A gate that matches no moment can never fire, which is worse than deleting it."""
    with pytest.raises(PolicyError):
        parse_policy(_policy("[]"), source="t")


def test_a_moment_this_gate_type_cannot_run_at_is_refused() -> None:
    with pytest.raises(PolicyError) as exc:
        parse_policy(_policy("[stop.verifier]"), source="t")
    assert "cannot run at stop.verifier" in str(exc.value)


def test_a_command_gate_cannot_claim_a_path_moment() -> None:
    policy = _policy("[pre_tool_use.edit_target]", gate_type="command", extra='    forbidden: ["npm install"]\n')
    with pytest.raises(PolicyError) as exc:
        parse_policy(policy, source="t")
    assert "pre_tool_use.command" in str(exc.value)


def test_evidence_from_a_declared_moment_still_fails_the_gate() -> None:
    result = evaluate_path(
        _gate("pre_tool_use.edit_target", "stop.working_tree"),
        ChangedPathEvidence(changed_paths=("CHANGELOG.md",), source="pre_tool_use.edit_target"),
    )
    assert result.outcome is Outcome.FAIL
    assert result.detail == "CHANGELOG.md"


def test_evidence_from_an_undeclared_moment_is_not_this_gates_to_judge() -> None:
    """The fix itself: a Stop-only gate must stay inert on a tool call rather than pass."""
    result = evaluate_path(
        _gate("stop.working_tree"),
        ChangedPathEvidence(changed_paths=("CHANGELOG.md",), source="pre_tool_use.edit_target"),
    )
    assert result.outcome is Outcome.NOT_APPLICABLE
    assert result.outcome.is_blocking is False
    assert "does not run at pre_tool_use.edit_target" in result.message


def test_the_undeclared_moment_reason_beats_the_empty_list_reason() -> None:
    """Both resolve not_applicable, so the more specific message is the useful one."""
    result = evaluate_path(
        _gate("stop.working_tree"),
        ChangedPathEvidence(changed_paths=(), source="pre_tool_use.edit_target"),
    )
    assert result.outcome is Outcome.NOT_APPLICABLE
    assert "does not run at" in result.message


def test_paths_submitted_without_a_moment_are_refused() -> None:
    """Neither default is safe, so the caller has to say; see run_policy_check."""
    with pytest.raises(PolicyCheckError) as exc:
        run_policy_check(
            _policy("[stop.working_tree]"),
            source="t",
            changed_paths=["CHANGELOG.md"],
            commands=None,
        )
    assert "changed_path_source=None" in str(exc.value)
    # The hint must name only the moments a path gate can actually declare, or a
    # caller follows it into silently losing every path gate.
    assert "pre_tool_use.edit_target, stop.working_tree" in str(exc.value)
    assert "stop.session" not in str(exc.value)


@pytest.mark.parametrize("source", ["stop.session", "stop.verifier", "pre_tool_use.command"])
def test_paths_labeled_with_a_moment_no_path_gate_can_declare_are_refused(source: RunsAt) -> None:
    """Accepting one would resolve every path gate not_applicable: enforcement lost silently.

    This is the same failure the whole field exists to remove, so it has to be
    an error rather than a quiet pass. Reachable in practice because the Hook
    Server is a plain HTTP API and `RunsAt` admits all five values on the wire.
    """
    with pytest.raises(PolicyCheckError) as exc:
        run_policy_check(
            _policy("[stop.working_tree]"),
            source="t",
            changed_paths=["CHANGELOG.md"],
            commands=None,
            changed_path_source=source,
        )
    message = str(exc.value)
    assert source in message
    assert "no path gate can be declared to run at" in message
    assert "pre_tool_use.edit_target, stop.working_tree" in message


def test_an_empty_path_list_needs_no_moment() -> None:
    """`[]` carries no paths to misattribute, and refusing it would abort the whole check."""
    result = run_policy_check(
        _policy("[stop.working_tree]"),
        source="t",
        changed_paths=[],
        commands=[],
    )
    assert result.blocked is False


def test_a_stop_only_gate_does_not_block_a_tool_call_but_still_blocks_the_turn() -> None:
    """End to end over the two moments, which is the behavior the field exists to give."""
    policy = _policy("[stop.working_tree]")
    on_call = run_policy_check(
        policy,
        source="t",
        changed_paths=["CHANGELOG.md"],
        commands=None,
        changed_path_source="pre_tool_use.edit_target",
    )
    assert on_call.blocked is False
    at_stop = run_policy_check(
        policy,
        source="t",
        changed_paths=["CHANGELOG.md"],
        commands=None,
        changed_path_source="stop.working_tree",
    )
    assert at_stop.blocked is True


def test_an_edit_target_only_gate_refuses_the_write_but_leaves_the_turn_alone() -> None:
    """The other half of the pairing, which the docs recommend and nothing pinned."""
    policy = _policy("[pre_tool_use.edit_target]")
    on_call = run_policy_check(
        policy,
        source="t",
        changed_paths=["CHANGELOG.md"],
        commands=None,
        changed_path_source="pre_tool_use.edit_target",
    )
    assert on_call.blocked is True
    at_stop = run_policy_check(
        policy,
        source="t",
        changed_paths=["CHANGELOG.md"],
        commands=None,
        changed_path_source="stop.working_tree",
    )
    assert at_stop.blocked is False
