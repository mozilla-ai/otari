from gateway.agent_runtime.domain.evaluators import evaluate_check_passed
from gateway.agent_runtime.domain.types import (
    ChangedPathEvidence,
    CheckEvidence,
    CheckPassedGate,
    CheckVerdict,
    Outcome,
)


def _gate(**overrides: object) -> CheckPassedGate:
    defaults: dict[str, object] = {
        "id": "no-leftover-conflict-markers",
        "enforcement": "required",
        "verifier": ".otari-gates/verifiers/no-conflict-markers.sh",
        "message": "A tracked file still carries a Git merge-conflict marker.",
    }
    defaults.update(overrides)
    return CheckPassedGate(**defaults)  # type: ignore[arg-type]


def test_not_applicable_when_evidence_is_omitted_entirely() -> None:
    """`evidence=None` means this caller's event type never runs check_passed gates at all

    (otari hook on PreToolUse), distinct from a caller that does and is
    missing a verdict for this one (`evidence=CheckEvidence(verdicts=())`,
    see test_unknown_when_evidence_has_no_verdict_for_this_gate): the former
    must never warn or block on every single matching PreToolUse edit.
    """
    result = evaluate_check_passed(_gate(), None, None)
    assert result.outcome is Outcome.NOT_APPLICABLE
    assert not result.outcome.is_blocking


def test_unknown_when_evidence_is_submitted_but_empty() -> None:
    """A caller that does run check_passed gates for this event (`CheckEvidence(verdicts=())`,

    not `None`) but is genuinely missing this one's verdict still resolves
    `unknown`, not `not_applicable`: `None` is the only signal that means
    "this event never checks", never an empty-but-present evidence object.
    """
    result = evaluate_check_passed(_gate(), None, CheckEvidence(verdicts=()))
    assert result.outcome is Outcome.UNKNOWN
    assert result.outcome.is_blocking


def test_unknown_when_evidence_has_no_verdict_for_this_gate() -> None:
    evidence = CheckEvidence(verdicts=(CheckVerdict(gate_id="some-other-gate", outcome="pass", detail=""),))
    result = evaluate_check_passed(_gate(), None, evidence)
    assert result.outcome is Outcome.UNKNOWN


def test_unconditional_gate_ignores_changed_path_evidence_entirely() -> None:
    """A gate with no `when_changed` (the default) always applies."""
    evidence = CheckEvidence(
        verdicts=(CheckVerdict(gate_id="no-leftover-conflict-markers", outcome="pass", detail=""),)
    )
    result = evaluate_check_passed(_gate(), ChangedPathEvidence(changed_paths=()), evidence)
    assert result.outcome is Outcome.PASS


def test_unknown_when_when_changed_is_set_but_no_changed_path_evidence_was_submitted() -> None:
    gate = _gate(when_changed=("src/**",))
    evidence = CheckEvidence(
        verdicts=(CheckVerdict(gate_id="no-leftover-conflict-markers", outcome="pass", detail=""),)
    )
    result = evaluate_check_passed(gate, None, evidence)
    assert result.outcome is Outcome.UNKNOWN
    assert result.outcome.is_blocking


def test_not_applicable_when_when_changed_globs_match_nothing_that_changed() -> None:
    gate = _gate(when_changed=("src/**",))
    changed_path_evidence = ChangedPathEvidence(changed_paths=("docs/README.md",))
    evidence = CheckEvidence(
        verdicts=(CheckVerdict(gate_id="no-leftover-conflict-markers", outcome="fail", detail="should not run"),)
    )
    result = evaluate_check_passed(gate, changed_path_evidence, evidence)
    assert result.outcome is Outcome.NOT_APPLICABLE
    assert not result.outcome.is_blocking


def test_when_changed_gate_still_resolves_the_verdict_once_a_matching_path_changed() -> None:
    gate = _gate(when_changed=("src/**",))
    changed_path_evidence = ChangedPathEvidence(changed_paths=("src/module.py",))
    evidence = CheckEvidence(
        verdicts=(CheckVerdict(gate_id="no-leftover-conflict-markers", outcome="fail", detail="conflicted.txt:2"),)
    )
    result = evaluate_check_passed(gate, changed_path_evidence, evidence)
    assert result.outcome is Outcome.FAIL
    assert result.detail == "conflicted.txt:2"


def test_first_matching_verdict_wins_when_a_gate_id_is_duplicated() -> None:
    evidence = CheckEvidence(
        verdicts=(
            CheckVerdict(gate_id="no-leftover-conflict-markers", outcome="pass", detail="first"),
            CheckVerdict(gate_id="no-leftover-conflict-markers", outcome="fail", detail="second"),
        )
    )
    result = evaluate_check_passed(_gate(), None, evidence)
    assert result.outcome is Outcome.PASS
    assert result.detail == "first"


def test_pass_when_the_verifier_exit_code_is_zero() -> None:
    evidence = CheckEvidence(
        verdicts=(CheckVerdict(gate_id="no-leftover-conflict-markers", outcome="pass", detail=""),)
    )
    result = evaluate_check_passed(_gate(), None, evidence)
    assert result.outcome is Outcome.PASS
    assert not result.outcome.is_blocking


def test_fail_when_the_verifier_exit_code_is_one() -> None:
    """Unlike judge, this gate's default enforcement here is `required`, so a fail genuinely blocks."""
    evidence = CheckEvidence(
        verdicts=(CheckVerdict(gate_id="no-leftover-conflict-markers", outcome="fail", detail="conflicted.txt:2"),)
    )
    gate = _gate()
    result = evaluate_check_passed(gate, None, evidence)
    assert result.outcome is Outcome.FAIL
    assert result.outcome.is_blocking
    assert result.enforcement == "required"
    assert result.message == gate.message
    assert result.detail == "conflicted.txt:2"


def test_error_when_the_verifier_could_not_be_run() -> None:
    """error is distinct from fail: the verifier itself never produced a real verdict.

    Still is_blocking, and unlike judge (always advisory), this gate can be
    required, so an error here can genuinely block.
    """
    evidence = CheckEvidence(
        verdicts=(
            CheckVerdict(
                gate_id="no-leftover-conflict-markers",
                outcome="error",
                detail="verifier '.otari-gates/verifiers/no-conflict-markers.sh' does not exist",
            ),
        )
    )
    gate = _gate()
    result = evaluate_check_passed(gate, None, evidence)
    assert result.outcome is Outcome.ERROR
    assert result.outcome.is_blocking
    assert result.enforcement == "required"
    assert result.detail == "verifier '.otari-gates/verifiers/no-conflict-markers.sh' does not exist"
