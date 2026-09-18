from gateway.agent_runtime.domain.evaluators import evaluate_judge
from gateway.agent_runtime.domain.types import ChangedPathEvidence, JudgeEvidence, JudgeGate, JudgeVerdict, Outcome


def _gate(**overrides: object) -> JudgeGate:
    defaults: dict[str, object] = {
        "id": "follows-error-handling-pattern",
        "enforcement": "advisory",
        "rubric": "Does this change follow the repository's error-handling conventions?",
        "message": "This change does not follow the error-handling conventions.",
    }
    defaults.update(overrides)
    return JudgeGate(**defaults)  # type: ignore[arg-type]


def test_unknown_when_no_evidence_was_submitted_at_all() -> None:
    result = evaluate_judge(_gate(), None, None)
    assert result.outcome is Outcome.UNKNOWN
    assert result.outcome.is_blocking, "unknown must block, never pass silently"


def test_unknown_when_evidence_has_no_verdict_for_this_gate() -> None:
    """A submitted list missing this gate's id is treated the same as no evidence at all.

    Unlike ChangedPathEvidence/CommandEvidence, a verdict already names the
    one gate it judged, so there is no separate "collected, and there is
    none" case to distinguish here.
    """
    evidence = JudgeEvidence(verdicts=(JudgeVerdict(gate_id="some-other-gate", outcome="pass", reasoning=""),))
    result = evaluate_judge(_gate(), None, evidence)
    assert result.outcome is Outcome.UNKNOWN


def test_unconditional_gate_ignores_changed_path_evidence_entirely() -> None:
    """A gate with no `when_changed` (the default) always applies.

    Passing changed-path evidence that matches nothing, or no evidence at
    all, must not affect a gate that never declared any `when_changed` globs
    to begin with.
    """
    evidence = JudgeEvidence(
        verdicts=(JudgeVerdict(gate_id="follows-error-handling-pattern", outcome="pass", reasoning="fine"),)
    )
    result = evaluate_judge(_gate(), ChangedPathEvidence(changed_paths=()), evidence)
    assert result.outcome is Outcome.PASS


def test_unknown_when_when_changed_is_set_but_no_changed_path_evidence_was_submitted() -> None:
    gate = _gate(when_changed=("src/**",))
    evidence = JudgeEvidence(
        verdicts=(JudgeVerdict(gate_id="follows-error-handling-pattern", outcome="pass", reasoning="fine"),)
    )
    result = evaluate_judge(gate, None, evidence)
    assert result.outcome is Outcome.UNKNOWN
    assert result.outcome.is_blocking


def test_not_applicable_when_when_changed_globs_match_nothing_that_changed() -> None:
    gate = _gate(when_changed=("src/**",))
    changed_path_evidence = ChangedPathEvidence(changed_paths=("docs/README.md",))
    evidence = JudgeEvidence(
        verdicts=(JudgeVerdict(gate_id="follows-error-handling-pattern", outcome="fail", reasoning="should not run"),)
    )
    result = evaluate_judge(gate, changed_path_evidence, evidence)
    assert result.outcome is Outcome.NOT_APPLICABLE
    assert not result.outcome.is_blocking


def test_when_changed_gate_still_resolves_the_verdict_once_a_matching_path_changed() -> None:
    gate = _gate(when_changed=("src/**",))
    changed_path_evidence = ChangedPathEvidence(changed_paths=("src/module.py",))
    evidence = JudgeEvidence(
        verdicts=(JudgeVerdict(gate_id="follows-error-handling-pattern", outcome="fail", reasoning="swallows"),)
    )
    result = evaluate_judge(gate, changed_path_evidence, evidence)
    assert result.outcome is Outcome.FAIL
    assert result.detail == "swallows"


def test_first_matching_verdict_wins_when_a_gate_id_is_duplicated() -> None:
    """A caller submitting two verdicts for the same gate id is malformed input,

    not something this evaluator can reject (evidence is a plain list, not
    keyed on gate_id, and Otari does not verify caller-reported evidence
    either way). The first match in submission order is used, deterministically,
    rather than the last or an arbitrary one.
    """
    evidence = JudgeEvidence(
        verdicts=(
            JudgeVerdict(gate_id="follows-error-handling-pattern", outcome="pass", reasoning="first"),
            JudgeVerdict(gate_id="follows-error-handling-pattern", outcome="fail", reasoning="second"),
        )
    )
    result = evaluate_judge(_gate(), None, evidence)
    assert result.outcome is Outcome.PASS
    assert result.detail == "first"


def test_pass_when_the_model_verdict_is_pass() -> None:
    evidence = JudgeEvidence(
        verdicts=(JudgeVerdict(gate_id="follows-error-handling-pattern", outcome="pass", reasoning="looks fine"),)
    )
    result = evaluate_judge(_gate(), None, evidence)
    assert result.outcome is Outcome.PASS
    assert not result.outcome.is_blocking
    assert result.detail == "looks fine"


def test_fail_when_the_model_verdict_is_fail() -> None:
    evidence = JudgeEvidence(
        verdicts=(
            JudgeVerdict(gate_id="follows-error-handling-pattern", outcome="fail", reasoning="swallows exceptions"),
        )
    )
    gate = _gate()
    result = evaluate_judge(gate, None, evidence)
    assert result.outcome is Outcome.FAIL
    assert result.outcome.is_blocking
    assert result.message == gate.message
    assert result.detail == "swallows exceptions"


def test_error_when_the_callers_model_call_itself_failed() -> None:
    """error is distinct from fail: the model call did not produce a real verdict.

    Still is_blocking like any other unresolved check, but the gate's own
    enforcement is always "advisory" (enforced at parse time), so this can
    only ever warn, never block a required gate.
    """
    evidence = JudgeEvidence(
        verdicts=(
            JudgeVerdict(
                gate_id="follows-error-handling-pattern",
                outcome="error",
                reasoning="the `claude` CLI was not found on PATH",
            ),
        )
    )
    result = evaluate_judge(_gate(), None, evidence)
    assert result.outcome is Outcome.ERROR
    assert result.outcome.is_blocking
    assert result.enforcement == "advisory"
    assert result.detail == "the `claude` CLI was not found on PATH"
