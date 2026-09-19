from gateway.agent_runtime.domain.check import PolicyCheckError, run_policy_check
from gateway.agent_runtime.domain.types import CheckVerdict, JudgeVerdict, Outcome

_CHANGED_PATH_POLICY = (
    'schema_version: "1.0"\npolicy:\n  id: test\ngates:\n'
    "  - id: g\n    type: changed_path\n    enforcement: required\n"
    '    forbidden: ["CHANGELOG.md"]\n    message: no hand edits\n'
)

_JUDGE_POLICY = (
    'schema_version: "1.0"\npolicy:\n  id: test\ngates:\n'
    "  - id: j\n    type: judge\n    enforcement: advisory\n"
    "    rubric: does it follow convention\n    message: check this\n"
)

_CHECK_POLICY = (
    'schema_version: "1.0"\npolicy:\n  id: test\ngates:\n'
    "  - id: c\n    type: check_passed\n    enforcement: required\n"
    "    verifier: verify.sh\n    message: verifier failed\n"
)


def test_a_forbidden_changed_path_blocks() -> None:
    result = run_policy_check(_CHANGED_PATH_POLICY, source="test", changed_paths=["CHANGELOG.md"], commands=None)
    assert result.policy_id == "test"
    assert result.blocked is True
    assert result.results[0].outcome is Outcome.FAIL


def test_an_unmatched_changed_path_passes_and_does_not_block() -> None:
    result = run_policy_check(_CHANGED_PATH_POLICY, source="test", changed_paths=["README.md"], commands=None)
    assert result.blocked is False
    assert result.results[0].outcome is Outcome.PASS


def test_omitted_changed_paths_resolves_unknown_and_blocks() -> None:
    """None (never collected) is distinct from [] (collected, and there is none)."""
    result = run_policy_check(_CHANGED_PATH_POLICY, source="test", changed_paths=None, commands=None)
    assert result.results[0].outcome is Outcome.UNKNOWN
    assert result.blocked is True


def test_empty_changed_paths_resolves_not_applicable_and_does_not_block() -> None:
    result = run_policy_check(_CHANGED_PATH_POLICY, source="test", changed_paths=[], commands=None)
    assert result.results[0].outcome is Outcome.NOT_APPLICABLE
    assert result.blocked is False


def test_duplicate_changed_paths_do_not_change_the_outcome() -> None:
    result = run_policy_check(
        _CHANGED_PATH_POLICY, source="test", changed_paths=["CHANGELOG.md", "CHANGELOG.md"], commands=None
    )
    assert result.results[0].outcome is Outcome.FAIL


def test_malformed_policy_raises_policy_check_error() -> None:
    try:
        run_policy_check("not: valid: yaml: at: all:\n  - [", source="test", changed_paths=[], commands=None)
    except PolicyCheckError as exc:
        assert "not valid YAML" in str(exc)
    else:
        raise AssertionError("expected PolicyCheckError")


def test_oversize_changed_path_entry_raises_policy_check_error() -> None:
    try:
        run_policy_check(_CHANGED_PATH_POLICY, source="test", changed_paths=["a" * 5000], commands=None)
    except PolicyCheckError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("expected PolicyCheckError")


def test_judge_verdict_is_relayed_into_the_result() -> None:
    result = run_policy_check(
        _JUDGE_POLICY,
        source="test",
        changed_paths=None,
        commands=None,
        judge_results=[JudgeVerdict(gate_id="j", outcome="fail", reasoning="does not follow it")],
    )
    assert result.results[0].outcome is Outcome.FAIL
    assert result.results[0].detail == "does not follow it"
    # advisory: never blocks, regardless of outcome.
    assert result.blocked is False


def test_omitted_judge_results_resolves_not_applicable() -> None:
    result = run_policy_check(_JUDGE_POLICY, source="test", changed_paths=None, commands=None, judge_results=None)
    assert result.results[0].outcome is Outcome.NOT_APPLICABLE


def test_check_passed_verdict_is_relayed_into_the_result() -> None:
    result = run_policy_check(
        _CHECK_POLICY,
        source="test",
        changed_paths=None,
        commands=None,
        check_results=[CheckVerdict(gate_id="c", outcome="fail", detail="conflict markers found")],
    )
    assert result.results[0].outcome is Outcome.FAIL
    assert result.blocked is True


def test_omitted_check_results_resolves_not_applicable_and_does_not_block() -> None:
    result = run_policy_check(_CHECK_POLICY, source="test", changed_paths=None, commands=None, check_results=None)
    assert result.results[0].outcome is Outcome.NOT_APPLICABLE
    assert result.blocked is False
