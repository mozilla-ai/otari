from gateway.agent_runtime.domain.evaluators import (
    _command_segments,
    _contains_subsequence,
    evaluate_command_match,
)
from gateway.agent_runtime.domain.types import CommandEvidence, CommandMatchGate, Outcome


def _gate(**overrides: object) -> CommandMatchGate:
    defaults: dict[str, object] = {
        "id": "no-force-push",
        "enforcement": "required",
        "forbidden": ("git push --force", "git push -f"),
        "message": "Force-pushing is not allowed.",
    }
    defaults.update(overrides)
    return CommandMatchGate(**defaults)  # type: ignore[arg-type]


def test_pass_when_no_forbidden_command_run() -> None:
    result = evaluate_command_match(_gate(), CommandEvidence(commands=("git status", "npm test")))
    assert result.outcome is Outcome.PASS
    assert not result.outcome.is_blocking


def test_fail_when_a_forbidden_phrase_matches() -> None:
    result = evaluate_command_match(_gate(), CommandEvidence(commands=("git push --force",)))
    assert result.outcome is Outcome.FAIL
    assert "git push --force" in (result.detail or "")


def test_unknown_when_evidence_was_not_collected() -> None:
    result = evaluate_command_match(_gate(), None)
    assert result.outcome is Outcome.UNKNOWN
    assert result.outcome.is_blocking, "unknown must block a required gate, never pass silently"


def test_advisory_gate_does_not_block_required() -> None:
    gate = _gate(enforcement="advisory")
    result = evaluate_command_match(gate, CommandEvidence(commands=("git push --force",)))
    assert result.outcome is Outcome.FAIL
    assert result.enforcement == "advisory"


# --- The two false-positive bugs a raw substring check would have had ------


def test_pnpm_is_not_flagged_by_a_forbidden_npm_phrase() -> None:
    """'pnpm install' contains the substring 'npm ' but not the token 'npm'."""
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=("pnpm install",)))
    assert result.outcome is Outcome.PASS


def test_npm_invocation_is_flagged() -> None:
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=("npm install",)))
    assert result.outcome is Outcome.FAIL


def test_force_with_lease_is_not_flagged_by_a_forbidden_force_phrase() -> None:
    """--force-with-lease is the safer alternative this gate should allow,

    not a string containing the substring "--force".
    """
    gate = _gate(forbidden=("git push --force",))
    result = evaluate_command_match(gate, CommandEvidence(commands=("git push --force-with-lease",)))
    assert result.outcome is Outcome.PASS


# --- Contiguous-subsequence matching, not anchored to position 0 -----------


def test_sudo_prefixed_command_is_still_matched() -> None:
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=("sudo npm install",)))
    assert result.outcome is Outcome.FAIL


def test_extra_trailing_flags_do_not_prevent_a_match() -> None:
    gate = _gate(forbidden=("git push --force",))
    result = evaluate_command_match(gate, CommandEvidence(commands=("git push --force --no-verify",)))
    assert result.outcome is Outcome.FAIL


# --- Segments: &&, ;, |, || separate simple commands ------------------------


def test_compound_command_with_and_operator_is_caught() -> None:
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=("cd frontend && npm install",)))
    assert result.outcome is Outcome.FAIL


def test_piped_command_is_caught() -> None:
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=("echo hi | npm install",)))
    assert result.outcome is Outcome.FAIL


def test_quoted_argument_containing_operator_text_is_not_split() -> None:
    """A quoted argument that happens to contain '&&' is one token, not a

    segment boundary: shlex tokenizes it as a single argument before this
    ever looks for separators, so a comment or commit message mentioning the
    forbidden phrase is not itself a forbidden invocation of it.
    """
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=('git commit -m "npm && build"',)))
    assert result.outcome is Outcome.PASS


def test_unbalanced_quotes_do_not_crash_and_do_not_match() -> None:
    """A command shlex cannot tokenize becomes its own opaque single token,

    never equal to a real multi-token forbidden phrase.
    """
    gate = _gate(forbidden=("git push --force",))
    result = evaluate_command_match(gate, CommandEvidence(commands=('git push "--force',)))
    assert result.outcome is Outcome.PASS


def test_operators_glued_with_no_surrounding_whitespace_are_a_known_gap() -> None:
    """Documented, not fixed: shlex only sees '&&' as its own token when

    whitespace-separated, so 'a&&b' stays one token ('a&&b'/'frontend&&npm')
    and is never split into two segments. Pinned here so a future change to
    this behavior is a deliberate decision, not a silent regression either
    way.
    """
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=("cd frontend&&npm install",)))
    assert result.outcome is Outcome.PASS


# --- _command_segments / _contains_subsequence: direct unit coverage -------


def test_command_segments_splits_on_control_operators() -> None:
    assert _command_segments("npm install && git push --force") == [
        ["npm", "install"],
        ["git", "push", "--force"],
    ]


def test_command_segments_keeps_a_quoted_operator_as_one_token() -> None:
    assert _command_segments('echo "a && b"') == [["echo", "a && b"]]


def test_command_segments_falls_back_to_one_token_on_unbalanced_quotes() -> None:
    assert _command_segments('git push "--force') == [['git push "--force']]


def test_contains_subsequence_finds_the_phrase_anywhere() -> None:
    assert _contains_subsequence(["sudo", "npm", "install"], ["npm"]) is True
    assert _contains_subsequence(["npm", "install"], ["npm", "run"]) is False


def test_contains_subsequence_requires_contiguous_order() -> None:
    assert _contains_subsequence(["git", "push", "--force"], ["push", "git"]) is False


def test_contains_subsequence_rejects_an_empty_phrase() -> None:
    assert _contains_subsequence(["npm", "install"], []) is False
