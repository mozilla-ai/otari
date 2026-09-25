from otari_agent.domain.evaluators import evaluate_command_if_changed, tokenize_commands
from otari_agent.domain.types import CommandEvidence, CommandIfChangedGate, Outcome, PathEvidence


def _gate(**overrides: object) -> CommandIfChangedGate:
    defaults: dict[str, object] = {
        "runs": ("stop.session",),
        "id": "openapi-needs-postman",
        "enforcement": "required",
        "when_changed": ("docs/public/openapi.json",),
        "require": ("make postman",),
        "message": "Run make postman after changing openapi.json.",
    }
    defaults.update(overrides)
    return CommandIfChangedGate(**defaults)  # type: ignore[arg-type]


def test_pass_when_path_matches_and_required_command_ran() -> None:
    result = evaluate_command_if_changed(
        _gate(),
        PathEvidence(paths=("docs/public/openapi.json",)),
        CommandEvidence(commands=("make postman",), scope="session"),
    )
    assert result.outcome is Outcome.PASS
    assert not result.outcome.is_blocking


def test_fail_when_path_matches_and_required_command_did_not_run() -> None:
    result = evaluate_command_if_changed(
        _gate(),
        PathEvidence(paths=("docs/public/openapi.json",)),
        CommandEvidence(commands=("git status", "make lint"), scope="session"),
    )
    assert result.outcome is Outcome.FAIL
    assert "docs/public/openapi.json" in (result.detail or "")


def test_not_applicable_under_call_scope_even_when_a_path_matches() -> None:
    """A PreToolUse call cannot answer this gate, so it must not try.

    Such a call submits its own edited path as changed_paths and its one
    command (or none, for an edit tool) as call-scoped evidence, before the
    edit has even run. Failing there would permanently block every edit to a
    when_changed-matched path, since the required command can never have
    already run in response to a change that has not happened yet.
    """
    result = evaluate_command_if_changed(
        _gate(),
        PathEvidence(paths=("docs/public/openapi.json",)),
        CommandEvidence(commands=("git status",), scope="call"),
    )
    assert result.outcome is Outcome.NOT_APPLICABLE
    assert not result.outcome.is_blocking


def test_fail_when_session_scope_collected_no_commands_at_all() -> None:
    """Session-scoped and empty is a real answer, not a missing one.

    The session changed a matched path and ran no command at all, so the
    required one is among the commands it did not run. Reading this as
    not_applicable (which it had to be before scope existed, since an empty
    list from a PreToolUse edit call looked identical) let a session satisfy
    the gate by never invoking Bash.
    """
    result = evaluate_command_if_changed(
        _gate(),
        PathEvidence(paths=("docs/public/openapi.json",)),
        CommandEvidence(commands=(), scope="session"),
    )
    assert result.outcome is Outcome.FAIL
    assert result.outcome.is_blocking
    assert "docs/public/openapi.json" in (result.detail or "")


def test_not_applicable_when_no_path_matches() -> None:
    result = evaluate_command_if_changed(
        _gate(),
        PathEvidence(paths=("README.md",)),
        CommandEvidence(commands=(), scope="session"),
    )
    assert result.outcome is Outcome.NOT_APPLICABLE
    assert not result.outcome.is_blocking


def test_not_applicable_wins_over_missing_commands_when_nothing_relevant_changed() -> None:
    """Nothing this gate cares about changed, so it never gets to ask whether

    a command ran: not_applicable, not unknown, even with no command evidence.
    """
    result = evaluate_command_if_changed(
        _gate(),
        PathEvidence(paths=()),
        CommandEvidence(commands=(), scope="session"),
    )
    assert result.outcome is Outcome.NOT_APPLICABLE


def test_unknown_when_changed_path_evidence_was_not_collected() -> None:
    result = evaluate_command_if_changed(_gate(), None, CommandEvidence(commands=("make postman",), scope="session"))
    assert result.outcome is Outcome.UNKNOWN
    assert result.outcome.is_blocking, "unknown must block a required gate, never pass silently"


def test_unknown_when_command_evidence_was_not_collected() -> None:
    result = evaluate_command_if_changed(_gate(), PathEvidence(paths=("docs/public/openapi.json",)), None)
    assert result.outcome is Outcome.UNKNOWN


def test_unknown_when_neither_evidence_kind_was_collected() -> None:
    result = evaluate_command_if_changed(_gate(), None, None)
    assert result.outcome is Outcome.UNKNOWN


def test_any_one_require_phrase_satisfies_the_gate() -> None:
    gate = _gate(require=("make postman", "python scripts/generate_postman.py"))
    result = evaluate_command_if_changed(
        gate,
        PathEvidence(paths=("docs/public/openapi.json",)),
        CommandEvidence(commands=("python scripts/generate_postman.py --check",), scope="session"),
    )
    assert result.outcome is Outcome.PASS


def test_advisory_gate_does_not_block_required() -> None:
    gate = _gate(enforcement="advisory")
    result = evaluate_command_if_changed(
        gate,
        PathEvidence(paths=("docs/public/openapi.json",)),
        CommandEvidence(commands=("git status",), scope="session"),
    )
    assert result.outcome is Outcome.FAIL
    assert result.enforcement == "advisory"


def test_shares_a_precomputed_segment_cache() -> None:
    """Mirrors evaluate_command's own segment_cache contract: a caller

    evaluating several command-evidence gates against the same evidence
    tokenizes once via tokenize_commands and passes the same cache to every
    call, rather than each call re-tokenizing from scratch.
    """
    evidence = CommandEvidence(commands=("make postman",), scope="session")
    cache = tokenize_commands(evidence.commands)
    result = evaluate_command_if_changed(
        _gate(),
        PathEvidence(paths=("docs/public/openapi.json",)),
        evidence,
        segment_cache=cache,
    )
    assert result.outcome is Outcome.PASS
