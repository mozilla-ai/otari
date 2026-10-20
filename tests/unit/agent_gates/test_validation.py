"""`validate_policy`: what a guardrail would otherwise teach by running badly.

Two of these are regressions in the literal sense. This repo's own
guardrail carries a comment for each, recording a footgun found
by a gate quietly not matching (`**/CLAUDE.md` alone) or matching the wrong
thing (a bare `npm` phrase). Both are mechanical properties of the gate
grammar, so both are checkable before the gate ever runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Any, get_args

import pytest

from otari_agent.domain.evaluators import (
    _command_segments,
    _contains_subsequence,
    matched_changed_paths,
    tokenize_phrase,
)
from otari_agent.domain.policy import parse_policy
from otari_agent.domain.types import GATE_TYPES, GateSpec, GateType, VerifierGate, WarningCode
from otari_agent.domain.validation import (
    Finding,
    shallower_twin,
    unmatchable_phrase,
    unreachable_glob,
    validate_policy,
)


def _policy(gates: str) -> str:
    return 'schema_version: "1.0"\npolicy:\n  id: test\ngates:\n' + gates


def _validate(
    gates: str,
    *,
    judge_gate_limit: int = 5,
    verifier_gate_limit: int = 20,
    probe_verifier: Callable[[VerifierGate], str | None] | None = None,
) -> tuple[Finding, ...]:
    spec = parse_policy(_policy(gates), source="test")
    return validate_policy(
        spec,
        judge_gate_limit=judge_gate_limit,
        verifier_gate_limit=verifier_gate_limit,
        probe_verifier=probe_verifier,
    )


def _findings(gates: str, **options: Any) -> tuple[tuple[str, str | None, str], ...]:
    return tuple((finding.severity, finding.gate_id, finding.message) for finding in _validate(gates, **options))


_PATH_GATE = (
    "  - id: g\n"
    "    type: path\n"
    "    runs: [pre_tool_use.edit_target, stop.working_tree]\n"
    "    enforcement: required\n"
    "    forbidden: {forbidden}\n"
    "    message: m\n"
)


def test_a_double_star_glob_without_its_shallower_twin_warns() -> None:
    """The `**/CLAUDE.md` footgun: `**` consumes a segment, so the root file never matches."""
    findings = _findings(_PATH_GATE.format(forbidden='["**/CLAUDE.md"]'))
    assert len(findings) == 1
    severity, gate_id, message = findings[0]
    assert (severity, gate_id) == ("warning", "g")
    assert "'**/CLAUDE.md' can never match 'CLAUDE.md'" in message


def test_the_twin_beside_it_clears_the_warning() -> None:
    assert _findings(_PATH_GATE.format(forbidden='["CLAUDE.md", "**/CLAUDE.md"]')) == ()


def test_a_trailing_double_star_is_not_warned_about() -> None:
    """Its twin would name a bare directory, which Git never reports as a changed path."""
    assert _findings(_PATH_GATE.format(forbidden='["src/gateway/static/dashboard/**"]')) == ()


def test_a_mid_glob_double_star_is_warned_about_at_its_own_depth() -> None:
    findings = _findings(_PATH_GATE.format(forbidden='["src/**/conftest.py"]'))
    assert "can never match 'src/conftest.py'" in findings[0][2]


@pytest.mark.parametrize(
    ("gate_type", "extra"),
    [
        ("command_if_changed", '    require: ["make x"]\n'),
        ("judge", "    rubric: r\n"),
        ("verifier", "    verifier: v.sh\n"),
    ],
)
def test_when_changed_is_checked_on_every_gate_type_that_has_one(gate_type: str, extra: str) -> None:
    runs = "stop.verifier" if gate_type == "verifier" else "stop.session"
    enforcement = "advisory" if gate_type == "judge" else "required"
    findings = _findings(
        f"  - id: g\n    type: {gate_type}\n    runs: [{runs}]\n"
        f"    enforcement: {enforcement}\n"
        '    when_changed: ["**/*.md"]\n'
        f"{extra}    message: m\n"
    )
    assert len(findings) == 1
    assert "when_changed glob '**/*.md' can never match '*.md'" in findings[0][2]


def test_a_single_token_forbidden_phrase_warns() -> None:
    """The bare-`npm` footgun: a phrase matches a token in any position, not only at the head."""
    findings = _findings(
        "  - id: g\n    type: command\n    runs: [pre_tool_use.command]\n"
        '    enforcement: required\n    forbidden: ["npm", "npm install"]\n    message: m\n'
    )
    assert len(findings) == 1
    assert "'npm' is a single token" in findings[0][2]


def test_a_single_token_require_phrase_does_not_warn() -> None:
    """Over-matching a `require` phrase accepts a session sooner; it does not refuse real work."""
    assert (
        _findings(
            "  - id: g\n    type: command_if_changed\n    runs: [stop.session]\n"
            '    enforcement: required\n    when_changed: ["src/x.py"]\n'
            '    require: ["tests/unit/test_x.py"]\n    message: m\n'
        )
        == ()
    )


def test_a_path_gate_that_only_runs_before_an_edit_tool_warns() -> None:
    """It sees a declared edit target and nothing a shell command writes."""
    findings = _findings(
        "  - id: g\n    type: path\n    runs: [pre_tool_use.edit_target]\n"
        '    enforcement: required\n    forbidden: ["CHANGELOG.md"]\n    message: m\n'
    )
    assert len(findings) == 1
    assert "runs at pre_tool_use.edit_target with no stop.working_tree" in findings[0][2]


def test_a_path_gate_with_the_stop_backstop_does_not_warn() -> None:
    assert _findings(_PATH_GATE.format(forbidden='["CHANGELOG.md"]')) == ()


def test_a_stop_only_path_gate_does_not_warn() -> None:
    """After the fact, but complete over the tree: a deliberate choice, not a hole."""
    assert (
        _findings(
            "  - id: g\n    type: path\n    runs: [stop.working_tree]\n"
            '    enforcement: required\n    forbidden: ["CHANGELOG.md"]\n    message: m\n'
        )
        == ()
    )


def test_an_edit_gate_that_also_runs_at_read_still_warns_about_its_missing_backstop() -> None:
    """The edit warning is about the sources present, not about the list being exactly one.

    A gate naming both prevention moments and no `stop.working_tree` has the
    same shell-write hole as an edit-only one, so an exact-tuple test would
    have stopped warning the moment a second source was added beside it.
    """
    findings = _findings(
        "  - id: g\n    type: path\n    runs: [pre_tool_use.edit_target, pre_tool_use.read_target]\n"
        '    enforcement: required\n    forbidden: ["**/.env", ".env"]\n    message: m\n'
    )
    messages = [message for _severity, _gate, message in findings]
    assert any("no stop.working_tree beside it" in message for message in messages)


def test_a_read_gate_is_warned_about_the_shell_and_never_told_to_add_a_backstop() -> None:
    """A read leaves nothing in the tree, so `stop.working_tree` is not the repair.

    Pointing at it would read as a fix and would not be one. The repair is a
    separate `command` gate, which is what this warning has to name.
    """
    findings = _findings(
        "  - id: g\n    type: path\n    runs: [pre_tool_use.read_target, stop.working_tree]\n"
        '    enforcement: required\n    forbidden: [".env"]\n    message: m\n'
    )
    assert len(findings) == 1
    severity, gate_id, message = findings[0]
    assert (severity, gate_id) == ("warning", "g")
    assert "pre_tool_use.read_target" in message
    assert "`cat`" in message
    assert "no command gate here names any of these paths" in message
    assert "Add stop.working_tree" not in message


def test_a_command_gate_naming_one_of_the_paths_clears_the_read_warning() -> None:
    """A warning that cannot be cleared fails `--strict` forever.

    Left unconditional, a correct read gate could never pass `--strict`, which
    would make the source unusable in CI and teach everyone to stop passing it.
    The clearing condition is mechanical: a command gate whose own phrase
    tokenizes to something these globs match.
    """
    assert (
        _findings(
            "  - id: g\n    type: path\n    runs: [pre_tool_use.read_target]\n"
            '    enforcement: required\n    forbidden: [".env"]\n    message: m\n'
            "  - id: shell\n    type: command\n    runs: [pre_tool_use.command]\n"
            '    enforcement: required\n    forbidden: ["cat .env"]\n    message: m\n'
        )
        == ()
    )


def test_an_unrelated_command_gate_does_not_clear_the_read_warning() -> None:
    """Presence of any command gate is too weak a proxy; it has to name a matching path."""
    findings = _findings(
        "  - id: g\n    type: path\n    runs: [pre_tool_use.read_target]\n"
        '    enforcement: required\n    forbidden: [".env"]\n    message: m\n'
        "  - id: pnpm\n    type: command\n    runs: [pre_tool_use.command]\n"
        '    enforcement: required\n    forbidden: ["npm install"]\n    message: m\n'
    )
    assert len(findings) == 1
    assert "pre_tool_use.read_target" in findings[0][2]


def test_a_gate_naming_all_three_moments_warns_only_about_the_read() -> None:
    """The edit half is answered by the backstop; the read half never can be."""
    findings = _findings(
        "  - id: g\n    type: path\n"
        "    runs: [pre_tool_use.edit_target, pre_tool_use.read_target, stop.working_tree]\n"
        '    enforcement: required\n    forbidden: [".env"]\n    message: m\n'
    )
    assert len(findings) == 1
    assert "pre_tool_use.read_target" in findings[0][2]


def _judge_gates(count: int) -> str:
    return "".join(
        f"  - id: j{index}\n    type: judge\n    runs: [stop.session]\n"
        "    enforcement: advisory\n    rubric: r\n    message: m\n"
        for index in range(count)
    )


def test_judge_gates_past_the_per_stop_cap_are_named_in_declaration_order() -> None:
    findings = _findings(_judge_gates(7), judge_gate_limit=5)
    assert len(findings) == 1
    severity, gate_id, message = findings[0]
    assert (severity, gate_id) == ("warning", None)
    assert "these are skipped: j5, j6" in message


def test_judge_gates_at_the_cap_do_not_warn() -> None:
    assert _findings(_judge_gates(5), judge_gate_limit=5) == ()


_VERIFIER_GATE = (
    "  - id: g\n    type: verifier\n    runs: [stop.verifier]\n"
    "    enforcement: required\n    verifier: v.sh\n    message: m\n"
)


def test_a_verifier_problem_is_an_error_not_a_warning() -> None:
    findings = _findings(_VERIFIER_GATE, probe_verifier=lambda _: "verifier 'v.sh' does not exist.")
    assert findings == (("error", "g", "verifier 'v.sh' does not exist."),)


def test_verifier_gates_are_left_alone_without_a_probe() -> None:
    """The probe is the caller's filesystem access; this module never reaches for one itself."""
    assert _findings(_VERIFIER_GATE) == ()


@pytest.mark.parametrize(
    ("glob", "expected"),
    [
        ("**/CLAUDE.md", "CLAUDE.md"),
        ("**/*.md", "*.md"),
        ("src/**/conftest.py", "src/conftest.py"),
        ("web/src/**", None),
        ("CHANGELOG.md", None),
        ("a****b/c", None),
    ],
)
def test_shallower_twin(glob: str, expected: str | None) -> None:
    """`a****b` is a literal-with-stars segment, not a directory-crossing one."""
    assert shallower_twin(glob) == expected


# --- The claims these warnings make about the matcher, pinned to the matcher ---
#
# Every finding below asserts a property of domain/evaluators.py. Without a
# test binding the two, a change there turns a warning into a confident lie and
# nothing fails, which is the exact failure mode this module exists to remove.


def test_the_double_star_claim_holds_in_the_real_matcher() -> None:
    assert matched_changed_paths(("**/CLAUDE.md",), ("CLAUDE.md",)) == ()
    assert matched_changed_paths(("**/CLAUDE.md",), ("web/CLAUDE.md",)) == ("web/CLAUDE.md",)
    assert matched_changed_paths(("src/**/x.py",), ("src/x.py",)) == ()


def test_the_trailing_double_star_exemption_holds_in_the_real_matcher() -> None:
    """Its twin would name a bare directory, which is why it is exempt."""
    assert matched_changed_paths(("web/src/**",), ("web/src/a.ts",)) == ("web/src/a.ts",)
    assert matched_changed_paths(("web/src/**",), ("web/src",)) == ()


@pytest.mark.parametrize("glob", ["./CHANGELOG.md", "/CHANGELOG.md", "a//b", "../x.md", "a/./b"])
def test_the_unreachable_glob_claim_holds_in_the_real_matcher(glob: str) -> None:
    paths = ("CHANGELOG.md", "a/b", "x.md", "web/src/a.ts")
    assert unreachable_glob(glob) is not None
    assert matched_changed_paths((glob,), paths) == ()


def test_the_unmatchable_phrase_claim_holds_in_the_real_matcher() -> None:
    """A separator-carrying phrase does not match even the command it was copied from."""
    phrase = "make postman && make openapi"
    assert unmatchable_phrase(phrase, field="forbidden") is not None
    tokens = tokenize_phrase(phrase)
    assert not any(_contains_subsequence(segment, tokens) for segment in _command_segments(phrase))


# --- New checks ---


@pytest.mark.parametrize("field", ["forbidden", "require"])
def test_a_phrase_carrying_a_shell_separator_is_an_error(field: str) -> None:
    """On `require` this is a required gate nothing can ever satisfy."""
    if field == "forbidden":
        gates = (
            "  - id: g\n    type: command\n    runs: [pre_tool_use.command]\n"
            '    enforcement: required\n    forbidden: ["make a && make b"]\n    message: m\n'
        )
    else:
        gates = (
            "  - id: g\n    type: command_if_changed\n    runs: [stop.session]\n"
            '    enforcement: required\n    when_changed: ["x.json"]\n'
            '    require: ["make a && make b"]\n    message: m\n'
        )
    findings = _findings(gates)
    assert len(findings) == 1
    severity, _gate_id, message = findings[0]
    assert severity == "error"
    assert f"{field} phrase 'make a && make b' contains the shell separator(s) '&&'" in message


@pytest.mark.parametrize("separator", ["|", ";", "||", "&"])
def test_every_separator_is_caught_not_just_the_obvious_one(separator: str) -> None:
    assert unmatchable_phrase(f"cat x {separator} grep y", field="forbidden") is not None


@pytest.mark.parametrize("phrase", ["npm install;", "make a&&make b", "(npm install)", "make a\nmake b"])
def test_a_separator_typed_without_spaces_is_caught_too(phrase: str) -> None:
    """The spelling people actually type, and the one plain `shlex.split` hides.

    Matching tokenization leaves a separator glued to its word (`install;`), so
    looking for a separator token there finds nothing while the phrase still
    matches no command. Detection uses the command-side tokenizer instead.
    """
    assert unmatchable_phrase(phrase, field="forbidden") is not None


def test_an_ordinary_phrase_is_not_flagged() -> None:
    for phrase in ("npm install", "git push --force-with-lease", "uv run pytest tests/unit/x.py"):
        assert unmatchable_phrase(phrase, field="forbidden") is None


def test_a_quoted_separator_is_left_alone() -> None:
    """It is one argument, not a boundary, and the phrase matches a real command."""
    assert unmatchable_phrase('echo "a && b"', field="forbidden") is None


@pytest.mark.parametrize(
    "phrase",
    [
        "npm install",
        "git push --force",
        'echo "a && b"',
        "uv run pytest tests/unit/x.py",
        "make a && make b",
        "npm install;",
        "make a&&make b",
        "(npm install)",
        "make a\nmake b",
    ],
)
def test_a_phrase_is_flagged_exactly_when_it_cannot_match_itself(phrase: str) -> None:
    """The invariant behind this check, asserted against the real matcher.

    A phrase that matches nothing must be reported, and a phrase that matches
    something must not be. Checked against the command it was copied from,
    which is the weakest command that could possibly match it, so anything
    failing here is a phrase no command can satisfy.
    """
    matches_itself = any(
        _contains_subsequence(segment, tokenize_phrase(phrase)) for segment in _command_segments(phrase)
    )
    flagged = unmatchable_phrase(phrase, field="forbidden") is not None
    assert flagged is not matches_itself


@pytest.mark.parametrize("glob", ["./CHANGELOG.md", "/CHANGELOG.md", "../escape.md"])
def test_a_glob_that_can_never_match_is_an_error(glob: str) -> None:
    findings = _findings(_PATH_GATE.format(forbidden=f'["{glob}"]'))
    assert len(findings) == 1
    assert findings[0][0] == "error"
    assert "can never match" in findings[0][2]


def test_a_backslash_glob_is_a_warning_not_an_error() -> None:
    """A POSIX filename may legally contain a backslash, so this is advice, not a proof."""
    findings = _findings(_PATH_GATE.format(forbidden=r'["web\\src\\x.ts"]'))
    assert len(findings) == 1
    assert findings[0][0] == "warning"
    assert "contains a backslash" in findings[0][2]


@pytest.mark.parametrize("companion", ["*.md", "CLAUDE.*", "CLAUDE.md"])
def test_the_twin_warning_respects_coverage_by_another_glob(companion: str) -> None:
    """Another glob already reaching that depth means there is nothing to advise."""
    assert _findings(_PATH_GATE.format(forbidden=f'["{companion}", "**/CLAUDE.md"]')) == ()


def test_the_twin_warning_still_fires_when_nothing_covers_that_depth() -> None:
    findings = _findings(_PATH_GATE.format(forbidden='["docs/*.md", "**/CLAUDE.md"]'))
    assert len(findings) == 1
    assert "can never match 'CLAUDE.md', and no other forbidden glob covers it either" in findings[0][2]


def _verifier_gates(count: int) -> str:
    return "".join(
        f"  - id: v{index}\n    type: verifier\n    runs: [stop.verifier]\n"
        "    enforcement: required\n    verifier: v.sh\n    message: m\n"
        for index in range(count)
    )


def test_verifier_gates_past_their_own_cap_are_warned_about_too() -> None:
    """The judge cap has a sibling; warning about one and not the other reads as though it has none."""
    findings = _findings(_verifier_gates(4), verifier_gate_limit=3)
    assert len(findings) == 1
    severity, gate_id, message = findings[0]
    assert (severity, gate_id) == ("warning", None)
    assert "4 verifier gates, over the 3" in message
    assert "these are skipped: v3" in message


def test_verifier_gates_at_their_cap_do_not_warn() -> None:
    assert _findings(_verifier_gates(3), verifier_gate_limit=3) == ()


def test_the_separator_repair_differs_by_field() -> None:
    """Splitting a `forbidden` list blocks more; splitting a `require` list requires less.

    `require` entries are alternatives, so `["make a", "make b"]` is satisfied
    once `make a` has run, which is the opposite of what an author who wrote
    `make a && make b` meant. This repo's own guardrail spells the right answer
    out beside `openapi-changed-needs-generator`: one gate per required command.
    """
    forbidden = unmatchable_phrase("make a && make b", field="forbidden")
    require = unmatchable_phrase("make a && make b", field="require")
    assert forbidden is not None and require is not None
    assert "Split it into one phrase per command" in forbidden
    assert "Give each required command a gate of its own" in require
    assert "Do not split it across `require` entries" in require


def test_an_unmatchable_phrase_does_not_also_draw_the_single_token_warning() -> None:
    """Two findings of which one is false is worse than one that is true.

    `npm;` is a single token *and* unmatchable, but "matches that word anywhere
    in a command" is the opposite of what it does: it matches nothing.
    """
    findings = _findings(
        "  - id: g\n    type: command\n    runs: [pre_tool_use.command]\n"
        '    enforcement: required\n    forbidden: ["npm;"]\n    message: m\n'
    )
    assert [severity for severity, _gate, _message in findings] == ["error"]
    assert "single token" not in findings[0][2]


def _codes(gates: str, **options: Any) -> list[WarningCode | None]:
    return [finding.code for finding in _validate(gates, **options)]


# A glob field of each gate type that has one, holding `{glob}`, with every other field clean.
_GLOB_GATE_FIELDS: dict[GateType, str] = {
    "command_if_changed": (
        '    runs: [stop.session]\n    enforcement: required\n    require: ["make x"]\n    when_changed: [{glob}]\n'
    ),
    "judge": "    runs: [stop.session]\n    enforcement: advisory\n    rubric: r\n    when_changed: [{glob}]\n",
    "path": "    runs: [stop.working_tree]\n    enforcement: required\n    forbidden: [{glob}]\n",
    "verifier": (
        "    runs: [stop.verifier]\n    enforcement: required\n    verifier: v.sh\n    when_changed: [{glob}]\n"
    ),
}

_PROVOKING_GLOBS = {
    WarningCode.BACKSLASH_IN_GLOB: '"docs\\\\x.md"',
    WarningCode.GLOB_MISSES_SHALLOWER_DEPTH: '"**/CLAUDE.md"',
}

_PROVOKING_POLICIES: dict[WarningCode, tuple[str, dict[str, int]]] = {
    WarningCode.JUDGE_GATE_CAP: (_judge_gates(2), {"judge_gate_limit": 1}),
    WarningCode.SHELL_READ_UNSEEN: (
        "  - id: g\n    type: path\n    runs: [pre_tool_use.read_target]\n"
        '    enforcement: required\n    forbidden: [".env"]\n',
        {},
    ),
    WarningCode.SHELL_WRITE_UNSEEN: (
        "  - id: g\n    type: path\n    runs: [pre_tool_use.edit_target]\n"
        '    enforcement: required\n    forbidden: ["CHANGELOG.md"]\n',
        {},
    ),
    WarningCode.SINGLE_TOKEN_PHRASE: (
        "  - id: g\n    type: command\n    runs: [pre_tool_use.command]\n"
        '    enforcement: required\n    forbidden: ["npm"]\n',
        {},
    ),
    WarningCode.VERIFIER_GATE_CAP: (_verifier_gates(2), {"verifier_gate_limit": 1}),
}


@pytest.mark.parametrize(
    ("code", "gate_type"),
    [(code, gate_type) for code in WarningCode for gate_type in sorted(code.gate_types)],
)
def test_every_gate_type_a_code_names_draws_that_code(code: WarningCode, gate_type: GateType) -> None:
    """A gate accepts a warning by its code, so each code must arrive on every gate type it names."""
    if code in _PROVOKING_GLOBS:
        gates = f"  - id: g\n    type: {gate_type}\n" + _GLOB_GATE_FIELDS[gate_type].format(glob=_PROVOKING_GLOBS[code])
        limits: dict[str, int] = {}
    else:
        gates, limits = _PROVOKING_POLICIES[code]
    assert _codes(gates, **limits) == [code]


def test_gate_types_are_the_type_of_each_gate_dataclass() -> None:
    """A type missing from either side would parse as unsupported or be named by no warning code."""
    declared = {
        field.default for gate_class in get_args(GateSpec) for field in fields(gate_class) if field.name == "type"
    }
    assert declared == set(GATE_TYPES)


def test_an_error_carries_no_code() -> None:
    """Only a warning can be accepted, so only a warning is named by a code."""
    assert _codes(_PATH_GATE.format(forbidden='["/CHANGELOG.md"]')) == [None]


def test_a_warning_finding_needs_a_code() -> None:
    with pytest.raises(ValueError, match="A warning finding needs a code"):
        Finding("warning", "g", "m")


def test_an_error_finding_cannot_have_a_code() -> None:
    with pytest.raises(ValueError, match="An error finding cannot have the code 'single-token-phrase'"):
        Finding("error", "g", "m", code=WarningCode.SINGLE_TOKEN_PHRASE)


def _accepted(gates: str, **options: Any) -> list[tuple[WarningCode | None, bool]]:
    return [(finding.code, finding.accepted) for finding in _validate(gates, **options)]


_EDIT_ONLY_GATE = (
    "  - id: g\n    type: path\n    runs: [pre_tool_use.edit_target]\n"
    '    enforcement: required\n    forbidden: ["**/CLAUDE.md"]\n'
    "    accept_warnings: [{accepted}]\n"
)


def test_a_gate_accepts_the_warning_it_names_and_no_other() -> None:
    assert _accepted(_EDIT_ONLY_GATE.format(accepted="shell-write-unseen")) == [
        (WarningCode.GLOB_MISSES_SHALLOWER_DEPTH, False),
        (WarningCode.SHELL_WRITE_UNSEEN, True),
    ]


def _judge_gates_accepting(count: int, accepting: set[int]) -> str:
    return "".join(
        f"  - id: j{index}\n    type: judge\n    runs: [stop.session]\n"
        "    enforcement: advisory\n    rubric: r\n"
        + ("    accept_warnings: [judge-gate-cap]\n" if index in accepting else "")
        for index in range(count)
    )


def test_the_cap_warning_is_accepted_when_every_gate_past_the_cap_accepts_it() -> None:
    """A gate inside the cap always runs, so only the skipped ones have anything to accept."""
    assert _accepted(_judge_gates_accepting(4, {2, 3}), judge_gate_limit=2) == [(WarningCode.JUDGE_GATE_CAP, True)]


def test_the_cap_warning_names_the_skipped_gates_that_do_not_accept_it() -> None:
    (finding,) = _validate(_judge_gates_accepting(4, {2}), judge_gate_limit=2)
    assert not finding.accepted
    assert finding.message.endswith("Of those, j3 does not accept judge-gate-cap.")


def test_the_verifier_cap_warning_is_accepted_the_same_way() -> None:
    gates = "".join(
        f"  - id: v{index}\n    type: verifier\n    runs: [stop.verifier]\n"
        "    enforcement: required\n    verifier: v.sh\n    accept_warnings: [verifier-gate-cap]\n"
        for index in range(3)
    )
    assert _accepted(gates, verifier_gate_limit=2) == [(WarningCode.VERIFIER_GATE_CAP, True)]


def test_only_a_warning_can_be_accepted() -> None:
    with pytest.raises(ValueError, match="error finding cannot be accepted"):
        Finding("error", "g", "m", accepted=True)
