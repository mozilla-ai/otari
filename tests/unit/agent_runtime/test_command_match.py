import time

from gateway.agent_runtime.domain.evaluators import (
    _command_segments,
    _contains_subsequence,
    _strip_shell_comment,
    evaluate_command_match,
    tokenize_commands,
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


def test_not_applicable_when_evidence_is_an_explicit_empty_list() -> None:
    """An empty commands list is what a PreToolUse edit call or a Stop event

    submits: evidence was collected and there is no command to report, not
    "checked, none forbidden". A required gate must not read this as a clean
    pass, since nothing was actually checked.
    """
    result = evaluate_command_match(_gate(), CommandEvidence(commands=()))
    assert result.outcome is Outcome.NOT_APPLICABLE
    assert not result.outcome.is_blocking


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


def test_unbalanced_quotes_do_not_crash_and_still_match_bare_words() -> None:
    """A command shlex cannot tokenize falls back to a whitespace split.

    The old fallback kept it as one opaque token, which could never equal a
    forbidden phrase: `npm install "unterminated` silently passed a required
    gate forbidding `npm`. A whitespace split still finds a phrase spelled as
    bare words, and a token that carries the stray quote (`"--force`) still
    does not equal the phrase's own `--force`.
    """
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    assert evaluate_command_match(gate, CommandEvidence(commands=('npm install "unterminated',))).outcome is (
        Outcome.FAIL
    )

    force_gate = _gate(forbidden=("git push --force",))
    assert evaluate_command_match(force_gate, CommandEvidence(commands=('git push "--force',))).outcome is (
        Outcome.PASS
    )


def test_an_unparseable_heredoc_does_not_block_an_unrelated_command() -> None:
    """The reason the fallback is a whitespace split rather than `unknown`.

    A Bash call carrying a heredoc of another language is routinely
    unparseable to shlex. Reporting `unknown` there blocks a required gate on
    every such call, which is every ordinary session, not the forbidden ones.
    """
    gate = _gate(id="use-pnpm", forbidden=("npm install",), message="Use pnpm, not npm.")
    heredoc = "python3 - <<'EOF'\ns = \"it's fine\"\nprint(s)\nEOF"
    assert evaluate_command_match(gate, CommandEvidence(commands=(heredoc,))).outcome is Outcome.PASS


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


def test_command_segments_falls_back_to_a_whitespace_split_on_unbalanced_quotes() -> None:
    assert _command_segments('git push "--force') == [["git", "push", '"--force']]


def test_contains_subsequence_finds_the_phrase_anywhere() -> None:
    assert _contains_subsequence(["sudo", "npm", "install"], ["npm"]) is True
    assert _contains_subsequence(["npm", "install"], ["npm", "run"]) is False


def test_contains_subsequence_requires_contiguous_order() -> None:
    assert _contains_subsequence(["git", "push", "--force"], ["push", "git"]) is False


def test_contains_subsequence_rejects_an_empty_phrase() -> None:
    assert _contains_subsequence(["npm", "install"], []) is False


# --- Shell comments: word-boundary-aware, not shlex's own comments=True ----
#
# shlex.split(..., comments=True) treats *any* '#' as starting a comment,
# even mid-word ("echo a#b" -> ["echo", "a"], where real bash prints "a#b"
# unchanged). That is unsafe here: a mid-word '#' (a URL fragment, a
# --flag=value#123) would silently swallow everything after it, including a
# genuinely separate, later command joined by &&/;/|. _strip_shell_comment
# only treats a '#' as a comment at a real POSIX word boundary.


def test_strip_shell_comment_cases() -> None:
    cases = [
        ("npm install # don't use yarn", "npm install "),
        ('npm install # a comment with a stray " quote', "npm install "),
        ('git commit -m "fix #123"', 'git commit -m "fix #123"'),
        ("curl https://x.com/#frag && git push --force", "curl https://x.com/#frag && git push --force"),
        ("echo a#b", "echo a#b"),
        ("echo a #b", "echo a "),
        ("#just a comment", ""),
        ("git commit -m 'has a # inside single quotes'", "git commit -m 'has a # inside single quotes'"),
        # A comment ends at its own line, not at the end of the string: an
        # earlier comment must not swallow a later, real command.
        ("# install dependencies\nnpm install", "\nnpm install"),
        ("echo ready # setup\nnpm install", "echo ready \nnpm install"),
        # A backslash-escaped double quote does not close the quote it is
        # inside, so the '#' right after it is still inside quotes, not a
        # comment start; nothing after it is discarded.
        ('echo "a\\" # b" && npm install', 'echo "a\\" # b" && npm install'),
        # Bash ANSI-C quoting ($'...'): backslash escapes are active inside
        # it, so \' is a literal apostrophe, not the closing quote; the '#'
        # right after it is still inside the string, not a comment start.
        ("echo $'a\\' # b' && npm install", "echo 'a'\\'' # b' && npm install"),
    ]
    for command, expected in cases:
        assert _strip_shell_comment(command) == expected, command


def test_a_comment_on_an_earlier_line_does_not_swallow_a_later_real_command() -> None:
    """A multi-line command's own comment only extends to its own line.

    A prior version of _strip_shell_comment truncated everything from the
    first comment to the end of the whole string, so a leading comment line
    silently discarded every real command after it.
    """
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    for command in [
        "# install dependencies\nnpm install",
        "echo ready # setup\nnpm install",
    ]:
        result = evaluate_command_match(gate, CommandEvidence(commands=(command,)))
        assert result.outcome is Outcome.FAIL, command


def test_an_escaped_quote_does_not_end_double_quoting_early() -> None:
    r"""'echo "a\" # b" && npm install' is one double-quoted argument

    (`a" # b`, via the escaped quote) followed by a real, separate `&&
    npm install`. A prior version treated the backslash-escaped `"` as the
    real closing quote, so the `#` right after it looked unquoted and
    word-start, discarding "&& npm install" as a bogus comment.
    """
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=('echo "a\\" # b" && npm install',)))
    assert result.outcome is Outcome.FAIL


def test_ansi_c_quoted_escaped_apostrophe_does_not_evade_the_gate() -> None:
    r"""echo $'a\' # b' && npm install is one ANSI-C-quoted argument

    (`a' # b`, via the escaped apostrophe) followed by a real, separate `&&
    npm install`. Bash keeps this apostrophe as literal content because
    backslash escapes are active inside $'...', unlike a plain '...' quote.
    A prior version treated the escaped apostrophe as the real closing
    quote, so the '#' right after it looked unquoted and word-start,
    discarding "&& npm install" as a bogus comment.
    """
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=("echo $'a\\' # b' && npm install",)))
    assert result.outcome is Outcome.FAIL


# --- Empty segments: two separators back to back, or at either end --------


def test_empty_segments_are_dropped_not_matched_or_iterated() -> None:
    # Space-separated, so shlex sees three distinct ";" tokens rather than
    # gluing them into one literal "; ; ;"-shaped token (the already-known,
    # separately documented gap for operators with no surrounding whitespace).
    assert _command_segments("; ; ;") == []
    assert _command_segments("npm install ; ; git status") == [["npm", "install"], ["git", "status"]]


def test_many_separators_with_no_real_content_resolve_quickly() -> None:
    """A command built from many separators and no real tokens (";" * n) used

    to keep an empty segment per separator gap, each compared against every
    forbidden phrase for a guaranteed-False result: 500 forbidden phrases
    against 100 such commands (~50,000 empty segments, ~25,000,000
    comparisons) measured ~1.1s. Dropping empty segments at the source
    collapses this, since there is nothing left to iterate.
    """
    gate = _gate(forbidden=tuple(f"p{i}" for i in range(500)))
    commands = tuple("; " * 500 + " " * i for i in range(100))
    evidence = CommandEvidence(commands=commands)
    start = time.time()
    result = evaluate_command_match(gate, evidence, segment_cache=tokenize_commands(commands))
    elapsed = time.time() - start
    assert elapsed < 0.5
    assert result.outcome is Outcome.PASS


def test_apostrophe_in_a_trailing_comment_no_longer_evades_the_gate() -> None:
    """A '#'-comment with an apostrophe used to raise ValueError from shlex

    (comments=False, the default, does not strip it, and the apostrophe is
    an unbalanced quote), falling back to one opaque token that could never
    match a real forbidden phrase: 'npm install # don't use yarn' silently
    passed a required gate forbidding 'npm'.
    """
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=("npm install # don't use yarn",)))
    assert result.outcome is Outcome.FAIL


def test_unmatched_quote_inside_a_comment_does_not_crash_or_evade() -> None:
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(
        gate, CommandEvidence(commands=('npm install # a comment with a stray " quote',))
    )
    assert result.outcome is Outcome.FAIL


def test_quoted_literal_hash_is_preserved_not_treated_as_a_comment() -> None:
    """The quoted message is one token, "fix #123 with npm", not stripped at

    the '#' and not split into separate words: it is not itself an npm
    invocation, so this must not match a phrase forbidding the standalone
    word "npm", the same way a real shell keeps a quoted argument intact.
    """
    gate = _gate(id="use-pnpm", forbidden=("npm",), message="Use pnpm, not npm.")
    result = evaluate_command_match(gate, CommandEvidence(commands=('git commit -m "fix #123 with npm"',)))
    assert result.outcome is Outcome.PASS


def test_mid_word_hash_does_not_swallow_a_later_forbidden_command() -> None:
    """A URL fragment ('#frag') is a mid-word '#', not a comment start in

    real bash; everything after it, including a chained && command, must
    still be visible to this gate.
    """
    gate = _gate(forbidden=("git push --force",))
    result = evaluate_command_match(
        gate, CommandEvidence(commands=("curl https://x.com/page#frag && git push --force",))
    )
    assert result.outcome is Outcome.FAIL


# --- Tokenizing once per request, shared across command_match gates --------


def test_segment_cache_produces_the_same_result_as_computing_internally() -> None:
    gate = _gate(forbidden=("git push --force",))
    evidence = CommandEvidence(commands=("git push --force", "git status"))
    without_cache = evaluate_command_match(gate, evidence)
    with_cache = evaluate_command_match(gate, evidence, segment_cache=tokenize_commands(evidence.commands))
    assert without_cache == with_cache


def test_shared_segment_cache_avoids_retokenizing_per_gate() -> None:
    """Review's P1 repro: 100 command_match gates forbidding "npm" against

    250 distinct, mostly-whitespace ~4,000-character commands passed every
    request-level budget (low token content, few phrases) yet measured ~7s,
    because each of the 100 gates independently re-tokenized every command
    from scratch. Tokenizing once and sharing the result, as
    tests/integration/test_hooks_route.py's route-level counterpart
    exercises through the real endpoint, collapses this to a fraction of a
    second.
    """
    gates = [_gate(id=f"g{i}", forbidden=("npm",)) for i in range(100)]
    commands = tuple(" " * 4000 + str(i) for i in range(250))
    evidence = CommandEvidence(commands=commands)
    cache = tokenize_commands(commands)
    start = time.time()
    for gate in gates:
        evaluate_command_match(gate, evidence, segment_cache=cache)
    elapsed = time.time() - start
    assert elapsed < 1.0
