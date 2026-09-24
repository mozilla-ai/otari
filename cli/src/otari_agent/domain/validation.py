"""Static checks over a parsed guardrail, the ones running it would teach the hard way.

Pure like every other module in this package: what a check *is* lives here,
and a caller supplies anything needing a filesystem (``probe_verifier``).

Everything here is deliberately *beyond* what ``domain/policy.py`` rejects. A
parse error is a policy that cannot run; a finding here is a policy that runs
and does not mean what its author intended, which is the worse failure of the
two, because a gate that silently never matches is indistinguishable from a
clean result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from otari_agent.domain.evaluators import (
    COMMAND_SEPARATORS,
    matched_changed_paths,
    tokenize_phrase,
    tokenize_phrase_with_separators,
)
from otari_agent.domain.types import (
    CommandGate,
    CommandIfChangedGate,
    GateSpec,
    JudgeGate,
    PathGate,
    PolicySpec,
    VerifierGate,
)

# An error is a gate that provably cannot do its job: its verifier is
# missing, or one of its patterns can never match anything, whatever the
# session does. A warning is a gate that runs and may not mean what it says.
# Warnings never block on their own, because each has a legitimate exception:
# a `**` glob whose shallower depth genuinely should not match, a one-token
# phrase where refusing the tool by name is the whole intent.
Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class Finding:
    """One problem found in a guardrail without running it.

    ``gate_id`` is ``None`` for a finding about the guardrail as a whole
    rather than any one gate.
    """

    severity: Severity
    gate_id: str | None
    message: str


def _glob_fields(gate: GateSpec) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Every field of this gate holding repo-relative globs, with its own name."""
    if isinstance(gate, PathGate):
        yield "forbidden", gate.forbidden
    elif isinstance(gate, CommandIfChangedGate | JudgeGate | VerifierGate):
        yield "when_changed", gate.when_changed


def _phrase_fields(gate: GateSpec) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Every field of this gate holding shell phrases, with its own name."""
    if isinstance(gate, CommandGate):
        yield "forbidden", gate.forbidden
    elif isinstance(gate, CommandIfChangedGate):
        yield "require", gate.require


def shallower_twin(glob: str) -> str | None:
    """The path depth this glob's ``**`` puts out of its own reach, spelled as a glob.

    ``**`` must consume at least one path segment (``_segments_match`` in
    domain/evaluators.py), so ``**/CLAUDE.md`` reaches ``web/CLAUDE.md`` and
    never the root file, and ``src/**/x.py`` reaches ``src/a/x.py`` and never
    ``src/x.py``. The twin is the same glob with the ``**`` segment removed.

    ``None`` when there is nothing to miss: no ``**`` at all, or a trailing
    one, where the twin would name a bare directory and Git never reports one
    as a changed path.
    """
    segments = glob.split("/")
    if "**" not in segments:
        return None
    index = segments.index("**")
    if index == len(segments) - 1:
        return None
    return "/".join(segments[:index] + segments[index + 1 :])


# A repo-relative path from Git never carries an empty, `.` or `..` segment,
# so a glob that does is not a narrow pattern: it is one that cannot match
# anything the caller will ever submit. Covers a leading `/` and a leading
# `./` alike, both of which split to exactly such a segment.
_UNREACHABLE_SEGMENTS = frozenset({"", ".", ".."})


def unreachable_glob(glob: str) -> str | None:
    """Why this glob can never match any submitted path, or ``None`` if it can.

    Distinct from :func:`shallower_twin`, which reports a glob that matches
    some paths and misses others. This one reports a glob that matches
    nothing at all, which is an error rather than advice.
    """
    for segment in glob.split("/"):
        if segment in _UNREACHABLE_SEGMENTS:
            spelled = "an empty" if segment == "" else f"a {segment!r}"
            return (
                f"glob {glob!r} has {spelled} path segment, and a repo-relative path never "
                "does, so this pattern can never match. Spell it relative to the repo root "
                "with no leading '/' or './'."
            )
    return None


# What to do instead, per field, because the two lists mean opposite things
# even though both are an OR. Splitting a `forbidden` list is the fix: any one
# entry matching blocks the call, so two entries block strictly more than one.
# Splitting a `require` list is the opposite of the fix: any one entry
# satisfies the gate, so `["make a", "make b"]` passes once `make a` has run,
# and an author who wrote `make a && make b` wanted both. One gate per required
# command is how this repo's own guardrail spells that (see the
# `openapi-changed-needs-generator` comment in .otari-guardrails.yml).
_SPLIT_ADVICE = {
    "forbidden": "Split it into one phrase per command; any one of them matching refuses the call.",
    "require": (
        "Give each required command a gate of its own. Do not split it across `require` entries: "
        "any one entry satisfies a gate, so the split list would pass as soon as the first command ran."
    ),
}


def unmatchable_phrase(phrase: str, *, field: str) -> str | None:
    """Why this shell phrase can never match any submitted command, or ``None``.

    A phrase matches a contiguous token run *within one* separator-delimited
    segment of a command, so a phrase carrying a separator of its own spans a
    boundary no single segment has. It never matches, not even the command it
    was copied from. On a `command` gate that is silent non-enforcement; on
    `command_if_changed`'s ``require`` it is worse, because the gate then
    cannot be satisfied by anything and a required one blocks every turn that
    touches a matching path.

    ``field`` picks the repair, which is not the same on both: see
    ``_SPLIT_ADVICE``.
    """
    # Command-style tokenization, not the matching one: a separator typed
    # without spaces (`npm install;`) stays glued to its word otherwise, so
    # nothing here equals a separator and the phrase that can never match
    # reads as clean. See tokenize_phrase_with_separators.
    tokens = tokenize_phrase_with_separators(phrase)
    offenders = sorted({token for token in tokens if token in COMMAND_SEPARATORS})
    if not offenders:
        return None
    return (
        f"phrase {phrase!r} contains the shell separator(s) {', '.join(repr(o) for o in offenders)}, "
        "and a phrase is matched within one separator-delimited segment of a command, so this can "
        f"never match any command at all. {_SPLIT_ADVICE[field]}"
    )


def validate_policy(
    spec: PolicySpec,
    *,
    judge_gate_limit: int,
    verifier_gate_limit: int,
    probe_verifier: Callable[[str], str | None] | None = None,
) -> tuple[Finding, ...]:
    """Check a parsed guardrail for gates that run without meaning what they say.

    ``probe_verifier`` is given a verifier gate's repo-relative path and
    returns a problem to report, or ``None`` when the script is fine. Left
    unset, verifier gates are not checked at all, which is what keeps this
    module free of the filesystem access such a check needs.

    ``judge_gate_limit`` and ``verifier_gate_limit`` are the caller's own
    per-Stop caps on the two gate types that run something
    (``otari hook``'s ``_HOOK_JUDGE_MAX_GATES_PER_RUN`` and
    ``_HOOK_CHECK_MAX_GATES_PER_RUN``), passed in rather than duplicated here
    so the number an author is warned about is the number that will actually
    be enforced. Both are needed: warning about one cap and not the other
    would read as though the other has none.

    Findings come back in declaration order, so a reader walks them beside the
    file rather than jumping around it.
    """
    findings: list[Finding] = []

    for gate in spec.gates:
        for field, globs in _glob_fields(gate):
            for glob in globs:
                dead = unreachable_glob(glob)
                if dead is not None:
                    findings.append(Finding("error", gate.id, f"{field} {dead}"))
                    continue
                if "\\" in glob:
                    findings.append(
                        Finding(
                            "warning",
                            gate.id,
                            f"{field} glob {glob!r} contains a backslash, and a glob is matched "
                            "against repo-relative POSIX paths split on '/'. If that was meant as "
                            "a path separator, spell it with '/'.",
                        )
                    )
                twin = shallower_twin(glob)
                # Coverage, not string identity: `["*.md", "**/CLAUDE.md"]`
                # already reaches the root file through `*.md`, so demanding a
                # literal twin there would advise adding a redundant entry.
                if twin is not None and not matched_changed_paths(globs, (twin,)):
                    findings.append(
                        Finding(
                            "warning",
                            gate.id,
                            f"{field} glob {glob!r} can never match {twin!r}, and no other {field} "
                            "glob covers it either: '**' must consume at least one path segment. "
                            f"Add {twin!r} beside it to cover that depth too.",
                        )
                    )

        # Both phrase-carrying fields, because a phrase that can never match
        # is an error either way, and on `require` it is the worse of the two.
        for field, phrases in _phrase_fields(gate):
            findings.extend(
                Finding("error", gate.id, f"{field} {problem}")
                for phrase in phrases
                if (problem := unmatchable_phrase(phrase, field=field)) is not None
            )

        if isinstance(gate, CommandGate):
            # Single-token is checked on `forbidden` only. The same phrase in
            # `command_if_changed`'s `require` over-matches identically, but
            # there it is usually the point: an author requiring a test file
            # has no way to know whether the session will spell it `pytest x`,
            # `uv run pytest x` or `make test-unit x`, and the bare path is
            # what covers all three. Over-matching a `forbidden` phrase
            # refuses real work; over-matching a `require` phrase only accepts
            # a session sooner.
            findings.extend(
                Finding(
                    "warning",
                    gate.id,
                    f"forbidden phrase {phrase!r} is a single token, so it matches that word "
                    "anywhere in a command, including one that only mentions it "
                    f"(`grep -rn {phrase} .`). Prefer a phrase naming a real invocation.",
                )
                for phrase in gate.forbidden
                # A phrase that can never match cannot over-match either. The
                # error above already says it matches nothing, and "it matches
                # that word anywhere in a command" beside it would be two
                # findings of which one is false.
                if unmatchable_phrase(phrase, field="forbidden") is None and len(tokenize_phrase(phrase)) == 1
            )

        if isinstance(gate, PathGate) and tuple(gate.runs) == ("pre_tool_use.edit_target",):
            findings.append(
                Finding(
                    "warning",
                    gate.id,
                    "runs only at pre_tool_use.edit_target, which sees the path an edit tool declares "
                    "and nothing a shell command writes (a redirect, `sed -i`, a heredoc, `cp`, a "
                    "script). Add stop.working_tree for a backstop over the finished tree.",
                )
            )

        if isinstance(gate, VerifierGate) and probe_verifier is not None:
            problem = probe_verifier(gate.verifier)
            if problem is not None:
                findings.append(Finding("error", gate.id, problem))

    # Both gate types that run something carry a per-Stop cap, and both caps
    # apply after `when_changed` filtering, so this counts the worst case: a
    # session where every one of them applies at once. Said that way rather
    # than flatly, because a well-scoped policy may never reach either.
    for label, limit, ids in (
        ("judge", judge_gate_limit, [g.id for g in spec.gates if isinstance(g, JudgeGate)]),
        ("verifier", verifier_gate_limit, [g.id for g in spec.gates if isinstance(g, VerifierGate)]),
    ):
        if len(ids) > limit:
            findings.append(
                Finding(
                    "warning",
                    None,
                    f"{len(ids)} {label} gates, over the {limit} one Stop event evaluates. On a "
                    f"session where every one applies, these are skipped in declaration order: "
                    f"{', '.join(ids[limit:])}. Scope them with when_changed so fewer apply at once.",
                )
            )

    return tuple(findings)
