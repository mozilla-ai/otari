"""Pure gate evaluators. No filesystem, git, network, or clock access.

Each evaluator takes a gate spec and evidence already supplied by the caller
and returns a ``GateResult``. Evidence the caller could not supply is ``None``,
never an empty collection standing in for "nothing changed": an evaluator
cannot tell those apart, and conflating them would let missing evidence read
as a clean result.
"""

from __future__ import annotations

from gateway.agent_runtime.domain.types import (
    ChangedPathEvidence,
    ChangedPathGate,
    GateResult,
    Outcome,
)


def _segment_matches(pattern: str, text: str) -> bool:
    """Match `*` (zero or more characters) within one path segment against text.

    Splits on `*` into literal chunks and locates each one with `str.find`/
    `str.startswith`/`str.endswith`, never searching backward: CPython
    implements string search with the two-way algorithm, bounded at
    O(len(haystack) + len(needle)), so one call here costs
    O(len(pattern) + len(text)) regardless of how many `*` the pattern has
    or how close `text` comes to matching without succeeding. A prior
    two-pointer version of this function looked linear but wasn't: on a
    mismatch it rewound to just after the last `*` and rescanned the literal
    that followed from scratch, which is O(len(pattern) * len(text)) in the
    adversarial case (a pattern like `"*" + "a" * 2000 + "b"` against text
    with no trailing `b` measured in the seconds against a handful of paths).
    It also had a correctness bug in the same code path: it consumed a text
    character the instant it saw `*`, so `*a` failed to match the single
    character `a` (a false negative that would have let a forbidden path
    through undetected). Splitting into chunks up front avoids both: each
    chunk is found once, matching zero-or-more is just an empty chunk
    contributing nothing, and nothing is ever rescanned.
    """
    chunks = pattern.split("*")
    if len(chunks) == 1:
        return pattern == text

    first, *middle, last = chunks
    if first and not text.startswith(first):
        return False
    if last and not text.endswith(last):
        return False

    # The room left for the middle chunks once `first` and `last` are
    # reserved. If reserving both leaves no room (or a negative amount),
    # nothing arranges the remaining chunks, even all-empty ones, so a
    # length check up front avoids a `find` call in an invalid range.
    start, end = len(first), len(text) - len(last)
    if start > end:
        return False

    position = start
    for chunk in middle:
        if not chunk:
            continue
        found = text.find(chunk, position, end)
        if found == -1:
            return False
        position = found + len(chunk)
    return True


def _segments_match(pattern_segments: list[str], path_segments: list[str]) -> bool:
    """Match glob path segments against path segments; `**` crosses directories.

    Non-recursive: `domain/policy.py` caps a glob at one standalone `**`
    segment (`_MAX_DOUBLE_STAR_PER_GLOB`), so its position is exactly
    determined by `.index()`, and the segments before and after it match
    the corresponding number of segments at the start and end of the path
    directly, with no need to try more than one split point. Trying every
    split point (recursing once per possible position) is what a second
    `**` in the same glob would require, which is why the parser bounds it
    instead of this function trying to stay safe with more than one.
    """
    if "**" not in pattern_segments:
        return len(pattern_segments) == len(path_segments) and all(
            _segment_matches(p, s) for p, s in zip(pattern_segments, path_segments)
        )

    star_index = pattern_segments.index("**")
    before, after = pattern_segments[:star_index], pattern_segments[star_index + 1 :]
    # ** must consume at least one path segment: a bare directory is never,
    # on its own, a changed path in Git's output.
    if len(path_segments) < len(before) + len(after) + 1:
        return False
    head = path_segments[: len(before)]
    tail = path_segments[len(path_segments) - len(after) :] if after else []
    return all(_segment_matches(p, s) for p, s in zip(before, head)) and all(
        _segment_matches(p, s) for p, s in zip(after, tail)
    )


def _matches_any(path: str, patterns: tuple[str, ...]) -> str | None:
    path_segments = path.split("/")
    for pattern in patterns:
        if _segments_match(pattern.split("/"), path_segments):
            return pattern
    return None


def evaluate_changed_path(gate: ChangedPathGate, evidence: ChangedPathEvidence | None) -> GateResult:
    """Fail when a changed path matches one of the gate's forbidden globs."""
    if evidence is None:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.UNKNOWN,
            message="Change evidence was not submitted.",
        )

    matched = sorted(path for path in evidence.changed_paths if _matches_any(path, gate.forbidden) is not None)
    if matched:
        return GateResult(
            gate_id=gate.id,
            enforcement=gate.enforcement,
            outcome=Outcome.FAIL,
            message=gate.message,
            detail=", ".join(matched),
        )
    return GateResult(
        gate_id=gate.id,
        enforcement=gate.enforcement,
        outcome=Outcome.PASS,
        message="No forbidden paths changed.",
    )
