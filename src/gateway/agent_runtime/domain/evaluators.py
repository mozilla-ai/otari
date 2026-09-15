"""Pure gate evaluators. No filesystem, git, network, or clock access.

Each evaluator takes a gate spec and evidence already supplied by the caller
and returns a ``GateResult``. Evidence the caller could not supply is ``None``,
never an empty collection standing in for "nothing changed": an evaluator
cannot tell those apart, and conflating them would let missing evidence read
as a clean result.
"""

from __future__ import annotations

import re
from functools import lru_cache

from gateway.agent_runtime.domain.types import (
    ChangedPathEvidence,
    ChangedPathGate,
    GateResult,
    Outcome,
)


@lru_cache(maxsize=4096)
def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a repo-relative POSIX glob to an anchored regex.

    ``*`` matches within one path segment; ``**`` crosses segment boundaries
    and requires at least one character, so a ``dir/**`` pattern matches
    files under ``dir/`` without also matching ``dir`` itself (a directory
    is never, on its own, a changed path in Git's output).

    Cached: without it, matching evaluates ``len(forbidden) * len(changed_paths)``
    times per gate, recompiling the same pattern for every path checked against
    it. A caller-submitted policy and evidence list are each bounded, but not
    small enough for that product to stay cheap uncached (500 forbidden globs
    against 5,000 changed paths measures over a second of blocking CPU time on
    the event loop without this cache).
    """
    parts = [r".+" if segment == "**" else re.escape(segment).replace(r"\*", "[^/]*") for segment in pattern.split("/")]
    return re.compile("^" + "/".join(parts) + "$")


def _matches_any(path: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        if _glob_to_regex(pattern).match(path):
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
