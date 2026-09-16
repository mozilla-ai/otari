"""Policy, evidence, and result types shared by every gate evaluator.

Kept dependency-free (stdlib only) so the Hook Server route and a future
native ``otari hook`` dispatcher can share the exact same types without
either pulling in YAML parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Enforcement = Literal["required", "advisory"]

# Gate results that mean "no objection". Every other outcome blocks a required
# gate: unknown and error are deliberately on the blocking side, not the
# passing one, so a check that could not run is never mistaken for one that
# passed. See docs/agent-gates-production-plan.md#5-evidence-and-decision-semantics.
_NON_BLOCKING = frozenset({"pass", "not_applicable"})


class Outcome(str, Enum):
    """A single gate's result. Values are the wire/JSON spelling."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"
    NOT_RUN = "not_run"

    @property
    def is_blocking(self) -> bool:
        """Whether this outcome, on a required gate, keeps it from certifying pass.

        Independent of any one gate's actual enforcement: an advisory gate
        with a blocking outcome still only warns. Callers combine this with
        the gate's ``enforcement`` to decide whether to block.
        """
        return self.value not in _NON_BLOCKING


@dataclass(frozen=True, slots=True)
class ChangedPathGate:
    """A gate that fails when a caller-submitted path matches a forbidden glob.

    The caller decides what "changed" means for the evidence it submits: a
    Git diff after a turn, or a single tool call's target path before it
    runs. ``forbidden`` entries are repo-relative POSIX globs: ``*`` matches
    within one path segment, ``**`` crosses segment boundaries. This is the
    v0 glob grammar; the full symlink/monorepo/rename grammar is AG-005.
    """

    id: str
    enforcement: Enforcement
    forbidden: tuple[str, ...]
    message: str
    type: Literal["changed_path"] = "changed_path"


# The only gate type this first slice ships. Extend this alias (a Union, once
# there is a second member) as command_match, check_passed, and judge land;
# do not let a new gate type skip it, or the policy loader's dispatch on
# ``type`` silently stops covering it.
GateSpec = ChangedPathGate


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """A parsed, validated ``.otari-gates.yml``."""

    schema_version: str
    policy_id: str
    gates: tuple[GateSpec, ...]


@dataclass(frozen=True, slots=True)
class ChangedPathEvidence:
    """Repo-relative paths the caller reports as changed or about to change.

    Otari does not collect or verify this itself; see the module docstring.
    """

    changed_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's evaluated outcome."""

    gate_id: str
    enforcement: Enforcement
    outcome: Outcome
    message: str
    detail: str | None = None
