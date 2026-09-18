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

# What a submitted command list covers. See CommandEvidence.scope.
EvidenceScope = Literal["call", "session"]

# Gate results that mean "no objection". Every other outcome blocks a required
# gate: unknown and error are deliberately on the blocking side, not the
# passing one, so a check that could not run is never mistaken for one that
# passed. See docs/agent-gates.md.
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


@dataclass(frozen=True, slots=True)
class CommandMatchGate:
    """A gate that fails when a caller-submitted command matches a forbidden phrase.

    A ``forbidden`` entry is a shell phrase (``"git push --force"``,
    ``"npm"``); matching is token-based, not substring: the phrase's own
    tokens must appear as a contiguous run within one ``&&``/``;``/``|``/``||``
    -separated segment of the submitted command. Token-based matching is what
    keeps ``"npm"`` from matching inside ``"pnpm"``, and ``"--force"`` from
    matching inside the deliberately-safer ``"--force-with-lease"``; a plain
    substring check would get both wrong. A phrase matches a token run in any
    position, not only at the head, so a one-word phrase also matches where
    that word is an argument; prefer a phrase naming a real invocation
    (``"npm install"``) over a bare tool name. A separator needs no whitespace
    around it (``"npm install;"`` and ``"(npm install)"`` split the same as the
    spaced forms). See domain/evaluators.py for the tokenizer and its
    whitespace-split fallback for a command shlex cannot parse.

    This gate sees only the literal command text of one tool call; it does
    not, and cannot, see what a script or program that command invokes does
    internally. It is a footgun-catcher for a cooperative agent, not a
    sandbox against one deliberately working around it.
    """

    id: str
    enforcement: Enforcement
    forbidden: tuple[str, ...]
    message: str
    type: Literal["command_match"] = "command_match"


@dataclass(frozen=True, slots=True)
class CommandIfChangedGate:
    """A gate that fails when a changed path matches but no required command ran.

    ``when_changed`` is a tuple of repo-relative POSIX globs, the same
    grammar ``ChangedPathGate.forbidden`` uses. ``require`` is a tuple of
    shell phrases, the same grammar ``CommandMatchGate.forbidden`` uses,
    matched the same token-based way; any one of them satisfies the gate
    (an OR, same as a ``forbidden`` list matching any one entry). This is
    what expresses "if this changed, that must have run" (e.g. regenerating
    a committed artifact), which neither of the other two gate types can:
    each of those checks one independent condition, not a correlation
    between two.

    Meaningful mainly when both evidence lists reflect a whole session, not
    one tool call: on a ``Stop`` event, where ``otari hook`` now collects
    real command evidence from the session's own transcript, not on a
    single ``PreToolUse`` call.
    """

    id: str
    enforcement: Enforcement
    when_changed: tuple[str, ...]
    require: tuple[str, ...]
    message: str
    type: Literal["command_if_changed"] = "command_if_changed"


@dataclass(frozen=True, slots=True)
class JudgeGate:
    """A gate whose verdict comes from a model, not a mechanical match.

    ``rubric`` is free text describing what the model should check (e.g.
    "Does this change follow the repository's error-handling conventions?").
    Otari itself never calls a model: the caller (``otari hook``) reads
    ``rubric``, builds a prompt from it plus its own diff and transcript, runs
    its own model call, and submits the resulting verdict as
    :class:`JudgeEvidence`. This route only relays that verdict.

    ``enforcement`` is always ``"advisory"``; ``domain/policy.py`` rejects
    ``required`` at parse time, for two independent reasons, either alone
    sufficient. A model's verdict is not reproducible the way a glob or
    phrase match is. And the diff and transcript text a verdict is judged
    from are the same untrusted, attacker-influenceable content a
    prompt-injection attack already targets elsewhere in this codebase (see
    ``services/url_safety.py`` and the MCP tool loop): a crafted diff or
    transcript could talk a model into a ``pass`` it should not give, and
    nothing here can rule that out, since the caller's own prompt
    construction is outside what Otari can see or verify. Advisory
    enforcement is what keeps that from ever mattering: at worst, a
    compromised verdict suppresses a warning, never a block. See
    docs/agent-gates.md.

    ``when_changed`` is optional and, like ``CommandIfChangedGate``'s own
    field of the same name, the same repo-relative POSIX glob grammar
    ``ChangedPathGate.forbidden`` uses. Empty (the default) means this gate
    always applies, the only behavior a judge gate had before this field
    existed. Non-empty scopes the model call to a session that actually
    touched a matching path, so a rubric about, say, error-handling
    conventions is not re-judged, at real model-call cost, on a session that
    never touched application code.
    """

    id: str
    enforcement: Literal["advisory"]
    rubric: str
    message: str
    when_changed: tuple[str, ...] = ()
    type: Literal["judge"] = "judge"


# Extend this alias as check_passed lands; do not let a new gate type skip
# it, or the policy loader's dispatch on ``type`` silently stops covering it.
GateSpec = ChangedPathGate | CommandMatchGate | CommandIfChangedGate | JudgeGate


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
class CommandEvidence:
    """Shell commands the caller reports as run or about to run.

    Otari does not collect or verify this itself; see the module docstring.

    ``scope`` says what the list covers, which decides which gate types can
    resolve against it at all. ``"call"`` is one tool call about to run (a
    ``PreToolUse`` hook): complete for "is this command forbidden", useless
    for "did that command ever run". ``"session"`` is every command the
    session has run so far (a ``Stop`` hook reading its own transcript): the
    reverse. Without this, an evaluator has to guess from an empty list
    alone, which cannot tell "nothing to collect here" from "collected, and
    there was none".
    """

    commands: tuple[str, ...]
    scope: EvidenceScope = "call"


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One judge gate's model-produced verdict, as the caller observed it.

    ``outcome`` is the caller's own report, not a value Otari computed:
    ``"error"`` means the caller's model call itself failed or returned
    something it could not parse as a verdict (no `claude` on PATH, a
    timeout, malformed JSON), distinct from ``"fail"``, which means the model
    call succeeded and judged the rubric unmet. Otari does not verify either.
    """

    gate_id: str
    outcome: Literal["pass", "fail", "error"]
    reasoning: str


@dataclass(frozen=True, slots=True)
class JudgeEvidence:
    """Verdicts the caller collected for this request's judge gates.

    Unlike :class:`ChangedPathEvidence`/:class:`CommandEvidence`, a verdict is
    already keyed to the one gate it judged (each judge gate carries its own
    rubric, so the caller's model call is necessarily one call per gate, not
    one shared fact every gate matches independently). A judge gate whose id
    has no entry here resolves ``unknown`` the same way a gate resolves
    ``unknown`` against an entirely absent evidence kind: omitting this field
    and submitting it empty are therefore equivalent, and neither is treated
    as its own tri-state, unlike the other two evidence kinds.
    """

    verdicts: tuple[JudgeVerdict, ...]


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's evaluated outcome."""

    gate_id: str
    enforcement: Enforcement
    outcome: Outcome
    message: str
    detail: str | None = None
