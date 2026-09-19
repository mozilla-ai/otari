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

    ``judge_cli`` is optional and names which locally-installed CLI(s)
    ``otari hook`` may use to make the model call this gate needs, in
    preference order; the first one whose own binary is found on ``PATH``
    wins. ``None`` (the default, and the only behavior a judge gate had
    before this field existed) means no preference: the caller falls back to
    whichever CLI its own invoking harness implies (Claude Code's hook ->
    ``claude``, Codex's -> ``codex``). Naming one explicitly is what lets a
    gate authored for, say, a Codex-only fleet require ``codex`` even when
    invoked by a Claude Code hook, or list both so whichever is actually
    installed on a given machine is used. This field changes nothing about
    where the call happens: still entirely within ``otari hook``, never here
    (see this gate's own opening paragraph).
    """

    id: str
    enforcement: Literal["advisory"]
    rubric: str
    message: str
    when_changed: tuple[str, ...] = ()
    judge_cli: tuple[str, ...] | None = None
    type: Literal["judge"] = "judge"


@dataclass(frozen=True, slots=True)
class CheckPassedGate:
    """A gate whose verdict comes from a repo-local verifier script's own exit status.

    ``verifier`` is a repo-relative path to an executable script in the
    calling repo (e.g. ``.otari-gates/verifiers/no-conflict-markers.sh``), not
    a closed set of otari-shipped implementations. Otari itself never runs
    it, the same way it never reads a caller's repository for any other gate:
    the caller (``otari hook``) runs the script with ``cwd`` at the repo
    root and submits the resulting verdict as :class:`CheckEvidence`. The
    exit-code contract is fixed and caller-independent: 0 is ``pass``, 1 is
    ``fail``, anything else (including a crash) is ``error``. Captured
    stdout, capped, becomes the verdict's ``detail``.

    Unlike :class:`JudgeGate`, ``enforcement`` is not restricted to
    ``advisory``: a verifier's exit code is reproducible the way a glob or
    phrase match is, not a model's opinion, so a ``required`` check_passed
    gate can genuinely block. This is also the first gate type that *runs*
    something the policy names, rather than matching text or prompting a
    model, so its trust boundary is the repo: a script checked into the repo
    and named by that repo's own policy is the same trust level as a
    Makefile target or a pre-commit hook, which is why the caller resolves
    the verifier against the repo root and refuses a path that climbs out.
    There is deliberately no guard requiring the verifier to predate the
    diff under check, and no sandboxing: a "must predate this diff" rule was
    considered and rejected because it breaks the primary workflow this gate
    type is for, someone writing a new verifier and using it in the same
    change. See docs/agent-gates.md, which records what that boundary does
    not cover.

    ``when_changed`` is optional and, like ``JudgeGate``'s own field of the
    same name, the same repo-relative POSIX glob grammar
    ``ChangedPathGate.forbidden`` uses. Empty (the default) means this gate
    always applies.
    """

    id: str
    enforcement: Enforcement
    verifier: str
    message: str
    when_changed: tuple[str, ...] = ()
    type: Literal["check_passed"] = "check_passed"


# Extend this alias as a new gate type lands; do not let one skip it, or the
# policy loader's dispatch on ``type`` silently stops covering it.
GateSpec = ChangedPathGate | CommandMatchGate | CommandIfChangedGate | JudgeGate | CheckPassedGate


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
    one shared fact every gate matches independently), so there is no
    "collected, and there is none for this gate" case distinct from "this
    gate's id is simply missing" the way an empty ``changed_paths``/
    ``commands`` list differs from one that names something. Both resolve
    ``unknown`` identically.

    What *is* a tri-state, the same as the other two evidence kinds, is
    ``evaluate_judge``'s own ``evidence`` parameter being ``None`` at all: a
    caller whose event type never runs judge gates (``otari hook`` on
    `PreToolUse`, which has neither a finished diff nor a transcript to judge
    yet) submits no ``JudgeEvidence`` rather than an empty one, and resolves
    ``not_applicable`` rather than the ``unknown`` a caller that does run
    judge gates but is genuinely missing a verdict for this one gets.
    """

    verdicts: tuple[JudgeVerdict, ...]


@dataclass(frozen=True, slots=True)
class CheckVerdict:
    """One check_passed gate's verifier-produced verdict, as the caller observed it.

    ``outcome`` mirrors :class:`JudgeVerdict`'s own shape: the caller's own
    report, not a value Otari computed. ``"error"`` means the verifier
    script itself could not be run or exited with a status other than 0 or
    1 (a crash, a missing script, a permissions problem), distinct from
    ``"fail"`` (exit 1: the script ran and found a violation). Otari does
    not verify either. ``detail`` is the verifier's own captured stdout,
    capped the same way :class:`JudgeVerdict.reasoning` is.
    """

    gate_id: str
    outcome: Literal["pass", "fail", "error"]
    detail: str


@dataclass(frozen=True, slots=True)
class CheckEvidence:
    """Verdicts the caller collected for this request's check_passed gates.

    Structured exactly like :class:`JudgeEvidence`, for the same reason: a
    verdict already names the one gate it checked, so there is no
    "collected, and there is none for this gate" case beyond a missing gate
    id. What *is* a tri-state, the same as :class:`JudgeEvidence`'s own, is
    ``evaluate_check_passed``'s ``evidence`` parameter being ``None`` at
    all: a caller whose event type never runs check_passed gates
    (``otari hook`` on `PreToolUse`) submits no ``CheckEvidence`` rather
    than an empty one, and resolves ``not_applicable`` rather than the
    ``unknown`` a caller that does run check_passed gates but is genuinely
    missing a verdict for this one gets.
    """

    verdicts: tuple[CheckVerdict, ...]


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's evaluated outcome."""

    gate_id: str
    enforcement: Enforcement
    outcome: Outcome
    message: str
    detail: str | None = None
