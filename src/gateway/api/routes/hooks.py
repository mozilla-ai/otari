"""Otari's Hook Server: evaluate an Agent Gates policy against caller-submitted evidence.

Otari never reads a caller's repository. The caller (an agent hook,
eventually the native ``otari hook`` dispatcher) already read its own
``.otari-gates.yml`` and collected its own Git evidence, and submits both
here in one request; this route parses and evaluates them and returns the
per-gate results. This is the integration mechanism that
docs/otari-product-foundation.md calls the Hook Server; see
docs/agent-gates.md for the request/response contract.

Every result is ``client_reported`` provenance: an authenticated request
identifies its sender, not the truth of what it claims about a repository
Otari cannot see. This route does no filesystem or Git I/O of its own, and
resolves no caller-selected local path (the exact mistake the production
plan's audit of the old POC calls out).
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.agent_runtime.domain.evaluators import (
    evaluate_changed_path,
    evaluate_command_if_changed,
    evaluate_command_match,
    evaluate_judge,
    tokenize_commands,
    tokenize_phrases,
)
from gateway.agent_runtime.domain.policy import MAX_GATE_ID_LENGTH, MAX_POLICY_BYTES, PolicyError, parse_policy
from gateway.agent_runtime.domain.types import (
    ChangedPathEvidence,
    ChangedPathGate,
    CommandEvidence,
    CommandIfChangedGate,
    CommandMatchGate,
    EvidenceScope,
    GateResult,
    GateSpec,
    JudgeEvidence,
    JudgeGate,
    JudgeVerdict,
)
from gateway.api.deps import get_config, get_db_if_needed, verify_api_key_or_master_key
from gateway.api.routes._platform import _extract_platform_user_token
from gateway.core.config import GatewayConfig


# ``AsyncSession`` and ``GatewayConfig`` are imported at runtime rather than
# under ``TYPE_CHECKING``, for the reason mcp.py spells out: this module uses
# postponed annotations, and FastAPI resolves a dependency's signature at
# import time to decide what each parameter is. Left as strings it cannot
# resolve, it reads both as query parameters and every request 422s before the
# handler runs.
async def verify_hook_caller(
    request: Request,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> None:
    """Authenticate a Hook Server caller in whichever mode this gateway runs.

    Standalone and hosted validate an API key or the master key against the
    local database, exactly as ``POST /api/v1/usage/external-events`` does.

    Hybrid has no local tenancy to validate against, so it only requires a
    bearer token to be present, the same thing the stateless MCP route does
    there. That is weaker on purpose and it is all this endpoint needs: it
    reads no tenant data, writes nothing, bills nothing, and evaluates only
    the policy and evidence the caller sent in the same request. What a
    request can cost is bounded by the work budgets below, not by who sent it.
    """
    if config.is_hybrid_mode:
        _extract_platform_user_token(request)
        return

    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        )
    await verify_api_key_or_master_key(request, db, config)


# The gate lives on the router, not the route: matches usage.ingest_router, so
# a route added here later inherits it rather than needing to remember it.
router = APIRouter(
    prefix="/hooks",
    tags=["hooks"],
    dependencies=[Depends(verify_hook_caller)],
)

# A submitted evidence list is caller-observed, not Otari-observed, but it is
# still bounded input: this caps a pathological request, not a real repo.
# The policy_yaml bound is domain.policy's own MAX_POLICY_BYTES, reused here
# rather than duplicated so the Pydantic-level and parser-level limits cannot
# drift apart.
_MAX_CHANGED_PATHS = 10_000
_MAX_PATH_LENGTH = 4096

# A per-match cost bound (domain/evaluators.py) does not bound the total cost
# of one request: MAX_POLICY_BYTES and _MAX_CHANGED_PATHS are each generous
# enough alone that maxing out both dimensions at once measured multiple
# seconds of matching in testing (100 forbidden globs x 10,000 changed paths,
# realistic lengths, took over 2.5s). This estimates total match work as
# pattern_count * total_path_length + path_count * total_pattern_length,
# which is what the matcher's own cost scales with, and rejects a request
# whose combination is disproportionate rather than let it run. Chosen with
# a safety margin under the ~350M-work / 0.58s point measured in benchmarking
# (tests/unit/agent_runtime/test_evaluators.py); a realistic policy (tens of
# gates, a handful of forbidden globs each) against a large changed-file set
# stays at least an order of magnitude under it.
_MAX_MATCH_WORK = 50_000_000

# A byte-weighted budget alone understates a request built from many *short*
# patterns and paths: each _segment_matches call costs a near-constant Python
# function-call overhead regardless of how few bytes it compares, so a
# request that is cheap by total bytes can still mean millions of individual
# calls. 2,500 one-byte forbidden globs against 10,000 one-byte changed paths
# measured 50,000,000 estimated work, exactly at (not over) _MAX_MATCH_WORK,
# for 25,000,000 real match calls that took ~5s. This bounds the raw call
# count directly, independent of length; benchmarking the same degenerate
# shape (short, non-matching, all-distinct strings, so neither the matcher's
# own short-circuits nor the deduplication in domain.policy and
# changed_path_evidence collapse the work) measured 2,000,000 calls at
# ~0.39-0.4s regardless of how that count split between pattern_count and
# path_count.
_MAX_COMPARISONS = 1_000_000

_MAX_COMMANDS = 10_000
_MAX_COMMAND_LENGTH = 4096

# command_match's per-(phrase, command) match cost is a product, not a sum:
# matching one forbidden phrase against one command segment is
# O(len(segment tokens) * len(phrase tokens)) (domain/evaluators.py's
# _contains_subsequence checks every candidate start position, each an
# O(len(phrase)) slice comparison). Summed over every phrase against every
# command, that product distributes into a single multiplication:
# total_pattern_tokens * total_command_tokens. This is a different shape from
# _MAX_MATCH_WORK's sum-of-cross-terms (changed_path's per-comparison cost is
# a *sum* of lengths, not a product), so it needs its own bound and its own
# calibration: 2,000 one-token forbidden phrases against 2,000 one-token
# commands (4,000,000 estimated work) measured ~0.7s; chosen with margin
# under that.
_MAX_COMMAND_MATCH_WORK = 2_000_000

# Independent of token length, for the same reason _MAX_COMPARISONS exists
# alongside _MAX_MATCH_WORK: many short phrases against many near-empty
# commands (e.g. all-whitespace strings, which tokenize to zero tokens each,
# so _MAX_COMMAND_MATCH_WORK's product is zero regardless of phrase count)
# is still one Python-level comparison per pair. 2,000 phrases against 10,000
# such commands (20,000,000 comparisons, 0 estimated work) measured ~4.7s.
# Chosen with margin under the ~1,000,000-comparisons / ~0.2s point measured
# at the same degenerate shape.
_MAX_COMMAND_COMPARISONS = 500_000

# Tokenizing itself is not free: shlex.split costs roughly 100-350ns per
# character it tokenizes, regardless of content, which is 50-150x the cost
# of a plain len() check. That is irrelevant at the scale of one command,
# but _MAX_COMMANDS * _MAX_COMMAND_LENGTH allows up to ~41,000,000
# characters in one request, and tokenizing all of it measured several
# seconds before either budget below ever saw a token count to reject: a
# policy with zero command_match gates would still pay this cost computing
# total_command_tokens, since that sum is what proves there is nothing to
# bound. This caps the raw character total *before* any command is
# tokenized, using only len() (uniformly cheap regardless of content).
# 2,000,000 characters measured ~0.17s; chosen with margin under that.
_MAX_TOTAL_COMMAND_CHARS = 2_000_000

# A judge verdict is a small, fixed-shape record (see JudgeVerdictRequest), not
# a pattern this route matches against other input, so its bound is a plain
# list-length/field-length cap rather than the work-estimate formula the
# path/command evidence kinds need: looking a verdict up by gate_id is O(n) in
# the number of judge gates in the policy, not a cross product.
_MAX_JUDGE_RESULTS = 1_000
_MAX_REASONING_LENGTH = 4_096


class JudgeVerdictRequest(BaseModel):
    """One judge gate's verdict, as the caller's own model call produced it."""

    model_config = ConfigDict(extra="forbid")

    # Shares domain.policy's own MAX_GATE_ID_LENGTH, not a separately chosen
    # 200: a verdict echoes back the gate id the policy itself named, and
    # policy.py's own parser rejects a longer one at parse time (422) for
    # exactly this reason, so every id this build accepts here can always
    # round-trip.
    gate_id: str = Field(min_length=1, max_length=MAX_GATE_ID_LENGTH)
    outcome: Literal["pass", "fail", "error"]
    reasoning: str = Field(default="", max_length=_MAX_REASONING_LENGTH)


class PolicyCheckRequest(BaseModel):
    """A policy body plus the evidence to check it against, both caller-supplied."""

    model_config = ConfigDict(extra="forbid")

    # Pydantic's max_length on a str counts characters, not UTF-8 bytes, so this
    # is a cheap early rejection, not the authoritative bound: parse_policy
    # re-checks the real byte length against the same MAX_POLICY_BYTES.
    policy_yaml: str = Field(min_length=1, max_length=MAX_POLICY_BYTES)
    # Tri-state, for the same reason `commands` below is: None (omitted, or an
    # explicit `null`) means this caller never collects path evidence at all,
    # and evaluate_changed_path reports `unknown`, blocking a required gate
    # rather than reading absent evidence as a pass; `[]` means it was
    # collected and there is none (`not_applicable`). This used to default to
    # `[]`, which collapsed the two and let an omitted field certify every
    # changed_path gate as passing.
    changed_paths: list[str] | None = Field(
        default=None,
        max_length=_MAX_CHANGED_PATHS,
        description="Repo-relative paths the caller observed changed (e.g. `git status --porcelain`).",
    )
    # None (omitted, or an explicit `null`) is distinct from `[]`: None means
    # this caller never collects command evidence at all (evaluate_command_match
    # reports `unknown`, blocking a required gate rather than reading absent
    # evidence as a pass); `[]` means it was collected and there is none right
    # now (`not_applicable`). Unlike changed_paths, an omitted commands field
    # is not defaulted to a list, because collapsing that distinction is
    # exactly the bug this field's default used to have.
    commands: list[str] | None = Field(
        default=None,
        max_length=_MAX_COMMANDS,
        description="Shell commands the caller observed run or is about to run.",
    )
    # Defaults to "call" so a client written before this field existed keeps
    # the semantics it was written against: one tool call's own command,
    # judged by command_match. Only a caller that really can see the whole
    # session (otari hook on a Stop event) says "session", and saying it is
    # what lets command_if_changed resolve and what takes command_match out
    # of the picture. See CommandEvidence.scope.
    command_scope: EvidenceScope = Field(
        default="call",
        description=(
            "What `commands` covers: `call` for the single tool call about to run, "
            "`session` for every command the session has run so far."
        ),
    )
    # A tri-state, like changed_paths/commands above, but for a different
    # reason: a verdict already names the one gate it judged, so there is no
    # "collected, and there is none for this gate" case an empty list needs
    # to express that a missing gate id doesn't already cover. What None
    # (omitted, or an explicit `null`) means instead is "this caller's event
    # type never runs judge gates at all" (otari hook on PreToolUse, which
    # has neither a finished diff nor a transcript to judge yet): resolving
    # that the same `unknown` a caller that does run judge gates but is
    # missing one gets would warn on every single PreToolUse edit to a
    # when_changed-matched path, regardless of how well-behaved the session
    # was (see JudgeEvidence's and evaluate_judge's own docstrings).
    judge_results: list[JudgeVerdictRequest] | None = Field(
        default=None,
        max_length=_MAX_JUDGE_RESULTS,
        description="Model verdicts the caller collected for this request's judge gates.",
    )

    @property
    def changed_path_evidence(self) -> ChangedPathEvidence | None:
        # A duplicate path adds nothing a single copy wouldn't already tell a
        # gate; collapsing it here means the work-budget check below and the
        # actual matching agree on the same, cheaper count rather than one
        # estimating off raw input and the other paying for the duplicates.
        if self.changed_paths is None:
            return None
        return ChangedPathEvidence(changed_paths=tuple(dict.fromkeys(self.changed_paths)))

    @property
    def command_evidence(self) -> CommandEvidence | None:
        if self.commands is None:
            return None
        return CommandEvidence(commands=tuple(dict.fromkeys(self.commands)), scope=self.command_scope)

    @property
    def judge_evidence(self) -> JudgeEvidence | None:
        if self.judge_results is None:
            return None
        return JudgeEvidence(
            verdicts=tuple(
                JudgeVerdict(gate_id=verdict.gate_id, outcome=verdict.outcome, reasoning=verdict.reasoning)
                for verdict in self.judge_results
            )
        )


class GateResultResponse(BaseModel):
    gate_id: str
    enforcement: str
    outcome: str
    message: str
    detail: str | None = None


class PolicyCheckResponse(BaseModel):
    policy_id: str
    schema_version: str
    provenance: str = "client_reported"
    results: list[GateResultResponse]
    blocked: bool


def _evaluate_gate(
    gate: GateSpec,
    changed_path_evidence: ChangedPathEvidence | None,
    command_evidence: CommandEvidence | None,
    judge_evidence: JudgeEvidence | None,
    segment_cache: dict[str, list[list[str]]] | None,
    phrase_cache: dict[str, list[str]] | None,
) -> GateResult:
    """Dispatch one gate to its evaluator. Extend as a new gate type joins ``GateSpec``."""
    if isinstance(gate, ChangedPathGate):
        return evaluate_changed_path(gate, changed_path_evidence)
    if isinstance(gate, CommandMatchGate):
        return evaluate_command_match(gate, command_evidence, segment_cache=segment_cache, phrase_cache=phrase_cache)
    if isinstance(gate, JudgeGate):
        return evaluate_judge(gate, changed_path_evidence, judge_evidence)
    return evaluate_command_if_changed(
        gate, changed_path_evidence, command_evidence, segment_cache=segment_cache, phrase_cache=phrase_cache
    )


@router.post("/check")
async def check_policy(request: PolicyCheckRequest) -> PolicyCheckResponse:
    """Evaluate a submitted policy against submitted evidence.

    Authenticated with either an API key or the master key (the router-level
    gate), like ``POST /api/v1/usage/external-events``: this identifies who
    sent the request, not whether its evidence is true. `blocked` is set when
    a required gate's outcome is not `pass`/`not_applicable` (an unresolved
    gate never counts as a pass).
    """
    try:
        spec = parse_policy(request.policy_yaml, source="request body")
    except PolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    for path in request.changed_paths or []:
        if len(path) > _MAX_PATH_LENGTH:
            raise HTTPException(status_code=422, detail=f"changed_paths entry exceeds {_MAX_PATH_LENGTH} characters.")
    for command in request.commands or []:
        if len(command) > _MAX_COMMAND_LENGTH:
            raise HTTPException(status_code=422, detail=f"commands entry exceeds {_MAX_COMMAND_LENGTH} characters.")

    changed_path_gates = [gate for gate in spec.gates if isinstance(gate, ChangedPathGate)]
    command_match_gates = [gate for gate in spec.gates if isinstance(gate, CommandMatchGate)]
    command_if_changed_gates = [gate for gate in spec.gates if isinstance(gate, CommandIfChangedGate)]
    judge_gates = [gate for gate in spec.gates if isinstance(gate, JudgeGate)]

    # Built once and reused below: gate.forbidden/when_changed/require are
    # already deduplicated at parse time (domain.policy), and
    # changed_path_evidence/command_evidence deduplicate their evidence lists
    # the same way, so each estimate and its matching evaluation below always
    # agree on the same, cheaper counts. command_if_changed's and judge's own
    # when_changed globs are path-matching work exactly like changed_path's
    # forbidden globs (evaluate_judge calls the same matched_changed_paths),
    # so all three share the same budget rather than needing a third or
    # fourth one.
    changed_path_evidence = request.changed_path_evidence
    changed_paths = changed_path_evidence.changed_paths if changed_path_evidence is not None else ()
    path_globs = (
        [glob for gate in changed_path_gates for glob in gate.forbidden]
        + [glob for gate in command_if_changed_gates for glob in gate.when_changed]
        + [glob for gate in judge_gates for glob in gate.when_changed]
    )
    pattern_count = len(path_globs)
    total_pattern_length = sum(len(glob) for glob in path_globs)
    path_count = len(changed_paths)
    total_path_length = sum(len(path) for path in changed_paths)
    estimated_work = pattern_count * total_path_length + path_count * total_pattern_length
    comparisons = pattern_count * path_count
    if estimated_work > _MAX_MATCH_WORK or comparisons > _MAX_COMPARISONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"This policy and evidence would take an estimated {estimated_work:,} match operations "
                f"across {comparisons:,} pattern/path comparisons, over this build's limits "
                f"({_MAX_MATCH_WORK:,} and {_MAX_COMPARISONS:,} respectively). Narrow the policy's "
                "forbidden globs or the submitted changed_paths."
            ),
        )

    command_evidence = request.command_evidence
    # Gated on there being a gate that reads command evidence at all:
    # tokenizing a command is exactly the cost _MAX_TOTAL_COMMAND_CHARS below
    # exists to bound. A policy with neither command_match nor
    # command_if_changed (every policy shipped before this gate type
    # existed) must not pay that cost just to prove there is nothing to
    # bound it against. Also gated on evidence actually being present:
    # `commands` omitted from the request means command_evidence is None,
    # and there is nothing to tokenize or bound in that case either.
    segment_cache: dict[str, list[list[str]]] | None = None
    phrase_cache: dict[str, list[str]] | None = None
    if (command_match_gates or command_if_changed_gates) and command_evidence is not None:
        total_command_chars = sum(len(command) for command in command_evidence.commands)
        if total_command_chars > _MAX_TOTAL_COMMAND_CHARS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Submitted commands total {total_command_chars:,} characters, over the "
                    f"{_MAX_TOTAL_COMMAND_CHARS:,} limit. Narrow the submitted commands."
                ),
            )

        # Tokenized exactly once here and reused for both the estimate below
        # and the real evaluation further down (passed to every
        # evaluate_command_match/evaluate_command_if_changed call as
        # segment_cache): each is called once per gate against this same
        # evidence, and without sharing this, each call would re-tokenize
        # every command from scratch, multiplying the already-checked cost
        # above by the number of gates. A request with 100 command_match
        # gates each forbidding "npm" against 250 distinct ~4,000-character
        # commands passed every budget here (low token content, few phrases,
        # under _MAX_TOTAL_COMMAND_CHARS) yet measured ~7s of synchronous
        # blocking from exactly that multiplication before this was shared.
        segment_cache = tokenize_commands(command_evidence.commands)

        # Policy parsing already proved every forbidden/require phrase
        # tokenizes (domain.policy's own validation), so this cannot raise.
        # Shared with the evaluation below via phrase_cache for the same
        # reason segment_cache is: tokenized here for the estimate and then
        # again inside every evaluate_command_match/evaluate_command_if_changed
        # call is the same work twice. command_if_changed's require phrases
        # are command-matching work exactly like command_match's forbidden
        # phrases, so they share the same budget and cache rather than
        # needing a third one.
        command_phrases = tuple(phrase for gate in command_match_gates for phrase in gate.forbidden) + tuple(
            phrase for gate in command_if_changed_gates for phrase in gate.require
        )
        phrase_cache = tokenize_phrases(command_phrases)
        # Per gate occurrence, not per distinct phrase text: phrase_cache
        # dedupes identical phrase text across gates so each is tokenized
        # once, but evaluate_command_match/evaluate_command_if_changed still
        # run _contains_subsequence once per gate that carries it. Summing
        # len(phrase_cache.values()) counted a shared phrase's tokens once
        # regardless of how many gates forbid/require it, undercounting the
        # real per-gate matching work whenever gates share phrase text.
        phrase_count = sum(len(gate.forbidden) for gate in command_match_gates) + sum(
            len(gate.require) for gate in command_if_changed_gates
        )
        total_phrase_tokens = sum(
            len(phrase_cache[phrase]) for gate in command_match_gates for phrase in gate.forbidden
        ) + sum(len(phrase_cache[phrase]) for gate in command_if_changed_gates for phrase in gate.require)
        command_count = len(command_evidence.commands)
        total_command_tokens = sum(len(segment) for segments in segment_cache.values() for segment in segments)
        estimated_command_work = total_phrase_tokens * total_command_tokens
        command_comparisons = phrase_count * command_count
        if estimated_command_work > _MAX_COMMAND_MATCH_WORK or command_comparisons > _MAX_COMMAND_COMPARISONS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"This policy and evidence would take an estimated {estimated_command_work:,} command match "
                    f"operations across {command_comparisons:,} phrase/command comparisons, over this build's "
                    f"limits ({_MAX_COMMAND_MATCH_WORK:,} and {_MAX_COMMAND_COMPARISONS:,} respectively). Narrow "
                    "the policy's forbidden phrases or the submitted commands."
                ),
            )

    # Evaluated in declaration order (not grouped by type) so a caller reading
    # `results` positionally sees the same order as the policy it submitted.
    judge_evidence = request.judge_evidence
    results = [
        _evaluate_gate(gate, changed_path_evidence, command_evidence, judge_evidence, segment_cache, phrase_cache)
        for gate in spec.gates
    ]
    blocked = any(result.enforcement == "required" and result.outcome.is_blocking for result in results)

    return PolicyCheckResponse(
        policy_id=spec.policy_id,
        schema_version=spec.schema_version,
        results=[
            GateResultResponse(
                gate_id=result.gate_id,
                enforcement=result.enforcement,
                outcome=result.outcome.value,
                message=result.message,
                detail=result.detail,
            )
            for result in results
        ],
        blocked=blocked,
    )
