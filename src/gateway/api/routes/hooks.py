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

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.agent_runtime.domain.evaluators import (
    evaluate_changed_path,
    evaluate_command_match,
    tokenize_commands,
    tokenize_phrases,
)
from gateway.agent_runtime.domain.policy import MAX_POLICY_BYTES, PolicyError, parse_policy
from gateway.agent_runtime.domain.types import ChangedPathEvidence, ChangedPathGate, CommandEvidence, CommandMatchGate
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
        return CommandEvidence(commands=tuple(dict.fromkeys(self.commands)))


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
            raise HTTPException(
                status_code=422, detail=f"commands entry exceeds {_MAX_COMMAND_LENGTH} characters."
            )

    changed_path_gates = [gate for gate in spec.gates if isinstance(gate, ChangedPathGate)]
    command_match_gates = [gate for gate in spec.gates if isinstance(gate, CommandMatchGate)]

    # Built once and reused below: gate.forbidden is already deduplicated at
    # parse time (domain.policy), and changed_path_evidence/command_evidence
    # deduplicate their evidence lists the same way, so each estimate and its
    # matching evaluation below always agree on the same, cheaper counts.
    changed_path_evidence = request.changed_path_evidence
    changed_paths = changed_path_evidence.changed_paths if changed_path_evidence is not None else ()
    pattern_count = sum(len(gate.forbidden) for gate in changed_path_gates)
    total_pattern_length = sum(len(glob) for gate in changed_path_gates for glob in gate.forbidden)
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
    # Gated on there being a command_match gate at all: tokenizing a command
    # is exactly the cost _MAX_TOTAL_COMMAND_CHARS below exists to bound. A
    # policy with none (every policy shipped before this gate type existed)
    # must not pay that cost just to prove there is nothing to bound it
    # against. Also gated on evidence actually being present: `commands`
    # omitted from the request means command_evidence is None, and there is
    # nothing to tokenize or bound in that case either.
    segment_cache: dict[str, list[list[str]]] | None = None
    phrase_cache: dict[str, list[str]] | None = None
    if command_match_gates and command_evidence is not None:
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
        # evaluate_command_match call as segment_cache): evaluate_command_match
        # is called once per command_match gate against this same evidence,
        # and without sharing this, each call would re-tokenize every command
        # from scratch, multiplying the already-checked cost above by the
        # number of gates. A request with 100 command_match gates each
        # forbidding "npm" against 250 distinct ~4,000-character commands
        # passed every budget here (low token content, few phrases, under
        # _MAX_TOTAL_COMMAND_CHARS) yet measured ~7s of synchronous blocking
        # from exactly that multiplication before this was shared.
        segment_cache = tokenize_commands(command_evidence.commands)

        # Policy parsing already proved every forbidden phrase tokenizes
        # (domain.policy's own validation), so this cannot raise. Shared with
        # the evaluation below via phrase_cache for the same reason
        # segment_cache is: tokenized here for the estimate and then again
        # inside every evaluate_command_match call is the same work twice.
        phrase_cache = tokenize_phrases(
            tuple(phrase for gate in command_match_gates for phrase in gate.forbidden)
        )
        phrase_count = sum(len(gate.forbidden) for gate in command_match_gates)
        total_phrase_tokens = sum(len(tokens) for tokens in phrase_cache.values())
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
    results = [
        evaluate_changed_path(gate, changed_path_evidence)
        if isinstance(gate, ChangedPathGate)
        else evaluate_command_match(
            gate, command_evidence, segment_cache=segment_cache, phrase_cache=phrase_cache
        )
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
