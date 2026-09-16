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

from gateway.agent_runtime.domain.evaluators import evaluate_changed_path
from gateway.agent_runtime.domain.policy import MAX_POLICY_BYTES, PolicyError, parse_policy
from gateway.agent_runtime.domain.types import ChangedPathEvidence
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


class PolicyCheckRequest(BaseModel):
    """A policy body plus the evidence to check it against, both caller-supplied."""

    model_config = ConfigDict(extra="forbid")

    # Pydantic's max_length on a str counts characters, not UTF-8 bytes, so this
    # is a cheap early rejection, not the authoritative bound: parse_policy
    # re-checks the real byte length against the same MAX_POLICY_BYTES.
    policy_yaml: str = Field(min_length=1, max_length=MAX_POLICY_BYTES)
    changed_paths: list[str] = Field(
        default_factory=list,
        max_length=_MAX_CHANGED_PATHS,
        description="Repo-relative paths the caller observed changed (e.g. `git status --porcelain`).",
    )

    @property
    def changed_path_evidence(self) -> ChangedPathEvidence:
        # A duplicate path adds nothing a single copy wouldn't already tell a
        # gate; collapsing it here means the work-budget check below and the
        # actual matching agree on the same, cheaper count rather than one
        # estimating off raw input and the other paying for the duplicates.
        return ChangedPathEvidence(changed_paths=tuple(dict.fromkeys(self.changed_paths)))


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

    for path in request.changed_paths:
        if len(path) > _MAX_PATH_LENGTH:
            raise HTTPException(status_code=422, detail=f"changed_paths entry exceeds {_MAX_PATH_LENGTH} characters.")

    # Built once and reused below: gate.forbidden is already deduplicated at
    # parse time (domain.policy), and changed_path_evidence deduplicates
    # changed_paths the same way, so this estimate and the actual evaluation
    # below always agree on the same, cheaper counts.
    evidence = request.changed_path_evidence
    pattern_count = sum(len(gate.forbidden) for gate in spec.gates)
    total_pattern_length = sum(len(glob) for gate in spec.gates for glob in gate.forbidden)
    path_count = len(evidence.changed_paths)
    total_path_length = sum(len(path) for path in evidence.changed_paths)
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

    results = [evaluate_changed_path(gate, evidence) for gate in spec.gates]
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
