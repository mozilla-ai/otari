"""Otari's Hook Server: evaluate an Agent Gates policy against caller-submitted evidence.

Otari never reads a caller's repository. The caller (an agent hook, e.g.
``otari hook``) already read its own ``.otari-gates.yml`` and collected its
own Git evidence, and submits both here in one request; this route parses
and evaluates them and returns the per-gate results, exactly the way ``otari
hook`` itself evaluates the same policy in process by default (see
``otari_agent.domain.check.run_policy_check``, which both call): this
route is the opt-in path for a caller that wants a gateway to be the one
deciding instead. This is the integration mechanism that
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

from gateway.api.deps import extract_credential_token, get_config, get_db_if_needed, verify_api_key_or_master_key
from gateway.core.config import GatewayConfig
from otari_agent.domain.check import PolicyCheckError, run_policy_check
from otari_agent.domain.policy import MAX_GATE_ID_LENGTH, MAX_POLICY_BYTES
from otari_agent.domain.types import CheckVerdict, EvidenceScope, JudgeVerdict


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
    credential header to be present, the same thing the stateless MCP route
    does there. That is weaker on purpose and it is all this endpoint needs: it
    reads no tenant data, writes nothing, bills nothing, and evaluates only
    the policy and evidence the caller sent in the same request. What a
    request can cost is bounded by ``otari_agent.domain.check``'s own work
    budgets, not by who sent it.
    """
    if config.is_hybrid_mode:
        extract_credential_token(request)
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
# drift apart. The match-cost budgets that used to sit here (per-entry
# length, and the total-work estimates for changed_path/command_match/
# command_if_changed) moved to otari_agent.domain.check.run_policy_check,
# since they guard the evaluator's own cost, not this route's: `otari hook`'s
# own local evaluation needs them just as much as an HTTP caller does, and
# sharing one place keeps the two from drifting apart.
_MAX_CHANGED_PATHS = 10_000
_MAX_COMMANDS = 10_000

# A judge verdict is a small, fixed-shape record (see JudgeVerdictRequest), not
# a pattern this route matches against other input, so its bound is a plain
# list-length/field-length cap rather than the work-estimate formula the
# path/command evidence kinds need: looking a verdict up by gate_id is O(n) in
# the number of judge gates in the policy, not a cross product.
_MAX_JUDGE_RESULTS = 1_000
_MAX_REASONING_LENGTH = 4_096

# A check_passed verdict is the same small, fixed-shape record shape as a
# judge verdict (see CheckVerdictRequest), bounded the same way and for the
# same reason: looking a verdict up by gate_id is O(n) in the number of
# check_passed gates in the policy, not a cross product.
_MAX_CHECK_RESULTS = 1_000
_MAX_CHECK_DETAIL_LENGTH = 4_096


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


class CheckVerdictRequest(BaseModel):
    """One check_passed gate's verdict, as the caller's own verifier run produced it.

    Mirrors ``JudgeVerdictRequest`` field-for-field: ``gate_id`` echoes back
    the gate the policy itself named (same bound, same reason), ``outcome``
    is the caller's own report of the verifier's exit code (0/1/anything
    else, mapped by the caller to pass/fail/error), and ``detail`` is the
    verifier's captured stdout, capped the same way ``reasoning`` is.
    """

    model_config = ConfigDict(extra="forbid")

    gate_id: str = Field(min_length=1, max_length=MAX_GATE_ID_LENGTH)
    outcome: Literal["pass", "fail", "error"]
    detail: str = Field(default="", max_length=_MAX_CHECK_DETAIL_LENGTH)


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
    # A tri-state exactly like judge_results, for the same reason: a verdict
    # already names the one gate its verifier checked, so there is no
    # "collected, and there is none for this gate" case an empty list needs
    # to express beyond a missing gate id. None (omitted, or an explicit
    # `null`) means this caller's event type never runs check_passed gates
    # at all (otari hook on PreToolUse, which has no finished session for a
    # verifier to check yet) and resolves every check_passed gate
    # not_applicable rather than the unknown a caller that does run them but
    # is genuinely missing one gets (see CheckEvidence's and
    # evaluate_check_passed's own docstrings).
    check_results: list[CheckVerdictRequest] | None = Field(
        default=None,
        max_length=_MAX_CHECK_RESULTS,
        description="Verifier verdicts the caller collected for this request's check_passed gates.",
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


@router.post("/check")
async def check_policy(request: PolicyCheckRequest) -> PolicyCheckResponse:
    """Evaluate a submitted policy against submitted evidence.

    Authenticated with either an API key or the master key (the router-level
    gate), like ``POST /api/v1/usage/external-events``: this identifies who
    sent the request, not whether its evidence is true. `blocked` is set when
    a required gate's outcome is not `pass`/`not_applicable` (an unresolved
    gate never counts as a pass).

    The actual parse-and-evaluate work is
    ``otari_agent.domain.check.run_policy_check``, shared with ``otari
    hook``'s own local evaluation: this route's own job is authentication,
    translating that function's tri-state request fields into its own typed
    ones, and turning ``PolicyCheckError`` into a 422.
    """
    try:
        result = run_policy_check(
            request.policy_yaml,
            source="request body",
            changed_paths=request.changed_paths,
            commands=request.commands,
            command_scope=request.command_scope,
            judge_results=(
                None
                if request.judge_results is None
                else [
                    JudgeVerdict(gate_id=verdict.gate_id, outcome=verdict.outcome, reasoning=verdict.reasoning)
                    for verdict in request.judge_results
                ]
            ),
            check_results=(
                None
                if request.check_results is None
                else [
                    CheckVerdict(gate_id=verdict.gate_id, outcome=verdict.outcome, detail=verdict.detail)
                    for verdict in request.check_results
                ]
            ),
        )
    except PolicyCheckError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PolicyCheckResponse(
        policy_id=result.policy_id,
        schema_version=result.schema_version,
        results=[
            GateResultResponse(
                gate_id=gate_result.gate_id,
                enforcement=gate_result.enforcement,
                outcome=gate_result.outcome.value,
                message=gate_result.message,
                detail=gate_result.detail,
            )
            for gate_result in result.results
        ],
        blocked=result.blocked,
    )
