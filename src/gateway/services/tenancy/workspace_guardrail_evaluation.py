"""What a hybrid peer answers the gateway's guardrail calls with.

A hybrid gateway holds no guardrail secrets. Resolve names the mandates as
descriptors, and the gateway asks the peer to run them
(``docs/hybrid-mode-protocol.md``, "Guardrail evaluation"). This module is that
peer's half, kept in Otari so every peer answers alike. The peer reads the
workspace's mandates itself (``resolve_organization_guardrails``) and hands
them in, so nothing here touches a database.

It reports outcomes and decides nothing. ``mode`` and ``on_unavailable`` travel
on the descriptor, and the gateway applies them.

A hosted guardrail spends the deployment's vendor secret, so it runs only when
the caller passes a port: a peer passes one for its own default gateway and
``None`` for any gateway someone self-hosts.
"""

import asyncio
import uuid
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from gateway.log_config import logger
from gateway.ports.hosted_guardrail_port import HostedGuardrailPort
from gateway.services.guardrails import (
    GuardrailsNotReachableError,
    GuardrailUnfundedError,
    InProcessGuardrail,
    run_input_guardrails,
)
from gateway.services.tenancy.hosted_guardrail_check import HostedGuardrailCheck, HostedMandate
from gateway.services.tenancy.organization_guardrail_runner import handle as guardrail_handle
from gateway.services.tenancy.organization_guardrail_service import ResolvedOrganizationGuardrail
from gateway.services.url_safety import UnsafeURLError

EvaluationStatus = Literal["evaluated", "unavailable", "unfunded"]

# One request's mandates, bounded by the per-organization cap; the wire allows a
# little headroom so a cap raised later does not break an older peer.
MAX_GUARDRAIL_IDS = 32


class GuardrailDescriptor(BaseModel):
    """A mandate as resolve names it: the check, and nothing about where it runs."""

    id: uuid.UUID
    profile: str
    mode: Literal["block", "monitor"]
    on_unavailable: Literal["block", "monitor"]


class GuardrailEvaluationRequest(BaseModel):
    """The body of ``POST /gateway/guardrails/evaluate``."""

    request_id: str = Field(min_length=1, max_length=256)
    guardrail_ids: list[uuid.UUID] = Field(min_length=1, max_length=MAX_GUARDRAIL_IDS)
    input_text: str


class GuardrailEvaluation(BaseModel):
    """One mandate's outcome. ``valid`` is ``None`` unless the check was evaluated."""

    id: uuid.UUID
    status: EvaluationStatus
    valid: bool | None = None
    explanation: str | None = None
    score: float | None = None


class GuardrailEvaluationResponse(BaseModel):
    results: list[GuardrailEvaluation]


def guardrail_descriptors(mandates: Sequence[ResolvedOrganizationGuardrail]) -> list[GuardrailDescriptor]:
    """The descriptors resolve returns for a workspace's mandates."""
    return [
        GuardrailDescriptor(
            id=mandate.id,
            profile=mandate.config.profile,
            mode=mandate.config.mode,
            on_unavailable=mandate.config.on_unavailable,
        )
        for mandate in mandates
        if mandate.id is not None
    ]


async def evaluate_guardrail_mandates(
    mandates: Sequence[ResolvedOrganizationGuardrail],
    *,
    organization_id: uuid.UUID,
    workspace_id: uuid.UUID,
    request_id: str,
    guardrail_ids: Sequence[uuid.UUID],
    input_text: str,
    hosted_guardrails: HostedGuardrailPort | None,
    default_url: str | None,
) -> list[GuardrailEvaluation]:
    """Run each named mandate over ``input_text`` and report one outcome per id.

    ``mandates`` are the workspace's, read by the caller for this call, which is
    what scopes the ids: one the workspace does not mandate is unavailable, never
    run. The checks run concurrently, since each may wait on a vendor.
    """
    in_scope = {mandate.id: mandate for mandate in mandates if mandate.id is not None}

    async def one(mandate_id: uuid.UUID) -> GuardrailEvaluation:
        mandate = in_scope.get(mandate_id)
        if mandate is None:
            return GuardrailEvaluation(id=mandate_id, status="unavailable")
        return await _evaluate_one(
            mandate,
            mandate_id=mandate_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            request_id=request_id,
            input_text=input_text,
            hosted_guardrails=hosted_guardrails,
            default_url=default_url,
        )

    return list(await asyncio.gather(*(one(mandate_id) for mandate_id in dict.fromkeys(guardrail_ids))))


async def _evaluate_one(
    mandate: ResolvedOrganizationGuardrail,
    *,
    mandate_id: uuid.UUID,
    organization_id: uuid.UUID,
    workspace_id: uuid.UUID,
    request_id: str,
    input_text: str,
    hosted_guardrails: HostedGuardrailPort | None,
    default_url: str | None,
) -> GuardrailEvaluation:
    profile = mandate.config.profile
    # Run as fail-closed so every failure raises rather than being folded into
    # an inconclusive result: telling the two apart is this answer's job.
    config = mandate.config.model_copy(update={"mode": "block", "on_unavailable": "block"})
    in_process: dict[str, InProcessGuardrail | None] = {}
    if mandate.definition_id is not None:
        in_process[profile] = guardrail_handle(organization_id, mandate.definition_id)
    elif mandate.hosted_guardrail_id is not None:
        in_process[profile] = (
            HostedGuardrailCheck(
                hosted_guardrails,
                organization_id=organization_id,
                workspace_id=workspace_id,
                mandate=HostedMandate(mandate_id=mandate_id, hosted_guardrail_id=mandate.hosted_guardrail_id),
                request_id=request_id,
            )
            if hosted_guardrails is not None
            else None
        )
    try:
        verdict = await run_input_guardrails(
            [config],
            input_text,
            default_url=default_url,
            credentials={profile: mandate.credential} if mandate.credential else None,
            mandated={profile},
            in_process=in_process,
        )
    except GuardrailUnfundedError:
        return GuardrailEvaluation(id=mandate_id, status="unfunded")
    except (GuardrailsNotReachableError, UnsafeURLError) as exc:
        logger.warning("guardrail mandate %s could not be evaluated for a hybrid gateway: %s", mandate_id, exc)
        return GuardrailEvaluation(id=mandate_id, status="unavailable")
    if not verdict.results:
        return GuardrailEvaluation(id=mandate_id, status="unavailable")
    result = verdict.results[0]
    score = result.score
    return GuardrailEvaluation(
        id=mandate_id,
        status="evaluated",
        valid=result.valid,
        explanation=None if result.explanation is None else str(result.explanation),
        score=float(score) if isinstance(score, int | float) and not isinstance(score, bool) else None,
    )
