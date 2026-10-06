"""Model recommendations for a coding agent's subagents (standalone only).

``POST /api/v1/routing/recommend`` is asked by an agent harness, such as a
Claude Code mod hooked on ``agent.spawn``, in the moment between deciding to
start a subagent and starting it. The gateway puts one choice question to the
configured decision model and recommends the candidate it picks; the harness
runs the subagent on it against the harness's own provider credentials.

The decision call goes through the decisions scaffold, so it is billed to the
caller and recorded in usage like a ``POST /api/v1/decisions`` call. Nothing
else is dispatched here, which is why any active API key may ask rather than
only an operator. See ``docs/use-with-claude-code.md``.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from gateway.api.deps import get_config, verify_api_key_or_master_key
from gateway.api.routes._passthrough import PASSTHROUGH_PROVIDER_ERROR_DETAIL, DecisionScaffold, get_decision_scaffold
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.schemas.routing import AgentModelRecommendation, AgentModelRecommendationRequest
from gateway.services.routing import (
    UnreadableRecommendationError,
    build_decision_request,
    recommendation_from_decision,
)

router = APIRouter(prefix="/routing", tags=["routing"])


@router.post("/recommend")
async def recommend_model_for_agent(
    raw_request: Request,
    response: Response,
    request: AgentModelRecommendationRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    scaffold: Annotated[DecisionScaffold, Depends(get_decision_scaffold)],
) -> AgentModelRecommendation:
    """Recommend the model a subagent about to start should run on.

    The harness sends the facts it holds at spawn time: its own ids for the
    session and the spawning tool call, the subagent type, the task, the
    parent's model and the model the caller asked for, if any. The gateway asks
    the decision model named by `agent_recommender_model` to pick one of the
    `agent_recommender_candidates`, and answers with that candidate, the
    model's probability for each candidate, and a one-line reason.

    Authentication modes:
    - Master key: the ``user`` field is required and names who the decision is billed to.
    - API key: the decision is billed to the key's own user.
    """
    decision_request = build_decision_request(
        request,
        decision_model=config.agent_recommender_model,
        candidates=config.agent_recommender_candidates,
    )
    decision = await scaffold.run(
        raw_request=raw_request, response=response, request=decision_request, auth_result=auth_result
    )
    try:
        return recommendation_from_decision(decision, candidates=config.agent_recommender_candidates)
    except UnreadableRecommendationError as exc:
        logger.error("Agent model recommendation from %s is unreadable: %s", decision.model, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=PASSTHROUGH_PROVIDER_ERROR_DETAIL) from exc
