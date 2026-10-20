"""Model recommendations for a coding agent's subagents (control plane).

``POST /api/v1/routing/recommend`` is asked by an agent harness, such as a
Claude Code plugin hooked on ``agent.spawn``, in the moment between deciding to
start a subagent and starting it. Whatever this build bound to
``AgentModelRecommenderPort`` answers: the core puts one choice question to the
configured decision model, and a hosted build may bind a recommender of its
own. The harness runs the subagent on the answer against its own provider
credentials.

The call goes through the decisions scaffold, so it is metered and recorded
in usage like a ``POST /api/v1/decisions`` call: priced from this deployment's
own rows, or held and charged at the price the recommender quotes. Nothing
else is dispatched here, which is why any active API key may ask rather than
only an operator. See ``docs/use-with-claude-code.md``.

``/api/v1/routing/recommenders`` is where the deployment's operator adds the
decision models the core recommender asks, with their keys and their prices,
and switches one on. It is on a router of its own, behind the operator gate
``/api/v1/hosted-providers`` holds, because it stores a deployment credential.
**It never returns a key**, only whether one is stored and its last four
characters.
"""

import uuid
from collections.abc import Mapping
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from gateway.api.deps import (
    AgentModelRecommenderPortDep,
    RoutingServiceDep,
    get_config,
    require_deployment_operator_or_absent,
    verify_api_key_or_master_key,
    verify_master_key,
)
from gateway.api.routes._passthrough import (
    ESTIMATED_OUTPUT_TOKENS_PER_QUESTION,
    PASSTHROUGH_PROVIDER_ERROR_DETAIL,
    DecisionCallError,
    DecisionOutcome,
    DecisionQuote,
    DecisionScaffold,
    DecisionUnavailableError,
    decision_input_chars,
    decision_status_error,
    get_decision_scaffold,
)
from gateway.core.config import GatewayConfig
from gateway.core.surface import Surface
from gateway.exceptions.routing_exceptions import (
    CandidatesNotAllowedError,
    RecommendationFailedError,
    RecommenderNotConfiguredError,
    UnreadableRecommendationError,
)
from gateway.models.api_keys import APIKey
from gateway.ports.agent_model_recommender_port import AgentModelRecommenderPort, ModelRecommendation, SubagentSpawn
from gateway.schemas.routing import (
    AgentModelRecommendation,
    AgentModelRecommendationRequest,
    AgentRecommenderCreate,
    AgentRecommenderPublic,
    AgentRecommendersPublic,
    AgentRecommenderUpdate,
)
from gateway.services.routing import build_decision_request, candidates_for

router = APIRouter(prefix="/routing", tags=["routing"])

recommenders_router = APIRouter(
    prefix="/routing/recommenders",
    tags=["routing"],
    dependencies=[Depends(verify_master_key), Depends(require_deployment_operator_or_absent)],
)

# Published by both topologies, operator-only on either, as the hosted
# providers page is.
SURFACE = Surface("subagent_router")


class _RecommenderCall:
    """The port's two steps, as the decisions scaffold runs them."""

    def __init__(
        self, recommender: AgentModelRecommenderPort, spawn: SubagentSpawn, candidates: Mapping[str, str | None]
    ) -> None:
        self._recommender = recommender
        self._spawn = spawn
        self._candidates = candidates

    def resolve(self) -> DecisionQuote:
        try:
            quote = self._recommender.quote()
        except RecommenderNotConfiguredError as exc:
            raise DecisionUnavailableError(str(exc)) from exc
        return DecisionQuote(provider=quote.provider, model=quote.model, charge=quote.charge)

    async def dispatch(self) -> DecisionOutcome[ModelRecommendation]:
        try:
            recommendation = await self._recommender.recommend(self._spawn, candidates=self._candidates)
        except RecommendationFailedError as exc:
            raise DecisionCallError(
                str(exc),
                logged_status=exc.upstream_status or status.HTTP_502_BAD_GATEWAY,
                response=decision_status_error(exc.upstream_status),
            ) from exc
        except UnreadableRecommendationError as exc:
            unreadable = HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=PASSTHROUGH_PROVIDER_ERROR_DETAIL
            )
            raise DecisionCallError(str(exc), logged_status=status.HTTP_502_BAD_GATEWAY, response=unreadable) from exc
        usage = recommendation.usage
        return DecisionOutcome(
            result=recommendation,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            charge=usage.charge,
        )


@router.post("/recommend")
async def recommend_model_for_agent(
    raw_request: Request,
    response: Response,
    request: AgentModelRecommendationRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    recommender: AgentModelRecommenderPortDep,
    scaffold: Annotated[DecisionScaffold, Depends(get_decision_scaffold)],
) -> AgentModelRecommendation:
    """Recommend the model a subagent about to start should run on.

    The harness sends the facts it holds at spawn time: its own ids for the
    session and the spawning tool call, the subagent type, the task, the
    parent's model and the model the caller asked for, if any. The recommender
    this build binds picks one of the candidate models and the answer names it.
    A standalone deployment asks the decision model `agent_recommender_model`
    names to choose among `agent_recommender_candidates` for Claude Code and
    among the request's own `candidates` for pi; a hosted build may bind a
    recommender of its own, which the caller never configures. Where
    the recommender reports them, the answer also carries a one-line reason
    and its probability for each candidate.

    Authentication modes:
    - Master key: the ``user`` field is required and names who the decision is billed to.
    - API key: the decision is billed to the key's own user.
    """
    spawn = SubagentSpawn(
        harness=request.harness,
        session_id=request.session_id,
        tool_use_id=request.tool_use_id,
        agent_type=request.agent_type,
        description=request.description,
        prompt=request.prompt,
        parent_model=request.parent_model,
        requested_model=request.requested_model,
    )
    try:
        candidates = candidates_for(
            request.harness, sent=request.candidates, claude_code=config.agent_recommender_candidates
        )
    except CandidatesNotAllowedError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    # The hold is sized from the question the core would put, description and
    # candidates included, which is also the best guess at what a bound recommender reads.
    question = build_decision_request(spawn, decision_model=config.agent_recommender_model, candidates=candidates)
    recommendation = await scaffold.run_call(
        raw_request=raw_request,
        response=response,
        auth_result=auth_result,
        selector=config.agent_recommender_model,
        user=request.user,
        prompt_chars=decision_input_chars(question),
        default_output_tokens=ESTIMATED_OUTPUT_TOKENS_PER_QUESTION,
        call=_RecommenderCall(recommender, spawn, candidates),
    )
    return AgentModelRecommendation(
        model=recommendation.model,
        reason=recommendation.reason,
        probabilities=dict(recommendation.probabilities) if recommendation.probabilities is not None else None,
    )


@recommenders_router.get("")
async def list_agent_recommenders(routing: RoutingServiceDep) -> AgentRecommendersPublic:
    """List the decision models added to recommend a subagent's model, oldest first.

    At most one is switched on, and it answers `POST /api/v1/routing/recommend`
    at its own price. While none has been added, `agent_recommender_model`
    answers; once one has, with every one switched off, nothing does. The answer
    also lists the decision providers one may be added on.
    """
    return await routing.list_agent_recommenders()


@recommenders_router.post("", status_code=status.HTTP_201_CREATED)
async def add_agent_recommender(routing: RoutingServiceDep, body: AgentRecommenderCreate) -> AgentRecommenderPublic:
    """Add a decision model that recommends a subagent's model, with its key and its price per million tokens.

    It lands switched off unless `enabled` is true, and switching it on
    switches every other one off. A recommendation is charged for the decision
    model's tokens at these prices, so at least one must be above zero.
    """
    return await routing.add_agent_recommender(body)


@recommenders_router.patch("/{recommender_id}")
async def update_agent_recommender(
    routing: RoutingServiceDep, recommender_id: uuid.UUID, body: AgentRecommenderUpdate
) -> AgentRecommenderPublic:
    """Change a recommender, or switch it on in place of the one that is.

    An omitted `api_key` keeps the stored key while the provider and the API base stay the same.
    """
    return await routing.update_agent_recommender(recommender_id, body)


@recommenders_router.delete("/{recommender_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_agent_recommender(routing: RoutingServiceDep, recommender_id: uuid.UUID) -> None:
    """Remove a recommender and its key. With none left, `agent_recommender_model` answers again."""
    await routing.remove_agent_recommender(recommender_id)
