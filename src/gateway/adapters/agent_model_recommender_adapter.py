"""The core recommender: one choice question to the decision model the deployment configured.

``agent_recommender_model`` is a ``decision_providers`` selector, so the model
may be TypeSafe's Jev, one behind OpenRouter or a local server, whichever the
operator pointed it at. The adapter owns no price: its quote carries none, and
its usage reports the charge the upstream itself stated, where it stated one.
"""

from collections.abc import Mapping

from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.exceptions.routing_exceptions import (
    RecommendationFailedError,
    RecommenderNotConfiguredError,
    UnreadableRecommendationError,
)
from gateway.ports.agent_model_recommender_port import (
    AgentModelRecommenderPort,
    ModelRecommendation,
    Quote,
    SubagentSpawn,
)
from gateway.schemas.inference import DecisionResponse
from gateway.services.inference import (
    DecisionProvider,
    DecisionProviderError,
    UnknownDecisionProviderError,
    decision_body,
    request_decision,
    resolve_decision_provider,
)
from gateway.services.routing import build_decision_request, recommendation_from_decision


class DecisionProviderRecommender(AgentModelRecommenderPort):
    """``AgentModelRecommenderPort`` adapter that asks the configured decision model.

    Holds no per-request state, so one instance serves every request.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    def quote(self) -> Quote:
        provider, model = self._resolve()
        return Quote(provider=provider.name, model=model)

    async def recommend(self, spawn: SubagentSpawn, *, candidates: Mapping[str, str | None]) -> ModelRecommendation:
        provider, model = self._resolve()
        request = build_decision_request(
            spawn, decision_model=self._config.agent_recommender_model, candidates=candidates
        )
        try:
            answer = DecisionResponse.model_validate(await request_decision(provider, decision_body(request, model)))
        except DecisionProviderError as exc:
            raise RecommendationFailedError(str(exc), upstream_status=exc.status_code) from exc
        except ValidationError as exc:
            # The error's own text quotes the upstream body, so only its size is kept.
            msg = f"{provider.provider} decisions returned an answer with {exc.error_count()} invalid field(s)"
            raise UnreadableRecommendationError(msg) from exc
        return recommendation_from_decision(answer, candidates=candidates)

    def _resolve(self) -> tuple[DecisionProvider, str]:
        try:
            return resolve_decision_provider(self._config, self._config.agent_recommender_model)
        except UnknownDecisionProviderError as exc:
            raise RecommenderNotConfiguredError(str(exc)) from exc
