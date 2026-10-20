"""The core recommender: one choice question to the decision model the deployment configured.

``agent_recommender_model`` is a ``decision_providers`` selector, so the model
may be TypeSafe's Jev, one behind OpenRouter or a local server, whichever the
operator pointed it at. The adapter owns no price: its quote carries none, and
its usage reports the charge the upstream itself stated, where it stated one.
"""

import asyncio
from collections.abc import Mapping
from typing import Any

from fastapi import status
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

# The otari-router plugin stops waiting after three seconds and starts the
# subagent on its usual model. Otari's own work before and after the call
# (authentication, the hold, the charge) comes out of those three seconds too,
# so the call ends early enough for its answer to arrive in time.
TIMEOUT_SECONDS = 2.5


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
        answer = await _ask(provider, decision_body(request, model))
        return recommendation_from_decision(answer, candidates=candidates)

    def _resolve(self) -> tuple[DecisionProvider, str]:
        try:
            return resolve_decision_provider(self._config, self._config.agent_recommender_model)
        except UnknownDecisionProviderError as exc:
            raise RecommenderNotConfiguredError(str(exc)) from exc


async def _ask(provider: DecisionProvider, body: dict[str, Any]) -> DecisionResponse:
    """Put the question, in the time the plugin waits, with no upstream text in what fails."""
    try:
        # httpx's timeout bounds each phase of the call, not the call.
        async with asyncio.timeout(TIMEOUT_SECONDS):
            raw = await request_decision(provider, body)
        return DecisionResponse.model_validate(raw)
    except TimeoutError:
        raise RecommendationFailedError("The recommender did not answer in time") from None
    except DecisionProviderError as exc:
        # The error's own text names the vendor, so only the status is kept. A
        # rate limit is the one status worth passing on: anything else is this
        # deployment's problem, not the caller's request.
        upstream = exc.status_code
        message = "The recommender could not be reached" if upstream is None else f"The recommender failed ({upstream})"
        rate_limited = upstream == status.HTTP_429_TOO_MANY_REQUESTS
        raise RecommendationFailedError(message, upstream_status=upstream if rate_limited else None) from None
    except ValidationError as exc:
        # The error's own text quotes the upstream body, so only its size is kept.
        msg = f"The recommender's answer had {exc.error_count()} invalid field(s)"
        raise UnreadableRecommendationError(msg) from None
