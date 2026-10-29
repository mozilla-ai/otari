"""The core recommender: one choice question to the decision model this deployment chose.

Which model that is follows the stored subagent routers
(``services/routing``): the one switched on answers at the token rates stored
with it, so the quote carries the most one call can cost and the usage reports
what the tokens came to, under the label ``otari:subagent-router``. With none
stored, ``agent_recommender_model``, a ``decision_providers`` selector, answers
at no price of its own, and the usage reports the charge the upstream itself
stated, where it stated one.

One instance per request: it keeps the router it quoted, so a change the
operator makes before the call cannot send it elsewhere or reprice it. What
the caller can see names no vendor: an error says the recommender failed, not
which provider did. The task in the spawn is sent to the decision model and
nowhere else; nothing here logs it.
"""

import asyncio
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
    RecommendationUsage,
    SubagentSpawn,
)
from gateway.schemas.inference import DecisionResponse
from gateway.services.inference import (
    DecisionProvider,
    DecisionProviderError,
    UnknownDecisionProviderError,
    decision_body,
    decision_provider_for,
    request_decision,
    resolve_decision_provider,
)
from gateway.services.routing import (
    RecommenderSource,
    StoredRecommender,
    active_agent_recommender,
    build_decision_request,
    recommendation_from_decision,
)

NOT_CONFIGURED = "No subagent recommender is configured on this deployment."

# What a stored router's call is recorded as: the usage row, and the label an
# API key's allow-list admits, never name the decision model behind it.
ROUTER_LABEL_PROVIDER = "otari"
ROUTER_LABEL_MODEL = "subagent-router"

# The otari-router plugin stops waiting after three seconds and starts the
# subagent on its usual model. The gateway's own work before and after the
# call (authentication, the hold, the charge) comes out of those three seconds
# too, so the call ends early enough for its answer to arrive in time. A call
# that fails is not charged.
TIMEOUT_SECONDS = 2.5


class DecisionProviderRecommender(AgentModelRecommenderPort):
    """``AgentModelRecommenderPort`` adapter that asks the deployment's decision model."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._stored: StoredRecommender | None = None
        self._configured: tuple[DecisionProvider, str] | None = None

    @property
    def quoted(self) -> StoredRecommender | None:
        """The stored router this instance quoted, or None where the configured model answers."""
        return self._stored

    def quote(self) -> Quote:
        active = active_agent_recommender()
        if active.source is RecommenderSource.STORED and active.stored is not None:
            self._stored = active.stored
            return Quote(
                provider=ROUTER_LABEL_PROVIDER,
                model=ROUTER_LABEL_MODEL,
                charge=active.stored.price.hold(),
                served_by=active.stored.provider,
            )
        if active.source is RecommenderSource.NONE:
            raise RecommenderNotConfiguredError(NOT_CONFIGURED)
        provider, model = self._resolve_configured()
        return Quote(provider=provider.name, model=model)

    async def recommend(self, spawn: SubagentSpawn, *, candidates: Mapping[str, str | None]) -> ModelRecommendation:
        if self._stored is None and self._configured is None:
            self.quote()
        if self._stored is not None:
            return await self._ask_stored(self._stored, spawn, candidates)
        provider, model = self._resolve_configured()
        request = build_decision_request(
            spawn, decision_model=self._config.agent_recommender_model, candidates=candidates
        )
        answer = await self._ask(provider, decision_body(request, model))
        return recommendation_from_decision(answer, candidates=candidates)

    async def _ask_stored(
        self, stored: StoredRecommender, spawn: SubagentSpawn, candidates: Mapping[str, str | None]
    ) -> ModelRecommendation:
        provider = decision_provider_for(
            stored.provider,
            stored.provider,
            api_key=stored.api_key,
            api_base=stored.api_base,
            timeout_s=TIMEOUT_SECONDS,
        )
        request = build_decision_request(
            spawn, decision_model=f"{stored.provider}:{stored.model}", candidates=candidates
        )
        answer = await self._ask(provider, decision_body(request, stored.model))
        recommendation = recommendation_from_decision(answer, candidates=candidates)
        tokens = recommendation.usage
        if tokens.input_tokens < 0 or tokens.output_tokens < 0:
            raise UnreadableRecommendationError("The recommender's answer reported a negative token count")
        # At the stored rates, never the upstream's reported cost. An answer
        # with no token counts leaves the charge to the scaffold, which charges
        # the hold, so a provider that reports no usage cannot make a
        # recommendation free.
        counted = answer.usage is not None and tokens.input_tokens + tokens.output_tokens > 0
        charge = (
            stored.price.charge(input_tokens=tokens.input_tokens, output_tokens=tokens.output_tokens)
            if counted
            else None
        )
        return ModelRecommendation(
            model=recommendation.model,
            reason=recommendation.reason,
            probabilities=recommendation.probabilities,
            usage=RecommendationUsage(
                input_tokens=tokens.input_tokens, output_tokens=tokens.output_tokens, charge=charge
            ),
        )

    @staticmethod
    async def _ask(provider: DecisionProvider, body: dict[str, object]) -> DecisionResponse:
        """Put the question, within the deadline, and read the answer; the errors name no vendor."""
        try:
            # httpx's timeout bounds each phase of the call, not the call.
            async with asyncio.timeout(TIMEOUT_SECONDS):
                raw = await request_decision(provider, body)
            return DecisionResponse.model_validate(raw)
        except TimeoutError:
            raise RecommendationFailedError("The recommender did not answer in time") from None
        except DecisionProviderError as exc:
            # The error's own text names the vendor, so only the status is kept.
            upstream = exc.status_code
            message = (
                "The recommender could not be reached" if upstream is None else f"The recommender failed ({upstream})"
            )
            raise RecommendationFailedError(message, upstream_status=upstream) from None
        except ValidationError as exc:
            # The error's own text quotes the upstream body, so only its size is kept.
            msg = f"The recommender's answer was unreadable ({exc.error_count()} invalid field(s))"
            raise UnreadableRecommendationError(msg) from None

    def _resolve_configured(self) -> tuple[DecisionProvider, str]:
        if self._configured is None:
            try:
                self._configured = resolve_decision_provider(self._config, self._config.agent_recommender_model)
            except UnknownDecisionProviderError as exc:
                raise RecommenderNotConfiguredError(str(exc)) from exc
        return self._configured
