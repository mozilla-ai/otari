"""The core recommender: one choice question to the subagent router this deployment switched on.

The stored subagent router switched on (``services/routing``) answers at the
token rates stored with it, so the quote carries the most one call can cost
and the usage reports what the tokens came to, under the label
``otari:subagent-router``. With none switched on, nothing answers.

One instance per request: it keeps the router it quoted, so a change the
operator makes before the call cannot send it elsewhere or reprice it. What
the caller can see names no vendor: an error says the recommender failed, not
which provider did. The task in the spawn is sent to the decision model and
nowhere else; nothing here logs it.
"""

import asyncio
from collections.abc import Mapping

from pydantic import ValidationError

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
    decision_body,
    decision_provider_for,
    request_decision,
)
from gateway.services.routing import (
    ROUTER_LABEL_MODEL,
    ROUTER_LABEL_PROVIDER,
    StoredRecommender,
    active_agent_recommender,
    build_decision_request,
    recommendation_from_decision,
)

NOT_CONFIGURED = "No subagent recommender is configured on this deployment."

# The otari-router plugin stops waiting after three seconds and starts the
# subagent on its usual model. The gateway's own work before and after the
# call (authentication, the hold, the charge) comes out of those three seconds
# too, so the call ends early enough for its answer to arrive in time. A call
# that fails is not charged.
TIMEOUT_SECONDS = 2.5


class DecisionProviderRecommender(AgentModelRecommenderPort):
    """``AgentModelRecommenderPort`` adapter that asks the deployment's subagent router."""

    def __init__(self) -> None:
        self._stored: StoredRecommender | None = None

    @property
    def quoted(self) -> StoredRecommender | None:
        """The stored router this instance quoted, or None before a quote."""
        return self._stored

    def quote(self) -> Quote:
        stored = self._router()
        return Quote(
            provider=ROUTER_LABEL_PROVIDER,
            model=ROUTER_LABEL_MODEL,
            charge=stored.price.hold(),
            served_by=stored.provider,
        )

    async def recommend(self, spawn: SubagentSpawn, *, candidates: Mapping[str, str | None]) -> ModelRecommendation:
        stored = self._router()
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

    def _router(self) -> StoredRecommender:
        """The router switched on when this instance first asked, kept for the rest of the request."""
        if self._stored is None:
            stored = active_agent_recommender()
            if stored is None:
                raise RecommenderNotConfiguredError(NOT_CONFIGURED)
            self._stored = stored
        return self._stored

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
