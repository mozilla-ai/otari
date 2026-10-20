"""The core recommender: one choice question to the decision model the deployment configured.

The decision model is the recommender row the operator switched on in the
dashboard, priced at that row's rates: the quote holds the most one call can
cost, and the call is charged for the tokens it used. While no row has been
added, ``agent_recommender_model`` answers instead, a ``decision_providers``
selector that quotes no price of its own and reports the charge the upstream
stated, where it stated one. With rows added and every one switched off,
nothing answers.

An edition sets a price of its own by binding a recommender that wraps this
one and changes the charges it quotes and reports, reading
:attr:`DecisionProviderRecommender.quoted_recommender_id` for the row it quoted.
"""

import asyncio
import uuid
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
    RecommendationUsage,
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
from gateway.services.routing import (
    RecommenderConfig,
    RecommenderSnapshot,
    build_decision_request,
    recommendation_from_decision,
)

NOT_CONFIGURED = (
    "No subagent recommender is switched on. Switch one on in the dashboard, or remove them all "
    "to use agent_recommender_model."
)

# The otari-router plugin stops waiting after three seconds and starts the
# subagent on its usual model. Otari's own work before and after the call
# (authentication, the hold, the charge) comes out of those three seconds too,
# so the call ends early enough for its answer to arrive in time.
TIMEOUT_SECONDS = 2.5


def _row_target(row: RecommenderConfig) -> tuple[DecisionProvider, str]:
    """Where to send a row's question, through the resolution a ``decision_providers`` entry takes."""
    entry: dict[str, Any] = {"provider": row.provider, "api_key": row.api_key, "timeout": TIMEOUT_SECONDS}
    if row.api_base:
        entry["api_base"] = row.api_base
    # Unvalidated, so it reads no environment, and every other field holds its default.
    view = GatewayConfig.model_construct(decision_providers={row.provider: entry})
    return resolve_decision_provider(view, f"{row.provider}:{row.model}")


class DecisionProviderRecommender(AgentModelRecommenderPort):
    """``AgentModelRecommenderPort`` adapter that asks the deployment's decision model.

    One instance per request: it keeps what it quoted, so a change the operator
    makes before the call cannot send it elsewhere or reprice it.
    """

    def __init__(self, config: GatewayConfig, snapshot: RecommenderSnapshot) -> None:
        self._config = config
        self._snapshot = snapshot
        self._row: RecommenderConfig | None = None
        self._target: tuple[DecisionProvider, str] | None = None

    @property
    def quoted_recommender_id(self) -> uuid.UUID | None:
        """The row the quote named, or None before a quote or where ``agent_recommender_model`` answered."""
        return self._row.id if self._row is not None else None

    def quote(self) -> Quote:
        self._row, self._target = self._pick()
        if self._row is None:
            provider, model = self._target
            return Quote(provider=provider.name, model=model)
        return Quote(provider=self._row.provider, model=self._row.model, charge=self._row.price.hold())

    async def recommend(self, spawn: SubagentSpawn, *, candidates: Mapping[str, str | None]) -> ModelRecommendation:
        if self._target is None:
            self._row, self._target = self._pick()
        provider, model = self._target
        row = self._row
        request = build_decision_request(spawn, decision_model=f"{provider.name}:{model}", candidates=candidates)
        answer = await _ask(provider, decision_body(request, model))
        recommendation = recommendation_from_decision(answer, candidates=candidates)
        if row is None:
            return recommendation
        tokens = recommendation.usage
        # At the row's rates, never the upstream's reported cost. An answer with
        # no token counts leaves the charge to the scaffold, which charges the
        # hold, so a provider that reports no usage cannot make a call free.
        counted = answer.usage is not None and tokens.input_tokens + tokens.output_tokens > 0
        charge = (
            row.price.charge(input_tokens=tokens.input_tokens, output_tokens=tokens.output_tokens) if counted else None
        )
        return ModelRecommendation(
            model=recommendation.model,
            reason=recommendation.reason,
            probabilities=recommendation.probabilities,
            usage=RecommendationUsage(
                input_tokens=tokens.input_tokens, output_tokens=tokens.output_tokens, charge=charge
            ),
        )

    def _pick(self) -> tuple[RecommenderConfig | None, tuple[DecisionProvider, str]]:
        """The row that answers and where to ask it, or no row and the configured selector's provider."""
        if self._snapshot.defers_to_config:
            try:
                return None, resolve_decision_provider(self._config, self._config.agent_recommender_model)
            except UnknownDecisionProviderError as exc:
                raise RecommenderNotConfiguredError(str(exc)) from exc
        row = self._snapshot.current()
        if row is None:
            raise RecommenderNotConfiguredError(NOT_CONFIGURED)
        return row, _row_target(row)


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
