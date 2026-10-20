"""Unit tests for the core recommender adapter: the configured decision model, asked one question.

The upstream call is stubbed at ``request_decision``, so these cover what the
adapter sends, what it makes of the answer, and how an upstream's failure
reaches the port's caller.
"""

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from gateway.adapters.agent_model_recommender_adapter import DecisionProviderRecommender
from gateway.core.config import GatewayConfig
from gateway.exceptions.routing_exceptions import (
    RecommendationFailedError,
    RecommenderNotConfiguredError,
    UnreadableRecommendationError,
)
from gateway.ports.agent_model_recommender_port import Quote, RecommendationUsage, SubagentSpawn
from gateway.services.inference import DecisionProviderError
from gateway.services.routing.recommend import RECOMMENDATION_QUESTION

CANDIDATES = {"haiku": "small work", "sonnet": "medium work", "opus": "hard work"}
SPAWN = SubagentSpawn(
    harness="claude-code",
    session_id="s1",
    tool_use_id="t1",
    agent_type="Explore",
    description="List files",
    prompt="List the files under src/ and say what each module does.",
    parent_model="claude-opus-5",
    requested_model="requested-xyz",
)
ANSWER: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {
        RECOMMENDATION_QUESTION: {
            "type": "choice",
            "choice": "sonnet",
            "probabilities": {"haiku": 0.2, "sonnet": 0.7, "opus": 0.1},
        }
    },
    "usage": {"input_tokens": 800, "output_tokens": 0},
}


def _recommender(**overrides: Any) -> DecisionProviderRecommender:
    settings: dict[str, Any] = {
        "decision_providers": {"typesafe": {"api_key": "ts-secret"}},
        "agent_recommender_model": "typesafe:jev-latest",
    }
    settings.update(overrides)
    return DecisionProviderRecommender(GatewayConfig(**settings))


def _mock_decision(answer: dict[str, Any] | None = None, *, side_effect: Exception | None = None) -> Any:
    mock = AsyncMock(return_value=answer if answer is not None else ANSWER, side_effect=side_effect)
    return patch("gateway.adapters.agent_model_recommender_adapter.request_decision", mock)


def test_the_quote_is_the_configured_model_at_no_price_of_its_own() -> None:
    assert _recommender().quote() == Quote(provider="typesafe", model="jev-latest", charge=None)


def test_a_selector_naming_no_configured_provider_is_not_configured() -> None:
    recommender = _recommender(agent_recommender_model="nope:jev")

    with pytest.raises(RecommenderNotConfiguredError, match="decision provider"):
        recommender.quote()


@pytest.mark.asyncio
async def test_recommend_puts_one_question_to_the_configured_model() -> None:
    with _mock_decision() as mock:
        recommendation = await _recommender().recommend(SPAWN, candidates=CANDIDATES)

    assert recommendation.model == "sonnet"
    assert recommendation.reason == "jev-1.13.0 chose sonnet with 70%"
    assert recommendation.usage == RecommendationUsage(input_tokens=800, output_tokens=0, charge=None)
    provider, sent = mock.await_args.args
    assert provider.name == "typesafe"
    assert sent["model"] == "jev-latest"
    assert list(sent["questions"]) == [RECOMMENDATION_QUESTION]
    assert sent["questions"][RECOMMENDATION_QUESTION]["criteria"] == CANDIDATES
    assert "Subagent type: Explore" in sent["state"]
    assert "requested-xyz" not in sent["state"]
    assert "user" not in sent


@pytest.mark.asyncio
async def test_the_upstreams_own_charge_passes_through() -> None:
    answer = {**ANSWER, "usage": {"input_tokens": 800, "output_tokens": 0, "cost": 0.0042}}
    with _mock_decision(answer):
        recommendation = await _recommender().recommend(SPAWN, candidates=CANDIDATES)

    assert recommendation.usage.charge == Decimal("0.0042")


@pytest.mark.asyncio
async def test_recommend_on_an_unconfigured_provider_is_not_configured() -> None:
    with _mock_decision() as mock, pytest.raises(RecommenderNotConfiguredError):
        await _recommender(agent_recommender_model="nope:jev").recommend(SPAWN, candidates=CANDIDATES)

    mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_upstream_failure_keeps_its_status_and_names_no_body() -> None:
    with (
        _mock_decision(side_effect=DecisionProviderError("typesafe decisions returned HTTP 429", status_code=429)),
        pytest.raises(RecommendationFailedError) as failure,
    ):
        await _recommender().recommend(SPAWN, candidates=CANDIDATES)

    assert failure.value.upstream_status == 429


@pytest.mark.asyncio
async def test_an_answer_the_schema_refuses_is_unreadable() -> None:
    answer = {"model": "jev", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 1, "cost": -1.0}}
    with _mock_decision(answer), pytest.raises(UnreadableRecommendationError, match="invalid field"):
        await _recommender().recommend(SPAWN, candidates=CANDIDATES)


@pytest.mark.asyncio
async def test_a_pick_outside_the_candidates_is_unreadable() -> None:
    answer = {"model": "jev", "answers": {RECOMMENDATION_QUESTION: {"type": "choice", "choice": "gpt-5"}}}
    with _mock_decision(answer), pytest.raises(UnreadableRecommendationError):
        await _recommender().recommend(SPAWN, candidates=CANDIDATES)
