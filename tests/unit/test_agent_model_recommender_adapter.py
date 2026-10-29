"""Unit tests for the core recommender adapter: the deployment's decision model, asked one question.

The upstream call is stubbed at ``request_decision``, so these cover what the
adapter sends, what it makes of the answer, how an upstream's failure reaches
the port's caller, and which decision model answers: the stored router
switched on at its rates, or the configured one at no price of its own.
"""

import asyncio
import uuid
from collections.abc import Iterator
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from gateway.adapters.agent_model_recommender_adapter import NOT_CONFIGURED, DecisionProviderRecommender
from gateway.core.config import GatewayConfig
from gateway.exceptions.routing_exceptions import (
    RecommendationFailedError,
    RecommenderNotConfiguredError,
    UnreadableRecommendationError,
)
from gateway.models.routing import AgentRecommender
from gateway.ports.agent_model_recommender_port import Quote, RecommendationUsage, SubagentSpawn
from gateway.services.inference import DecisionProviderError
from gateway.services.routing import apply_agent_recommenders, reset_agent_recommenders
from gateway.services.routing.recommend import RECOMMENDATION_QUESTION
from gateway.services.secret_box import encrypt_secret, generate_secret_key

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


STORED_KEY = "sk-stored-router-1234"


@pytest.fixture(autouse=True)
def _no_stored_routers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_agent_recommenders()
    yield
    reset_agent_recommenders()


def _stored(*, enabled: bool = True, model: str = "stored-decider", api_base: str | None = None) -> AgentRecommender:
    return AgentRecommender(
        id=uuid.uuid4(),
        provider="openrouter",
        model=model,
        api_base=api_base,
        encrypted_api_key=encrypt_secret(STORED_KEY),
        api_key_last4="1234",
        input_price_per_million=Decimal("20"),
        output_price_per_million=Decimal("0"),
        enabled=enabled,
    )


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
    assert recommendation.reason == "sonnet with 70%"
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


# =============================================================================
# The stored router switched on
# =============================================================================


def test_the_quote_is_the_stored_router_at_the_most_a_call_can_cost() -> None:
    apply_agent_recommenders([_stored()])
    recommender = _recommender()

    quote = recommender.quote()

    # 65,536 input tokens at $20 per million, and nothing for output.
    assert quote == Quote(provider="otari", model="subagent-router", charge=Decimal("1.310720"), served_by="openrouter")
    assert recommender.quoted is not None
    assert recommender.quoted.model == "stored-decider"


def test_with_every_stored_router_off_nothing_answers() -> None:
    apply_agent_recommenders([_stored(enabled=False)])

    with pytest.raises(RecommenderNotConfiguredError, match=NOT_CONFIGURED):
        _recommender().quote()


@pytest.mark.asyncio
async def test_the_stored_router_is_asked_on_its_own_key_and_address_within_the_deadline() -> None:
    apply_agent_recommenders([_stored(api_base="https://proxy.example/api")])

    with _mock_decision() as mock:
        recommendation = await _recommender().recommend(SPAWN, candidates=CANDIDATES)

    assert recommendation.model == "sonnet"
    # 800 input tokens at $20 per million.
    assert recommendation.usage == RecommendationUsage(input_tokens=800, output_tokens=0, charge=Decimal("0.016000"))
    provider, sent = mock.await_args.args
    assert (provider.name, provider.provider, provider.api_key) == ("openrouter", "openrouter", STORED_KEY)
    assert provider.url == "https://proxy.example/api/alpha/decisions"
    assert provider.timeout_s == 2.5
    assert sent["model"] == "stored-decider"


@pytest.mark.asyncio
async def test_an_answer_with_no_token_counts_leaves_the_charge_to_the_hold() -> None:
    apply_agent_recommenders([_stored()])
    uncounted = {key: value for key, value in ANSWER.items() if key != "usage"}
    zero = {**ANSWER, "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.5}}

    with _mock_decision(uncounted):
        assert (await _recommender().recommend(SPAWN, candidates=CANDIDATES)).usage.charge is None
    with _mock_decision(zero):
        assert (await _recommender().recommend(SPAWN, candidates=CANDIDATES)).usage.charge is None


@pytest.mark.asyncio
async def test_a_negative_token_count_is_unreadable_rather_than_priced() -> None:
    apply_agent_recommenders([_stored()])
    answer = {**ANSWER, "usage": {"input_tokens": -1, "output_tokens": 2}}

    with _mock_decision(answer), pytest.raises(UnreadableRecommendationError, match="negative token count"):
        await _recommender().recommend(SPAWN, candidates=CANDIDATES)


@pytest.mark.asyncio
async def test_the_upstreams_own_cost_is_ignored_on_the_stored_path() -> None:
    apply_agent_recommenders([_stored()])
    answer = {**ANSWER, "usage": {"input_tokens": 800, "output_tokens": 0, "cost": 99.0}}

    with _mock_decision(answer):
        recommendation = await _recommender().recommend(SPAWN, candidates=CANDIDATES)

    assert recommendation.usage.charge == Decimal("0.016000")


@pytest.mark.asyncio
async def test_the_router_quoted_is_the_one_asked_after_a_switch() -> None:
    apply_agent_recommenders([_stored(model="quoted-decider")])
    recommender = _recommender()
    recommender.quote()
    apply_agent_recommenders([_stored(model="switched-on-later")])

    with _mock_decision() as mock:
        await recommender.recommend(SPAWN, candidates=CANDIDATES)

    assert mock.await_args.args[1]["model"] == "quoted-decider"


@pytest.mark.asyncio
async def test_a_slow_upstream_is_a_failure_that_names_no_vendor() -> None:
    apply_agent_recommenders([_stored()])

    async def slow(*_: object, **__: object) -> dict[str, Any]:
        await asyncio.sleep(1)
        return ANSWER

    with (
        patch("gateway.adapters.agent_model_recommender_adapter.TIMEOUT_SECONDS", 0.01),
        patch("gateway.adapters.agent_model_recommender_adapter.request_decision", slow),
        pytest.raises(RecommendationFailedError, match="did not answer in time"),
    ):
        await _recommender().recommend(SPAWN, candidates=CANDIDATES)


@pytest.mark.asyncio
async def test_an_upstream_failure_names_no_vendor_on_either_path() -> None:
    for rows in ([_stored()], []):
        apply_agent_recommenders(rows)
        with (
            _mock_decision(
                side_effect=DecisionProviderError("openrouter decisions returned HTTP 500", status_code=500)
            ),
            pytest.raises(RecommendationFailedError) as failure,
        ):
            await _recommender().recommend(SPAWN, candidates=CANDIDATES)
        assert str(failure.value) == "The recommender failed (500)"
        assert failure.value.upstream_status == 500
        with (
            _mock_decision(side_effect=DecisionProviderError("openrouter decisions could not be reached")),
            pytest.raises(RecommendationFailedError, match="could not be reached"),
        ):
            await _recommender().recommend(SPAWN, candidates=CANDIDATES)


@pytest.mark.asyncio
async def test_recommend_without_a_quote_quotes_first() -> None:
    apply_agent_recommenders([_stored()])
    recommender = _recommender()

    with _mock_decision() as mock:
        recommendation = await recommender.recommend(SPAWN, candidates=CANDIDATES)

    assert recommendation.usage.charge == Decimal("0.016000")
    assert mock.await_args.args[0].api_key == STORED_KEY
