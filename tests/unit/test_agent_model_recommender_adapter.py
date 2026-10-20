"""Unit tests for the core recommender adapter: the deployment's decision model, asked one question.

The upstream call is stubbed at ``request_decision``, so these cover what the
adapter sends, what it makes of the answer, how an upstream's failure reaches
the port's caller, and which decision model answers: the row switched on,
priced at its rates, or ``agent_recommender_model`` while no row is added.
"""

import asyncio
import uuid
from collections.abc import Sequence
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
from gateway.models.routing import AgentRecommender
from gateway.ports.agent_model_recommender_port import Quote, RecommendationUsage, SubagentSpawn
from gateway.services.inference import DecisionProviderError
from gateway.services.routing import RecommenderSnapshot
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


ROW_ID = uuid.UUID("7f1c2a9e-0000-4000-8000-000000000001")


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A row's key is stored encrypted, so reading one back needs the deployment's secret key."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _row(**overrides: Any) -> AgentRecommender:
    fields: dict[str, Any] = {
        "id": ROW_ID,
        "provider": "openrouter",
        "model": "openai/gpt-5-mini",
        "api_base": None,
        "encrypted_api_key": encrypt_secret("or-secret"),
        "api_key_last4": "cret",
        "input_price_per_million": Decimal("2"),
        "output_price_per_million": Decimal("10"),
        "enabled": True,
    }
    fields.update(overrides)
    return AgentRecommender(**fields)


def _snapshot(rows: Sequence[AgentRecommender] | None = None) -> RecommenderSnapshot:
    snapshot = RecommenderSnapshot()
    if rows is not None:
        snapshot.replace(rows)
    return snapshot


def _recommender(snapshot: RecommenderSnapshot | None = None, **overrides: Any) -> DecisionProviderRecommender:
    settings: dict[str, Any] = {
        "decision_providers": {"typesafe": {"api_key": "ts-secret"}},
        "agent_recommender_model": "typesafe:jev-latest",
    }
    settings.update(overrides)
    return DecisionProviderRecommender(GatewayConfig(**settings), snapshot or _snapshot())


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
async def test_a_rate_limit_keeps_its_status_and_names_no_vendor() -> None:
    with (
        _mock_decision(side_effect=DecisionProviderError("typesafe decisions returned HTTP 429", status_code=429)),
        pytest.raises(RecommendationFailedError) as failure,
    ):
        await _recommender().recommend(SPAWN, candidates=CANDIDATES)

    assert failure.value.upstream_status == 429
    assert str(failure.value) == "The recommender failed (429)"


@pytest.mark.asyncio
async def test_any_other_upstream_status_is_the_deployments_problem_not_the_callers() -> None:
    """A rejected key upstream is the operator's to fix, so it is not passed on as the caller's credential."""
    with (
        _mock_decision(side_effect=DecisionProviderError("typesafe decisions returned HTTP 401", status_code=401)),
        pytest.raises(RecommendationFailedError) as failure,
    ):
        await _recommender().recommend(SPAWN, candidates=CANDIDATES)

    assert failure.value.upstream_status is None
    assert "typesafe" not in str(failure.value)


@pytest.mark.asyncio
async def test_an_upstream_slower_than_the_plugin_waits_fails() -> None:
    async def slow(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(1)
        return ANSWER

    with (
        patch("gateway.adapters.agent_model_recommender_adapter.TIMEOUT_SECONDS", 0.01),
        patch("gateway.adapters.agent_model_recommender_adapter.request_decision", slow),
        pytest.raises(RecommendationFailedError, match="in time"),
    ):
        await _recommender().recommend(SPAWN, candidates=CANDIDATES)


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


def test_with_no_row_added_the_configured_model_answers() -> None:
    recommender = _recommender(_snapshot([]))

    assert recommender.quote() == Quote(provider="typesafe", model="jev-latest", charge=None)
    assert recommender.quoted_recommender_id is None


def test_the_row_switched_on_answers_at_its_hold() -> None:
    recommender = _recommender(_snapshot([_row(enabled=False, id=uuid.uuid4()), _row()]))

    quote = recommender.quote()

    # 65,536 input tokens at $2 and 1,024 output at $10 per million, rounded up.
    assert quote == Quote(provider="openrouter", model="openai/gpt-5-mini", charge=Decimal("0.141312"))
    assert recommender.quoted_recommender_id == ROW_ID


def test_with_every_row_switched_off_nothing_answers() -> None:
    """Rows present and all off: the configured model is not a fallback for a choice the operator made."""
    recommender = _recommender(_snapshot([_row(enabled=False)]))

    with pytest.raises(RecommenderNotConfiguredError, match="switched on"):
        recommender.quote()


def test_a_row_whose_key_cannot_be_read_answers_nothing() -> None:
    recommender = _recommender(_snapshot([_row(encrypted_api_key="not-a-fernet-token")]))

    with pytest.raises(RecommenderNotConfiguredError):
        recommender.quote()


def test_nothing_answers_while_a_load_has_not_landed() -> None:
    snapshot = _snapshot()
    snapshot.await_load()

    with pytest.raises(RecommenderNotConfiguredError):
        _recommender(snapshot).quote()


@pytest.mark.asyncio
async def test_the_row_is_asked_on_its_own_key_and_charged_its_tokens_at_its_rates() -> None:
    recommender = _recommender(_snapshot([_row(api_base="https://proxy.example/api")]))
    recommender.quote()
    answer = {**ANSWER, "usage": {"input_tokens": 1_000, "output_tokens": 10, "cost": 9.99}}

    with _mock_decision(answer) as mock:
        recommendation = await recommender.recommend(SPAWN, candidates=CANDIDATES)

    provider, sent = mock.await_args.args
    assert (provider.provider, provider.api_key) == ("openrouter", "or-secret")
    assert provider.url == "https://proxy.example/api/alpha/decisions"
    assert sent["model"] == "openai/gpt-5-mini"
    # 1,000 at $2 and 10 at $10 per million: the upstream's own cost is not what is charged.
    assert recommendation.usage == RecommendationUsage(input_tokens=1_000, output_tokens=10, charge=Decimal("0.0021"))
    assert recommendation.reason == "jev-1.13.0 chose sonnet with 70%"


@pytest.mark.asyncio
async def test_an_answer_with_no_usage_leaves_the_hold_to_be_charged() -> None:
    recommender = _recommender(_snapshot([_row()]))
    recommender.quote()
    answer = {key: value for key, value in ANSWER.items() if key != "usage"}

    with _mock_decision(answer):
        recommendation = await recommender.recommend(SPAWN, candidates=CANDIDATES)

    assert recommendation.usage.charge is None


@pytest.mark.asyncio
async def test_the_row_quoted_answers_even_when_another_is_switched_on_before_the_call() -> None:
    snapshot = _snapshot([_row()])
    recommender = _recommender(snapshot)
    recommender.quote()
    snapshot.replace([_row(enabled=False), _row(id=uuid.uuid4(), provider="typesafe", model="jev-latest")])

    with _mock_decision() as mock:
        await recommender.recommend(SPAWN, candidates=CANDIDATES)

    provider, sent = mock.await_args.args
    assert (provider.provider, sent["model"]) == ("openrouter", "openai/gpt-5-mini")
    assert recommender.quoted_recommender_id == ROW_ID
