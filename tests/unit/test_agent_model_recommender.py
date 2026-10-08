"""Unit tests for the pure halves of the agent model recommender.

The question the core builds for the decision model, and how it reads the
answer into the port's terms. The call in between is the adapter's, covered
by ``test_agent_model_recommender_adapter.py``.
"""

from decimal import Decimal

import pytest

from gateway.core.config import DEFAULT_AGENT_MODEL_CANDIDATES, validate_agent_recommender
from gateway.exceptions.routing_exceptions import UnreadableRecommendationError
from gateway.ports.agent_model_recommender_port import RecommendationUsage, SubagentSpawn
from gateway.schemas.inference import DecisionResponse
from gateway.services.routing.recommend import (
    MAX_TASK_CHARS,
    RECOMMENDATION_QUESTION,
    build_decision_request,
    recommendation_from_decision,
)

CANDIDATES = {"haiku": "small work", "sonnet": "medium work", "opus": "hard work"}


def _spawn(**overrides: object) -> SubagentSpawn:
    fields: dict[str, object] = {
        "harness": "claude-code",
        "session_id": "s1",
        "tool_use_id": "t1",
        "agent_type": "Explore",
        "description": "List files",
        "prompt": "List the files under src/ and say what each module does.",
        "parent_model": "claude-opus-5",
        "requested_model": None,
    }
    fields.update(overrides)
    return SubagentSpawn(**fields)  # type: ignore[arg-type]


def _answer(**fields: object) -> DecisionResponse:
    return DecisionResponse.model_validate(
        {"model": "jev-1.13.0", "answers": {RECOMMENDATION_QUESTION: {"type": "choice", **fields}}}
    )


def test_builds_one_choice_question_over_the_candidates() -> None:
    decision = build_decision_request(_spawn(), decision_model="typesafe:jev-latest", candidates=CANDIDATES)

    assert decision.model == "typesafe:jev-latest"
    assert decision.user is None
    assert list(decision.questions) == [RECOMMENDATION_QUESTION]
    question = decision.questions[RECOMMENDATION_QUESTION]
    assert question.type == "choice"
    assert question.criteria == CANDIDATES
    assert isinstance(decision.state, str)
    assert "Subagent type: Explore" in decision.state
    assert "Task description: List files" in decision.state
    assert "Parent model: claude-opus-5" in decision.state
    assert "List the files under src/" in decision.state


def test_the_harnesss_model_request_stays_out_of_the_state() -> None:
    decision = build_decision_request(
        _spawn(requested_model="requested-xyz"), decision_model="typesafe:jev-latest", candidates=CANDIDATES
    )

    assert "requested-xyz" not in str(decision.state)


def test_a_long_task_is_shortened_and_says_so() -> None:
    prompt = "x" * (MAX_TASK_CHARS + 500)

    decision = build_decision_request(_spawn(prompt=prompt), decision_model="m", candidates=CANDIDATES)

    state = str(decision.state)
    assert "x" * MAX_TASK_CHARS in state
    assert "x" * (MAX_TASK_CHARS + 1) not in state
    assert f"shortened to its first {MAX_TASK_CHARS} characters" in state


def test_reads_the_chosen_candidate_and_its_probability() -> None:
    answer = _answer(choice="sonnet", probabilities={"haiku": 0.2, "sonnet": 0.7, "opus": 0.1}, confidence=0.7)

    recommendation = recommendation_from_decision(answer, candidates=CANDIDATES)

    assert recommendation.model == "sonnet"
    assert recommendation.reason == "jev-1.13.0 chose sonnet with 70%"
    assert recommendation.probabilities == {"haiku": 0.2, "sonnet": 0.7, "opus": 0.1}


def test_a_choice_without_probabilities_still_recommends() -> None:
    recommendation = recommendation_from_decision(_answer(choice="haiku"), candidates=CANDIDATES)

    assert recommendation.model == "haiku"
    assert recommendation.reason == "jev-1.13.0 chose haiku"
    assert recommendation.probabilities is None


def test_an_answer_without_usage_consumed_nothing_it_can_name() -> None:
    recommendation = recommendation_from_decision(_answer(choice="haiku"), candidates=CANDIDATES)

    assert recommendation.usage == RecommendationUsage()


def test_what_the_answer_consumed_and_the_upstreams_own_charge_come_along() -> None:
    response = DecisionResponse.model_validate(
        {
            "model": "jev-1.13.0",
            "answers": {RECOMMENDATION_QUESTION: {"type": "choice", "choice": "opus"}},
            "usage": {"input_tokens": 800, "output_tokens": 3, "cost": 0.0042},
        }
    )

    recommendation = recommendation_from_decision(response, candidates=CANDIDATES)

    assert recommendation.usage == RecommendationUsage(input_tokens=800, output_tokens=3, charge=Decimal("0.0042"))


def test_an_answer_without_a_choice_is_unreadable() -> None:
    with pytest.raises(UnreadableRecommendationError):
        recommendation_from_decision(_answer(noul=0.5), candidates=CANDIDATES)


def test_a_choice_outside_the_candidates_is_unreadable() -> None:
    with pytest.raises(UnreadableRecommendationError):
        recommendation_from_decision(_answer(choice="gpt-5"), candidates=CANDIDATES)


def test_an_answer_to_another_question_is_unreadable() -> None:
    response = DecisionResponse.model_validate(
        {"model": "jev-1.13.0", "answers": {"other": {"type": "choice", "choice": "haiku"}}}
    )

    with pytest.raises(UnreadableRecommendationError):
        recommendation_from_decision(response, candidates=CANDIDATES)


@pytest.mark.parametrize(
    ("model", "candidates"),
    [
        ("typesafe:jev-latest", DEFAULT_AGENT_MODEL_CANDIDATES),
        ("openrouter:typesafe/jev-1.13", {"small": None, "large": "the hard cases"}),
    ],
)
def test_valid_recommender_settings_are_accepted(model: str, candidates: dict[str, str | None]) -> None:
    validate_agent_recommender(model, candidates)


@pytest.mark.parametrize(
    ("model", "candidates", "message"),
    [
        ("jev-latest", DEFAULT_AGENT_MODEL_CANDIDATES, "'<provider>:<model>'"),
        ("typesafe:", DEFAULT_AGENT_MODEL_CANDIDATES, "'<provider>:<model>'"),
        ("typesafe:jev-latest", {"haiku": None}, "at least two"),
        ("typesafe:jev-latest", {"haiku": None, " ": None}, "empty model name"),
        ("typesafe:jev-latest", {f"model-{i}": None for i in range(256)}, "at most 255"),
    ],
)
def test_invalid_recommender_settings_are_refused(model: str, candidates: dict[str, str | None], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_agent_recommender(model, candidates)
