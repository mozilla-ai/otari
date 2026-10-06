"""Unit tests for the pure halves of the agent model recommender.

The question it builds for the decision model, and how it reads the answer.
The call in between is the decisions scaffold's, covered by the route tests.
"""

import pytest

from gateway.core.config import DEFAULT_AGENT_MODEL_CANDIDATES, validate_agent_recommender
from gateway.schemas.inference import DecisionResponse
from gateway.schemas.routing import AgentModelRecommendationRequest
from gateway.services.routing.recommend import (
    MAX_TASK_CHARS,
    RECOMMENDATION_QUESTION,
    UnreadableRecommendationError,
    build_decision_request,
    recommendation_from_decision,
)

CANDIDATES = {"haiku": "small work", "sonnet": "medium work", "opus": "hard work"}


def _request(**overrides: object) -> AgentModelRecommendationRequest:
    fields: dict[str, object] = {
        "harness": "claude-code",
        "session_id": "s1",
        "tool_use_id": "t1",
        "agent_type": "Explore",
        "description": "List files",
        "prompt": "List the files under src/ and say what each module does.",
        "parent_model": "claude-opus-5",
    }
    fields.update(overrides)
    return AgentModelRecommendationRequest.model_validate(fields)


def _answer(**fields: object) -> DecisionResponse:
    return DecisionResponse.model_validate(
        {"model": "jev-1.13.0", "answers": {RECOMMENDATION_QUESTION: {"type": "choice", **fields}}}
    )


def test_builds_one_choice_question_over_the_candidates() -> None:
    decision = build_decision_request(_request(), decision_model="typesafe:jev-latest", candidates=CANDIDATES)

    assert decision.model == "typesafe:jev-latest"
    assert list(decision.questions) == [RECOMMENDATION_QUESTION]
    question = decision.questions[RECOMMENDATION_QUESTION]
    assert question.type == "choice"
    assert question.criteria == CANDIDATES
    assert isinstance(decision.state, str)
    assert "Subagent type: Explore" in decision.state
    assert "Task description: List files" in decision.state
    assert "Parent model: claude-opus-5" in decision.state
    assert "List the files under src/" in decision.state


def test_the_callers_model_request_stays_out_of_the_state() -> None:
    decision = build_decision_request(
        _request(requested_model="requested-xyz"), decision_model="typesafe:jev-latest", candidates=CANDIDATES
    )

    assert "requested-xyz" not in str(decision.state)


def test_the_user_rides_along_for_billing_only() -> None:
    decision = build_decision_request(
        _request(user="alice"), decision_model="typesafe:jev-latest", candidates=CANDIDATES
    )

    assert decision.user == "alice"
    assert "alice" not in str(decision.state)


def test_a_long_task_is_shortened_and_says_so() -> None:
    prompt = "x" * (MAX_TASK_CHARS + 500)

    decision = build_decision_request(_request(prompt=prompt), decision_model="m", candidates=CANDIDATES)

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
