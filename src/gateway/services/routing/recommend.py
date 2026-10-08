"""Recommending which model a coding agent's subagent should run on, by asking a decision model.

A subagent starts with a prompt cache of its own, so the moment it is spawned is
the one place a cheaper model costs nothing in cache misses: switching the model
of a running conversation would cold-start its cache, but a new subagent has
none to lose. The harness asks before it starts the subagent and runs the whole
subagent on the answer.

The recommendation is one choice question to a decision model such as
TypeSafe's Jev. The state is what the harness knows at spawn time, the options
are the candidate models with what each is for, and the option the model picks
is the recommendation. This module builds that question and reads the answer
into the recommender port's own terms; the call in between is the adapter's.
"""

from collections.abc import Mapping

from gateway.exceptions.routing_exceptions import UnreadableRecommendationError
from gateway.ports.agent_model_recommender_port import ModelRecommendation, RecommendationUsage, SubagentSpawn
from gateway.schemas.inference import ChoiceQuestion, DecisionRequest, DecisionResponse
from gateway.services.inference import reported_charge

RECOMMENDATION_QUESTION = "model"
"""The one question's key, in the decision request and in its answer."""

MAX_TASK_CHARS = 32_000
"""How much of the task reaches the decision model.

Jev reads at most 32k tokens of state and question together, and a task
statement puts what matters first, so the tail is what goes.
"""

_INSTRUCTIONS = (
    "A coding agent is about to hand the task in the state to a subagent. Which is the "
    "cheapest model that would complete this task well on the first attempt? Pick the "
    "cheapest option whose description fits the task, and a stronger one only when the "
    "task needs it."
)


def build_decision_request(
    spawn: SubagentSpawn,
    *,
    decision_model: str,
    candidates: Mapping[str, str | None],
) -> DecisionRequest:
    """Turn a spawn into the one choice question the decision model answers.

    The model the harness asked for is left out of the state on purpose: the
    recommendation is the gateway's, and the orchestrator's guess would only
    pull the answer toward itself.
    """
    question = ChoiceQuestion(type="choice", instructions=_INSTRUCTIONS, criteria=dict(candidates))
    return DecisionRequest(
        model=decision_model,
        state=_state_text(spawn),
        questions={RECOMMENDATION_QUESTION: question},
    )


def recommendation_from_decision(
    response: DecisionResponse,
    *,
    candidates: Mapping[str, str | None],
) -> ModelRecommendation:
    """Read the chosen candidate, and what choosing it consumed, out of the decision model's answer.

    Raises :class:`UnreadableRecommendationError` when the answer has no choice
    for the question or chooses something that is not a candidate.
    """
    answer = response.answers.get(RECOMMENDATION_QUESTION)
    if answer is None or answer.choice is None:
        msg = "the decision model answered no model question"
        raise UnreadableRecommendationError(msg)
    if answer.choice not in candidates:
        msg = "the decision model chose something that is not a candidate"
        raise UnreadableRecommendationError(msg)
    share = answer.probabilities.get(answer.choice) if answer.probabilities else None
    reason = f"{response.model} chose {answer.choice}"
    if share is not None:
        reason += f" with {share:.0%}"
    usage = response.usage
    return ModelRecommendation(
        model=answer.choice,
        reason=reason,
        probabilities=answer.probabilities,
        usage=RecommendationUsage(
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            charge=reported_charge(response),
        ),
    )


def _state_text(spawn: SubagentSpawn) -> str:
    task = spawn.prompt
    shortened = ""
    if len(task) > MAX_TASK_CHARS:
        task = task[:MAX_TASK_CHARS]
        shortened = f"\n(task shortened to its first {MAX_TASK_CHARS} characters)"
    lines = [
        f"Harness: {spawn.harness}",
        f"Subagent type: {spawn.agent_type}",
        f"Parent model: {spawn.parent_model}",
    ]
    if spawn.description:
        lines.append(f"Task description: {spawn.description}")
    lines.append(f"Task:\n{task}{shortened}")
    return "\n".join(lines)
