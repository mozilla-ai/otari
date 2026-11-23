"""A decisions request answered by a chat model, which states its probabilities itself.

A decision model reads its probabilities off its own output distribution; a chat
model is asked for them. The state, its images and the questions are put to the
model as one user message, and a JSON schema built from the questions holds the
answer to one probability per yes, per option and per level. The answer is
returned in the shape a decision provider's is, so a caller cannot tell which
kind answered from its fields. The probabilities are the model's own estimate,
not a calibrated one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gateway.services.inference._decisions import _as_text

if TYPE_CHECKING:
    from gateway.schemas.inference import DecisionRequest

_SYSTEM = (
    "You answer questions about a state with probabilities. Reply with one JSON object holding an "
    "answer under each question's name. For a yes-or-no question, the answer is the probability "
    "that the answer is yes. For a choice question, it is the probability of each option, keyed by "
    "the option's name. For a score question, it is the probability of each level, keyed by the "
    "level's number, where 0 is the lowest. Every probability is between 0 and 1, and the "
    "probabilities of one choice or score question add up to 1. Judge from the state and its "
    "images alone."
)


def _number(value: object) -> float:
    """``value``, which the model stated as a probability.

    Raises:
        ValueError: If ``value`` is not a number.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"expected a probability, got {type(value).__name__}"
        raise ValueError(msg)
    return float(value)


def _probability(value: object) -> float:
    """``value`` as a probability, clamped into [0, 1].

    Raises:
        ValueError: If ``value`` is not a number.

    """
    return min(1.0, max(0.0, _number(value)))


def _distribution(stated: object, keys: list[str]) -> dict[str, float]:
    """The probability of each of ``keys`` the model stated, scaled to add up to 1.

    A model's figures need not add up, so they are scaled, a negative one read
    as 0; when every one is 0, each key gets an equal share.

    Raises:
        ValueError: If ``stated`` is not an object holding a number for every key.

    """
    if not isinstance(stated, dict):
        msg = f"expected an object of probabilities, got {type(stated).__name__}"
        raise ValueError(msg)
    weights = {key: max(0.0, _number(stated.get(key))) for key in keys}
    total = sum(weights.values())
    if total == 0:
        return dict.fromkeys(keys, 1 / len(keys))
    return {key: weight / total for key, weight in weights.items()}


def _object_of(properties: list[str]) -> dict[str, Any]:
    """A schema for an object holding a number under each of ``properties``, and nothing else."""
    return {
        "type": "object",
        "properties": {name: {"type": "number"} for name in properties},
        "required": properties,
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class DecisionPrompt:
    """A decisions request as a chat model is asked it, and its answer read back in TypeSafe's shape."""

    request: DecisionRequest

    def messages(self) -> list[dict[str, Any]]:
        """The system and user messages to send: the user's carries the state, its images and the questions."""
        text = f"State:\n{_as_text(self.request.state)}\n\nQuestions:\n{self._questions()}"
        images = self.request.images or []
        content: str | list[dict[str, Any]] = (
            [{"type": "text", "text": text}, *({"type": "image_url", "image_url": {"url": url}} for url in images)]
            if images
            else text
        )
        return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": content}]

    def response_format(self) -> dict[str, Any]:
        """A strict JSON schema holding one probability per yes, per option and per level."""
        properties: dict[str, Any] = {}
        for name, question in self.request.questions.items():
            if question.type == "noul":
                properties[name] = {"type": "number"}
            elif question.type == "choice":
                properties[name] = _object_of(list(question.criteria))
            else:
                properties[name] = _object_of([str(level) for level in range(len(question.criteria))])
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        return {"type": "json_schema", "json_schema": {"name": "decision", "schema": schema, "strict": True}}

    def answers(self, content: str | None) -> dict[str, dict[str, Any]]:
        """The model's reply as TypeSafe's answers, keyed like the questions.

        A ``noul`` answer is the probability of yes. A ``choice`` answer names the
        most probable option, with each option's probability. A ``score`` answer is
        the probability-weighted average level, with each level's probability and
        the criterion it stands for. ``confidence`` is the probability of the most
        probable outcome.

        Raises:
            ValueError: If the reply is not a JSON object answering every question.

        """
        try:
            reply = json.loads(content or "")
        except json.JSONDecodeError as exc:
            msg = "the answer is not JSON"
            raise ValueError(msg) from exc
        if not isinstance(reply, dict):
            msg = f"the answer is a {type(reply).__name__}, expected an object"
            raise ValueError(msg)
        answers: dict[str, dict[str, Any]] = {}
        for name, question in self.request.questions.items():
            if name not in reply:
                msg = "the answer leaves a question out"
                raise ValueError(msg)
            stated = reply[name]
            if question.type == "noul":
                yes = _probability(stated)
                answers[name] = {"type": "noul", "noul": yes, "confidence": max(yes, 1 - yes)}
            elif question.type == "choice":
                probabilities = _distribution(stated, list(question.criteria))
                choice = max(probabilities, key=lambda option: probabilities[option])
                answers[name] = {
                    "type": "choice",
                    "choice": choice,
                    "confidence": probabilities[choice],
                    "probabilities": probabilities,
                }
            else:
                levels = [str(level) for level in range(len(question.criteria))]
                probabilities = _distribution(stated, levels)
                answers[name] = {
                    "type": "score",
                    "score": sum(int(level) * probability for level, probability in probabilities.items()),
                    "confidence": max(probabilities.values()),
                    "legend": dict(zip(levels, question.criteria, strict=True)),
                    "probabilities": probabilities,
                }
        return answers

    def _questions(self) -> str:
        """The questions as the user message lists them, each under the name its answer is keyed by."""
        lines: list[str] = []
        for name, question in self.request.questions.items():
            instructions = _as_text(question.instructions)
            if question.type == "noul":
                lines.append(f"- {json.dumps(name)} (yes or no): {instructions}")
                criteria = question.criteria
                if criteria is not None and criteria.true is not None:
                    lines.append(f"  yes means: {_as_text(criteria.true)}")
                if criteria is not None and criteria.false is not None:
                    lines.append(f"  no means: {_as_text(criteria.false)}")
            elif question.type == "choice":
                lines.append(f"- {json.dumps(name)} (choice): {instructions}")
                lines.append("  options:")
                lines.extend(
                    f"  - {json.dumps(option)}" + ("" if meaning is None else f": {_as_text(meaning)}")
                    for option, meaning in question.criteria.items()
                )
            else:
                lines.append(f"- {json.dumps(name)} (score): {instructions}")
                lines.append("  levels, lowest first:")
                lines.extend(f"  - {level}: {_as_text(meaning)}" for level, meaning in enumerate(question.criteria))
        return "\n".join(lines)
