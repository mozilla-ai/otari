"""Unit tests for a decisions request put to a chat model: the prompt, the answer's schema, and reading it back."""

import json
from typing import Any

import pytest

from gateway.schemas.inference import DecisionRequest, DecisionResponse
from gateway.services.inference import DecisionPrompt

QUESTIONS: dict[str, Any] = {
    "urgent": {"type": "noul", "instructions": "Is this urgent?", "criteria": {"true": "Act today", "false": None}},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": "Outages"}},
    "mood": {"type": "score", "instructions": "How upset?", "criteria": ["calm", {"level": "annoyed"}, "angry"]},
}


def _prompt(**extra: Any) -> DecisionPrompt:
    return DecisionPrompt(
        DecisionRequest.model_validate({"model": "openai:gpt-5-mini", "state": "s", "questions": QUESTIONS, **extra})
    )


def test_the_user_message_carries_the_state_and_every_question_under_its_name() -> None:
    system, user = _prompt(state={"message": "charged twice"}).messages()
    assert system["role"] == "system"
    text = user["content"]
    assert text.startswith('State:\n{"message": "charged twice"}\n\nQuestions:\n')
    assert '- "urgent" (yes or no): Is this urgent?\n  yes means: Act today\n' in text
    assert "no means" not in text
    assert '- "team" (choice): Which team?\n  options:\n  - "billing"\n  - "tech": Outages\n' in text
    assert '- "mood" (score): How upset?\n  levels, lowest first:\n  - 0: calm\n  - 1: {"level": "annoyed"}\n' in text


def test_images_follow_the_text_as_image_url_parts() -> None:
    images = ["data:image/png;base64,AA", "data:image/jpeg;base64,BB"]
    _, user = _prompt(images=images).messages()
    text, *parts = user["content"]
    assert text["type"] == "text"
    assert text["text"].startswith("State:\ns")
    assert parts == [{"type": "image_url", "image_url": {"url": url}} for url in images]


def test_the_schema_holds_one_probability_per_yes_option_and_level() -> None:
    response_format = _prompt().response_format()
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["urgent", "team", "mood"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["urgent"] == {"type": "number"}
    assert schema["properties"]["team"] == {
        "type": "object",
        "properties": {"billing": {"type": "number"}, "tech": {"type": "number"}},
        "required": ["billing", "tech"],
        "additionalProperties": False,
    }
    assert schema["properties"]["mood"]["required"] == ["0", "1", "2"]


def test_the_answer_reads_back_in_typesafe_shape() -> None:
    reply = {"urgent": 0.8, "team": {"billing": 1, "tech": 3}, "mood": {"0": 0.2, "1": 0.3, "2": 0.5}}
    answers = _prompt().answers(json.dumps(reply))
    response = DecisionResponse.model_validate({"model": "gpt-5-mini", "answers": answers})

    urgent = response.answers["urgent"]
    assert (urgent.type, urgent.noul, urgent.confidence) == ("noul", 0.8, 0.8)
    team = response.answers["team"]
    assert (team.choice, team.confidence) == ("tech", 0.75)
    # A model's figures need not add up, so they are scaled to.
    assert team.probabilities == {"billing": 0.25, "tech": 0.75}
    mood = response.answers["mood"]
    # The probability-weighted average level, as a decision model reports it.
    assert mood.score == pytest.approx(1.3)
    assert mood.confidence == 0.5
    assert mood.legend == {"0": "calm", "1": {"level": "annoyed"}, "2": "angry"}
    assert mood.probabilities == pytest.approx({"0": 0.2, "1": 0.3, "2": 0.5})


def test_a_probability_out_of_range_is_clamped_and_all_zeros_share_evenly() -> None:
    reply = {"urgent": 1.7, "team": {"billing": 0, "tech": -1}, "mood": {"0": 0, "1": 0, "2": 0}}
    answers = _prompt().answers(json.dumps(reply))
    assert answers["urgent"]["noul"] == 1.0
    assert answers["team"]["probabilities"] == {"billing": 0.5, "tech": 0.5}
    assert answers["mood"]["score"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "reply",
    [
        None,
        "not json",
        "[0.5]",
        json.dumps({"urgent": 0.5, "team": {"billing": 1, "tech": 0}}),
        json.dumps({"urgent": True, "team": {"billing": 1, "tech": 0}, "mood": {"0": 1, "1": 0, "2": 0}}),
        json.dumps({"urgent": 0.5, "team": {"billing": 1}, "mood": {"0": 1, "1": 0, "2": 0}}),
        json.dumps({"urgent": 0.5, "team": [1, 0], "mood": {"0": 1, "1": 0, "2": 0}}),
    ],
)
def test_an_answer_that_does_not_answer_every_question_is_refused(reply: str | None) -> None:
    with pytest.raises(ValueError, match="answer|expected"):
        _prompt().answers(reply)
