"""A fallback candidate that does not reason is sent the request without its reasoning hint (#2881)."""

from typing import Any

import pytest
from any_llm import LLMProvider

from gateway.api.routes import _pipeline
from gateway.types.attempt import Attempt
from gateway.types.models_dev import parse_catalog

_CATALOG = parse_catalog(
    {
        "groq": {
            "models": {
                "llama-3.3-70b-versatile": {"reasoning": False},
                "openai/gpt-oss-20b": {"reasoning": True},
            }
        }
    }
)


@pytest.fixture(autouse=True)
def _catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_pipeline, "cached_models_dev_catalog", lambda: _CATALOG)


def _sent(model: str, *, instance: str = "groq", **fields: Any) -> dict[str, Any]:
    attempt = Attempt(position=1, instance=instance, provider=LLMProvider.GROQ, model=model, kwargs={})
    return _pipeline._without_unused_reasoning(attempt, {"model": model, **fields})


def test_a_model_that_does_not_reason_is_sent_no_effort() -> None:
    assert _sent("llama-3.3-70b-versatile", reasoning_effort="low") == {"model": "llama-3.3-70b-versatile"}


def test_an_instance_named_for_no_provider_is_looked_up_under_its_provider() -> None:
    """A hosted provider or owned endpoint can be named anything; the call still goes to Groq."""
    assert _sent("llama-3.3-70b-versatile", instance="team-groq", reasoning_effort="low") == {
        "model": "llama-3.3-70b-versatile"
    }


def test_a_reasoning_model_keeps_the_effort() -> None:
    assert _sent("openai/gpt-oss-20b", reasoning_effort="low")["reasoning_effort"] == "low"


def test_a_model_models_dev_does_not_describe_keeps_the_effort() -> None:
    assert _sent("unknown-model", reasoning_effort="low")["reasoning_effort"] == "low"


def test_response_format_is_never_dropped() -> None:
    """Dropping it would hand a caller who asked for JSON a reply in prose."""
    sent = _sent("llama-3.3-70b-versatile", response_format={"type": "json_object"}, reasoning_effort="low")

    assert sent == {"model": "llama-3.3-70b-versatile", "response_format": {"type": "json_object"}}


def test_a_responses_reasoning_is_left_off_a_model_that_does_not_reason() -> None:
    assert _sent("llama-3.3-70b-versatile", reasoning={"effort": "low", "summary": "auto"}) == {
        "model": "llama-3.3-70b-versatile"
    }


def test_a_reasoning_model_keeps_the_responses_reasoning() -> None:
    assert _sent("openai/gpt-oss-20b", reasoning={"effort": "low"})["reasoning"] == {"effort": "low"}


def test_encrypted_reasoning_is_not_asked_of_a_model_that_does_not_reason() -> None:
    """The other ``include`` entries still ask for what they asked for."""
    sent = _sent(
        "llama-3.3-70b-versatile",
        reasoning={"effort": "low"},
        include=["reasoning.encrypted_content", "message.output_text.logprobs"],
    )

    assert sent == {"model": "llama-3.3-70b-versatile", "include": ["message.output_text.logprobs"]}


def test_an_include_asking_only_for_encrypted_reasoning_is_left_off() -> None:
    sent = _sent("llama-3.3-70b-versatile", include=["reasoning.encrypted_content"])

    assert sent == {"model": "llama-3.3-70b-versatile"}


def test_a_reasoning_model_is_still_asked_for_encrypted_reasoning() -> None:
    include = ["reasoning.encrypted_content"]

    assert _sent("openai/gpt-oss-20b", reasoning={"effort": "low"}, include=include)["include"] == include


def test_a_responses_text_format_is_never_dropped() -> None:
    text = {"format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}, "strict": True}}

    sent = _sent("llama-3.3-70b-versatile", reasoning={"effort": "low"}, text=text)

    assert sent == {"model": "llama-3.3-70b-versatile", "text": text}


def test_a_messages_thinking_is_left_off_a_model_that_does_not_reason() -> None:
    assert _sent("llama-3.3-70b-versatile", thinking={"type": "enabled", "budget_tokens": 2048}) == {
        "model": "llama-3.3-70b-versatile"
    }


def test_a_reasoning_model_keeps_the_messages_thinking() -> None:
    thinking = {"type": "enabled", "budget_tokens": 2048}

    assert _sent("openai/gpt-oss-20b", thinking=thinking)["thinking"] == thinking
