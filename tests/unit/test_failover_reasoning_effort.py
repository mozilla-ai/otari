"""A fallback candidate that does not reason is sent the request without ``reasoning_effort`` (#2881)."""

from typing import Any

import pytest
from any_llm import LLMProvider

from gateway.api.routes import _pipeline
from gateway.core.config import GatewayConfig
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


def _sent(model: str, **fields: Any) -> dict[str, Any]:
    attempt = Attempt(position=1, instance="groq", provider=LLMProvider.GROQ, model=model, kwargs={})
    return _pipeline._without_unused_reasoning_effort(GatewayConfig(), attempt, {"model": model, **fields})


def test_a_model_that_does_not_reason_is_sent_no_effort() -> None:
    assert _sent("llama-3.3-70b-versatile", reasoning_effort="low") == {"model": "llama-3.3-70b-versatile"}


def test_a_reasoning_model_keeps_the_effort() -> None:
    assert _sent("openai/gpt-oss-20b", reasoning_effort="low")["reasoning_effort"] == "low"


def test_a_model_models_dev_does_not_describe_keeps_the_effort() -> None:
    assert _sent("unknown-model", reasoning_effort="low")["reasoning_effort"] == "low"


def test_response_format_is_never_dropped() -> None:
    """Dropping it would hand a caller who asked for JSON a reply in prose."""
    sent = _sent("llama-3.3-70b-versatile", response_format={"type": "json_object"}, reasoning_effort="low")

    assert sent == {"model": "llama-3.3-70b-versatile", "response_format": {"type": "json_object"}}
