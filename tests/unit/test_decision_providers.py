"""Unit tests for the decisions providers: their config rules, the client, and the request schema.

The HTTP layer is a mock transport, so the suite needs no provider key and no network.
"""

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig, validate_decision_provider_entry
from gateway.schemas.inference import MAX_DECISION_IMAGES, DecisionRequest
from gateway.services.inference import (
    DecisionProvider,
    DecisionProviderError,
    UnknownDecisionProviderError,
    request_decision,
    resolve_decision_provider,
)

QUESTIONS = {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}


@pytest.mark.parametrize(
    ("name", "entry"),
    [
        ("typesafe", {"api_key": "k"}),
        ("router", {"provider": "openrouter", "api_key": "k", "api_base": "https://proxy.example/api"}),
        ("local", {"provider": "llamacpp", "api_base": "http://127.0.0.1:8080"}),
        ("remote", {"provider": "llamacpp", "api_base": "https://llm.example", "api_key": "k", "timeout": 5}),
    ],
)
def test_a_valid_entry_is_accepted(name: str, entry: dict[str, Any]) -> None:
    validate_decision_provider_entry(name, entry)


@pytest.mark.parametrize(
    ("name", "entry", "message"),
    [
        ("typesafe", {}, "api_key is required"),
        ("openrouter", {"api_key": "k", "api_base": "http://proxy.example"}, "must use https"),
        ("local", {"provider": "llamacpp"}, "api_base is required"),
        ("local", {"provider": "llamacpp", "api_base": "http://h:8080", "api_key": "k"}, "must use https"),
        ("other", {"api_key": "k"}, "not a supported decision provider"),
        ("otari", {"provider": "typesafe", "api_key": "k"}, "reserved"),
        ("a:b", {"provider": "typesafe", "api_key": "k"}, "must not contain"),
        ("typesafe", {"api_key": "k", "timeout": 0}, "timeout"),
        ("typesafe", "not a mapping", "must be a mapping"),
    ],
)
def test_an_invalid_entry_is_refused(name: str, entry: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_decision_provider_entry(name, entry)


def test_selector_resolves_to_the_provider_endpoint() -> None:
    config = GatewayConfig(
        decision_providers={
            "typesafe": {"api_key": "ts"},
            "openrouter": {"api_key": "or"},
            "local": {"provider": "llamacpp", "api_base": "http://127.0.0.1:8080/"},
        }
    )
    typesafe, model = resolve_decision_provider(config, "typesafe:jev-latest")
    assert (typesafe.url, typesafe.api_key, model) == ("https://api.typesafe.ai/v1/systemone", "ts", "jev-latest")
    openrouter, model = resolve_decision_provider(config, "openrouter:typesafe/jev-1.13")
    assert (openrouter.url, model) == ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13")
    local, model = resolve_decision_provider(config, "local:openjev")
    assert (local.url, local.api_key, model) == ("http://127.0.0.1:8080/v1/systemone", None, "openjev")


@pytest.mark.parametrize("selector", ["jev-latest", "unknown:jev-latest", "openai:gpt-4o"])
def test_an_unconfigured_selector_is_refused(selector: str) -> None:
    config = GatewayConfig(decision_providers={"typesafe": {"api_key": "ts"}})
    with pytest.raises(UnknownDecisionProviderError, match="typesafe"):
        resolve_decision_provider(config, selector)


def _provider(api_key: str | None = "secret") -> DecisionProvider:
    return DecisionProvider(
        name="typesafe", provider="typesafe", api_key=api_key, url="https://api.typesafe.ai/v1/systemone", timeout_s=5
    )


async def _send(response: httpx.Response | Exception, provider: DecisionProvider) -> tuple[Any, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if isinstance(response, Exception):
            raise response
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        answer = await request_decision(provider, {"model": "jev-latest", "state": "s"}, client=client)
    return answer, seen


@pytest.mark.asyncio
async def test_the_body_and_key_reach_the_provider() -> None:
    answer, seen = await _send(httpx.Response(200, json={"model": "jev", "answers": {}}), _provider())
    assert answer == {"model": "jev", "answers": {}}
    (request,) = seen
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer secret"
    assert json.loads(request.content) == {"model": "jev-latest", "state": "s"}


@pytest.mark.asyncio
async def test_a_keyless_provider_sends_no_authorization_header() -> None:
    _, seen = await _send(httpx.Response(200, json={"answers": {}}), _provider(api_key=None))
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize("status_code", [401, 422, 429, 501, 529])
@pytest.mark.asyncio
async def test_an_error_status_carries_the_status_and_no_body(status_code: int) -> None:
    response = httpx.Response(status_code, json={"error": "the state you sent: SECRET"})
    with pytest.raises(DecisionProviderError) as raised:
        await _send(response, _provider())
    assert raised.value.status_code == status_code
    assert "SECRET" not in str(raised.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["a", "list"]),
        httpx.ConnectError("refused"),
    ],
)
@pytest.mark.asyncio
async def test_an_unreadable_or_unreachable_provider_raises(response: httpx.Response | Exception) -> None:
    with pytest.raises(DecisionProviderError):
        await _send(response, _provider())


def test_the_request_accepts_every_question_type_and_keeps_extra_fields() -> None:
    request = DecisionRequest.model_validate(
        {
            "model": "typesafe:jev-latest",
            "state": {"message": "hi", "plan": "pro"},
            "questions": {
                "urgent": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "yes", "false": "no"}},
                "team": {"type": "choice", "instructions": "Team?", "criteria": {"billing": None, "tech": "t"}},
                "mood": {"type": "score", "instructions": "Mood?", "criteria": ["calm", "angry"], "future": 1},
            },
        }
    )
    assert request.questions["mood"].model_dump(exclude_none=True)["future"] == 1


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "x", "criteria": {"only": None}},
        {"type": "score", "instructions": "x", "criteria": ["one"]},
        {"type": "score", "instructions": "x", "criteria": [str(n) for n in range(11)]},
        {"type": "unknown", "instructions": "x"},
    ],
)
def test_a_malformed_question_is_refused(question: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({"model": "typesafe:jev", "state": "s", "questions": {"q": question}})


def test_images_must_be_inline_and_few() -> None:
    base = {"model": "local:openjev", "state": "s", "questions": QUESTIONS}
    assert DecisionRequest.model_validate({**base, "images": ["data:image/png;base64,AA"]}).images
    with pytest.raises(ValidationError, match="data URL"):
        DecisionRequest.model_validate({**base, "images": ["https://example.com/a.png"]})
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({**base, "images": ["data:image/png;base64,AA"] * (MAX_DECISION_IMAGES + 1)})
