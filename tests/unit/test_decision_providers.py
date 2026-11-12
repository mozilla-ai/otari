"""Unit tests for the decisions providers: their config rules, the client, and the request schema.

The HTTP layer is a mock transport, so the suite needs no provider key and no network.
"""

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig, validate_decision_provider_entry
from gateway.schemas.inference import MAX_DECISION_IMAGES, DecisionRequest, DecisionResponse
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
        ("gemini", {"api_key": "k"}),
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
        ("google", {"provider": "gemini"}, "api_key is required"),
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
            "google": {"provider": "gemini", "api_key": "gm"},
            "local": {"provider": "llamacpp", "api_base": "http://127.0.0.1:8080/"},
        }
    )
    typesafe, model = resolve_decision_provider(config, "typesafe:jev-latest")
    assert (typesafe.url, typesafe.api_key, model) == ("https://api.typesafe.ai/v1/systemone", "ts", "jev-latest")
    openrouter, model = resolve_decision_provider(config, "openrouter:typesafe/jev-1.13")
    assert (openrouter.url, model) == ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13")
    google, model = resolve_decision_provider(config, "google:gemini-decision-1")
    assert (google.url, google.api_key, model) == (
        "https://generativelanguage.googleapis.com/v1beta/decisions",
        "gm",
        "gemini-decision-1",
    )
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


def _gemini() -> DecisionProvider:
    return DecisionProvider(
        name="google",
        provider="gemini",
        api_key="gkey",
        url="https://generativelanguage.googleapis.com/v1beta/decisions",
        timeout_s=5,
    )


async def _ask_gemini(body: dict[str, Any], reply: dict[str, Any]) -> tuple[dict[str, Any], httpx.Request]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        answer = await request_decision(_gemini(), body, client=client)
    (request,) = seen
    return answer, request


@pytest.mark.asyncio
async def test_gemini_is_asked_in_its_own_words_with_its_own_key_header() -> None:
    body = {
        "model": "gemini-decision-1",
        "state": {"message": "charged twice"},
        "questions": {
            "defect": {
                "type": "noul",
                "instructions": "Is this a defect?",
                "criteria": {"true": "Broken", "false": None},
                "typesafe_only": 1,
            },
            "team": {"type": "choice", "instructions": "Team?", "criteria": {"billing": None, "tech": {"k": "v"}}},
            "mood": {"type": "score", "instructions": ["Mood", "?"], "criteria": ["calm", "angry"]},
        },
    }
    _, request = await _ask_gemini(body, {"answers": {}, "usage": {"total_input_tokens": 1}})
    assert request.headers["x-goog-api-key"] == "gkey"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {
        "model": "gemini-decision-1",
        "input": '{"message": "charged twice"}',
        "questions": {
            "defect": {"type": "noul", "instructions": "Is this a defect?", "criteria": {"true_description": "Broken"}},
            "team": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "", "tech": '{"k": "v"}'}},
            "mood": {"type": "score", "instructions": '["Mood", "?"]', "criteria": ["calm", "angry"]},
        },
    }


@pytest.mark.asyncio
async def test_gemini_receives_inline_images_as_image_blocks() -> None:
    body = {
        "model": "gemini-decision-1",
        "state": "photo",
        "questions": QUESTIONS,
        "images": ["data:image/png;base64,AA"],
    }
    _, request = await _ask_gemini(body, {"answers": {}, "usage": {"total_input_tokens": 1}})
    assert json.loads(request.content)["input"] == [
        {"type": "text", "text": "photo"},
        {"type": "image", "mime_type": "image/png", "data": "AA"},
    ]


@pytest.mark.asyncio
async def test_gemini_answer_names_the_model_and_counts_tokens_under_typesafe_keys() -> None:
    reply = {
        "answers": {"urgent": {"type": "noul", "noul": 0.97, "confidence": 0.97}},
        "usage": {"total_input_tokens": 100, "total_output_tokens": 0, "total_tokens": 100},
    }
    answer, _ = await _ask_gemini({"model": "gemini-decision-1", "state": "s", "questions": QUESTIONS}, reply)
    response = DecisionResponse.model_validate(answer)
    assert response.model == "gemini-decision-1"
    assert response.answers["urgent"].noul == 0.97
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.cost) == (100, 0, None)


@pytest.mark.asyncio
async def test_gemini_answer_without_an_input_count_falls_back_to_the_total() -> None:
    reply = {"answers": {}, "usage": {"total_tokens": 42}}
    answer, _ = await _ask_gemini({"model": "gemini-decision-1", "state": "s", "questions": QUESTIONS}, reply)
    assert (answer["usage"]["input_tokens"], answer["usage"]["output_tokens"]) == (42, 0)


@pytest.mark.asyncio
async def test_gemini_answer_reporting_zero_input_tokens_keeps_the_zero() -> None:
    reply = {"answers": {}, "usage": {"total_input_tokens": 0, "total_tokens": 7}}
    answer, _ = await _ask_gemini({"model": "gemini-decision-1", "state": "s", "questions": QUESTIONS}, reply)
    assert answer["usage"]["input_tokens"] == 0


@pytest.mark.parametrize("reply", [{"answers": {}}, {"answers": {}, "usage": {"total_output_tokens": 0}}])
@pytest.mark.asyncio
async def test_gemini_answer_with_no_token_count_is_refused(reply: dict[str, Any]) -> None:
    with pytest.raises(DecisionProviderError, match="no input token count") as raised:
        await _ask_gemini({"model": "gemini-decision-1", "state": "s", "questions": QUESTIONS}, reply)
    assert raised.value.status_code == 200


@pytest.mark.parametrize("status_code", [401, 422, 429, 501, 529])
@pytest.mark.asyncio
async def test_an_error_status_carries_the_status_and_no_body(status_code: int) -> None:
    response = httpx.Response(status_code, json={"error": "the state you sent: SECRET"})
    with pytest.raises(DecisionProviderError) as raised:
        await _send(response, _provider())
    assert raised.value.status_code == status_code
    assert "SECRET" not in str(raised.value)


@pytest.mark.parametrize("status_code", [301, 302, 304, 307, 308])
@pytest.mark.asyncio
async def test_a_redirect_is_refused_and_not_followed(status_code: int) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status_code, headers={"location": "https://elsewhere.example/"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        with pytest.raises(DecisionProviderError) as raised:
            await request_decision(_provider(), {"model": "jev-latest", "state": "s"}, client=client)
    assert raised.value.status_code == status_code
    assert [str(request.url) for request in seen] == ["https://api.typesafe.ai/v1/systemone"]


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
