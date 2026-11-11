"""The same request settles the same way whichever path it takes through the gateway.

A request can reach a model through Chat Completions, Messages or Responses,
streamed or not, by naming the model or a routing policy that serves it, and
with or without a tool the gateway runs for the model. Each of those is its own
code path, and a fix made on one of them has repeatedly left the others behind
(#2958, #2960, #2991, #3034). This module sends one request down every
combination and asserts what must hold on all of them:

- the usage the response reports is every model round's, summed;
- the request settles at those tokens and at the cost the response carried;
- the response names its request, provider and model; and
- it parses as the official SDK's type for that endpoint.

The provider is ``parity_provider``, reached over real HTTP, so the gateway's
dispatch and any-llm's conversions run as they do in production. Which provider
it stands for is ``PARITY_BACKENDS``, a comma-separated list of ``BACKENDS``
names, ``fake`` when unset:

- ``fake`` scripts the answers, declared as a DeepInfra instance, which has no
  Responses API of its own, so all three endpoints end at its chat completions
  route. It needs no credential, so it runs on every pull request.
- Any other name forwards each call to that provider, with the key its
  ``key_env`` names, and records the tokens the provider reported, so the totals
  are checked against the provider's own numbers. ``otari-parity-live.yml`` runs
  these on every push to main.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest
from anthropic.types import Message, RawMessageStreamEvent
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import TypeAdapter, ValidationError

from gateway.core.config import API_ROOT, MODEL_HEADER, PROVIDER_HEADER, REQUEST_ID_HEADER, GatewayConfig

from . import parity_provider
from .conftest import build_test_client, declared_test_providers, shared_test_config

PROVIDER = "parity"
POLICY = "parity-policy"
# $1 per million tokens each way, so a settled cost is the token total in millionths.
PRICE_PER_MILLION = 1.0


@dataclass(frozen=True)
class Backend:
    """A provider the parity instance stands for."""

    provider_type: str
    model: str
    # The provider's API root the calls are forwarded to; None scripts the answers.
    upstream: str | None = None
    # The path the provider's SDK appends its routes to, below the instance's root.
    api_path: str = ""
    key_env: str | None = None

    @property
    def live(self) -> bool:
        return self.upstream is not None


BACKENDS: dict[str, Backend] = {
    "fake": Backend(provider_type="deepinfra", model="parity-model", api_path="/v1"),
    "deepseek": Backend(
        provider_type="deepseek",
        model="deepseek-chat",
        upstream="https://api.deepseek.com",
        key_env="DEEPSEEK_API_KEY",
    ),
    "anthropic": Backend(
        provider_type="anthropic",
        model="claude-haiku-4-5",
        upstream="https://api.anthropic.com",
        key_env="ANTHROPIC_API_KEY",
    ),
}

_STREAM_EVENTS: dict[str, TypeAdapter[Any]] = {
    "responses": TypeAdapter(ResponseStreamEvent),
    "messages": TypeAdapter(RawMessageStreamEvent),
}


def _selected_backends() -> list[str]:
    names = [name.strip() for name in os.environ.get("PARITY_BACKENDS", "fake").split(",") if name.strip()]
    unknown = sorted(set(names) - set(BACKENDS))
    if unknown:
        raise pytest.UsageError(f"PARITY_BACKENDS names unknown backends {unknown}; known: {sorted(BACKENDS)}")
    return names


@pytest.fixture(scope="module", params=_selected_backends())
def backend(request: pytest.FixtureRequest) -> Backend:
    selected: Backend = BACKENDS[request.param]
    if selected.key_env and not os.environ.get(selected.key_env):
        pytest.fail(f"PARITY_BACKENDS names {request.param}, but {selected.key_env} is not set")
    return selected


@pytest.fixture(scope="module")
def provider(backend: Backend) -> Iterator[parity_provider.ParityProvider]:
    with parity_provider.running(backend.upstream) as server:
        yield server


@pytest.fixture(scope="module")
def parity_config(postgres_url: str, backend: Backend, provider: parity_provider.ParityProvider) -> GatewayConfig:
    providers = declared_test_providers()
    providers[PROVIDER] = {
        "provider_type": backend.provider_type,
        "api_base": f"{provider.root}{backend.api_path}",
        "api_key": os.environ[backend.key_env] if backend.key_env else "parity-key",
    }
    return shared_test_config(postgres_url, providers)


@pytest.fixture
def client(parity_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    """The shared fixtures' client, on a config that declares the parity provider."""
    yield from build_test_client(parity_config)


@pytest.fixture
def model(backend: Backend) -> str:
    return f"{PROVIDER}:{backend.model}"


@pytest.fixture
def ready(
    client: TestClient,
    master_key_header: dict[str, str],
    model: str,
    provider: parity_provider.ParityProvider,
    serve_search_on_fake: Callable[..., None],
) -> None:
    """Price the model, store the policy that serves it, and serve the gateway's search."""
    priced = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": model,
            "input_price_per_million": PRICE_PER_MILLION,
            "output_price_per_million": PRICE_PER_MILLION,
        },
        headers=master_key_header,
    )
    assert priced.status_code == 200, priced.text
    policy = client.post(
        f"{API_ROOT}/routing/policies",
        json={"name": POLICY, "spec": {"select": [{"default": model}]}},
        headers=master_key_header,
    )
    assert policy.status_code == 200, policy.text
    serve_search_on_fake(extract=False)
    provider.reset()


@dataclass(frozen=True)
class Reported:
    """What one response told its caller."""

    input_tokens: int
    output_tokens: int
    cost_usd: str | None


def _body(endpoint: str, *, model: str, stream: bool, tools: bool) -> dict[str, Any]:
    if tools:
        prompt = "Use the web_search tool once to look up the Otari AI gateway, then describe it in one sentence."
    else:
        prompt = "Reply with the single word: ready."
    body: dict[str, Any] = {"model": model, "stream": stream}
    if endpoint == "chat/completions":
        body["messages"] = [{"role": "user", "content": prompt}]
        body["max_tokens"] = 256
    elif endpoint == "messages":
        body["messages"] = [{"role": "user", "content": prompt}]
        body["max_tokens"] = 256
    else:
        body["input"] = prompt
        body["max_output_tokens"] = 256
    if tools:
        body["tools"] = [{"type": "otari_web_search"}]
    return body


def _events(text: str) -> list[dict[str, Any]]:
    """The JSON payloads of a server-sent event stream, without the ``[DONE]`` marker."""
    payloads = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data = line.removeprefix("data:").strip()
            if data and data != "[DONE]":
                payloads.append(json.loads(data))
    return payloads


class ProtocolMismatch(AssertionError):
    """A response that does not have the shape its endpoint's official SDK expects."""


def _chat_usage(payload: Any, stream: bool) -> Reported:
    if stream:
        with_usage = [event for event in payload if event.get("usage")]
        assert with_usage, "a stream reports its usage"
        payload = with_usage[-1]
    usage = payload["usage"]
    return Reported(usage["prompt_tokens"], usage["completion_tokens"], usage.get("cost_usd"))


def _responses_usage(payload: Any, stream: bool) -> Reported:
    if stream:
        completed = [event for event in payload if event.get("type") == "response.completed"]
        assert len(completed) == 1, "a stream ends with one response.completed event"
        payload = completed[0]["response"]
    usage = payload["usage"]
    return Reported(usage["input_tokens"], usage["output_tokens"], usage.get("cost_usd"))


def _messages_usage(payload: Any, stream: bool) -> Reported:
    if not stream:
        usage = payload["usage"]
        return Reported(usage["input_tokens"], usage["output_tokens"], usage.get("cost_usd"))
    start = [event for event in payload if event.get("type") == "message_start"]
    deltas = [event for event in payload if event.get("type") == "message_delta"]
    assert len(start) == 1, "a stream opens with one message_start event"
    assert deltas, "a stream reports its final usage on a message_delta event"
    # message_delta usage is cumulative; input_tokens there, when present, supersedes message_start's.
    final = deltas[-1]["usage"]
    input_tokens = final.get("input_tokens") or start[0]["message"]["usage"]["input_tokens"]
    return Reported(input_tokens, final["output_tokens"], final.get("cost_usd"))


def _check_chat(payload: Any, stream: bool) -> None:
    if not stream:
        completion = ChatCompletion.model_validate(payload)
        if not completion.id or completion.created <= 0:
            raise ProtocolMismatch(f"a chat completion has an id and a creation time, got {completion.id!r}")
        return
    ids = {ChatCompletionChunk.model_validate(event).id for event in payload}
    if len(ids) != 1 or not next(iter(ids)):
        raise ProtocolMismatch(f"every chunk of one stream carries the same id, got {sorted(ids)}")


def _check_responses(payload: Any, stream: bool) -> None:
    if stream:
        for event in payload:
            _STREAM_EVENTS["responses"].validate_python(event)
        payload = next(event for event in payload if event.get("type") == "response.completed")["response"]
    if not Response.model_validate(payload).id:
        raise ProtocolMismatch("a response has an id")


def _check_messages(payload: Any, stream: bool) -> None:
    if not stream:
        if not Message.model_validate(payload).id:
            raise ProtocolMismatch("a message has an id")
        return
    for event in payload:
        _STREAM_EVENTS["messages"].validate_python(event)
    start = next(event for event in payload if event.get("type") == "message_start")
    if not start["message"]["id"]:
        raise ProtocolMismatch("a message has an id")


_USAGE_READERS: dict[str, Callable[[Any, bool], Reported]] = {
    "chat/completions": _chat_usage,
    "responses": _responses_usage,
    "messages": _messages_usage,
}
_PROTOCOL_CHECKS: dict[str, Callable[[Any, bool], None]] = {
    "chat/completions": _check_chat,
    "responses": _check_responses,
    "messages": _check_messages,
}


def _check_protocol(endpoint: str, payload: Any, stream: bool) -> None:
    """Raise ``ProtocolMismatch`` where the response is not what the endpoint's official SDK expects."""
    try:
        _PROTOCOL_CHECKS[endpoint](payload, stream)
    except ValidationError as exc:
        raise ProtocolMismatch(str(exc)) from exc


def _settlement(client: TestClient, request_id: str, headers: dict[str, str]) -> dict[str, Any]:
    """The request's settlement, polled because the background log writer can lag the response."""
    deadline = time.monotonic() + 5.0
    while True:
        response = client.get(f"{API_ROOT}/usage/requests/{request_id}", headers=headers)
        if response.status_code != 404 or time.monotonic() >= deadline:
            assert response.status_code == 200, response.text
            settled: dict[str, Any] = response.json()
            return settled
        time.sleep(0.1)


# Protocol mismatches that happen today, keyed by (endpoint, stream, tools). Only
# a ProtocolMismatch counts as the expected failure, checked after the accounting.
# Strict on the scripted provider, so a fix fails the run until its entry is
# removed; a live provider may not reproduce one, so there a pass is allowed.
_KNOWN_GAPS: dict[tuple[str, bool, bool], str] = {
    ("chat/completions", True, True): (
        "a streamed tool loop passes each round's provider id through, so one stream carries two ids"
    ),
    ("messages", True, False): "a Messages stream ends with an `event: done` that Anthropic's protocol does not have",
    ("messages", True, True): "a Messages stream ends with an `event: done` that Anthropic's protocol does not have",
}


@pytest.mark.usefixtures("ready")
@pytest.mark.parametrize("tools", [False, True], ids=["no-tools", "gateway-tool-loop"])
@pytest.mark.parametrize("target", ["model", "policy"])
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("endpoint", ["chat/completions", "messages", "responses"])
def test_every_path_reports_and_settles_the_same_request_alike(
    request: pytest.FixtureRequest,
    client: TestClient,
    api_key_header: dict[str, str],
    backend: Backend,
    model: str,
    provider: parity_provider.ParityProvider,
    endpoint: str,
    stream: bool,
    target: str,
    tools: bool,
) -> None:
    if gap := _KNOWN_GAPS.get((endpoint, stream, tools)):
        request.applymarker(pytest.mark.xfail(strict=not backend.live, raises=ProtocolMismatch, reason=gap))

    response = client.post(
        f"{API_ROOT}/{endpoint}",
        json=_body(endpoint, model=model if target == "model" else POLICY, stream=stream, tools=tools),
        headers=api_key_header,
    )

    assert response.status_code == 200, response.text
    if backend.live:
        assert provider.rounds, "the request reached the provider"
        if tools and len(provider.rounds) < 2:
            pytest.skip("the live model answered without searching, so no tool loop ran")
    else:
        rounds = 2 if tools else 1
        assert len(provider.rounds) == rounds, f"expected {rounds} model rounds, the provider saw {provider.requests}"
    request_id = response.headers.get(REQUEST_ID_HEADER)
    assert request_id, "the response names its request"
    assert response.headers.get(PROVIDER_HEADER) == PROVIDER, "the response names the provider that served it"
    assert response.headers.get(MODEL_HEADER) == backend.model, "the response names the model that served it"

    payload = _events(response.text) if stream else response.json()
    reported = _USAGE_READERS[endpoint](payload, stream)
    expected_input, expected_output = provider.usage()
    assert (reported.input_tokens, reported.output_tokens) == (expected_input, expected_output), (
        f"the response reports every round's tokens, summed: the provider reported {provider.rounds}"
    )
    assert reported.cost_usd is not None, "the response carries the cost it settled at"
    if not backend.live:
        # A live provider may serve part of the prompt from its cache, at a rate this price does not set.
        expected_cost = Decimal(expected_input + expected_output) * Decimal(str(PRICE_PER_MILLION)) / 1_000_000
        assert Decimal(reported.cost_usd) == expected_cost

    settled = _settlement(client, request_id, api_key_header)
    assert (settled["prompt_tokens"], settled["completion_tokens"]) == (expected_input, expected_output), (
        "the request settles at the tokens the response reported"
    )
    assert settled["cost_usd"] is not None
    assert Decimal(settled["cost_usd"]) == Decimal(reported.cost_usd), "the request settles at the cost it reported"

    _check_protocol(endpoint, payload, stream)
