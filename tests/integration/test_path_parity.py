"""The same request settles the same way whichever path it takes through the gateway.

A request can reach a model through Chat Completions, Messages or Responses,
streamed or not, by naming the model or a routing policy that serves it, and
with or without a tool the gateway runs for the model. Each of those is its own
code path, and a fix made on one of them has repeatedly left the others behind
(#2958, #2960, #2991, #3034). This module sends one request down every
combination, through the official SDK a user would call it with, and asserts:

- the usage the SDK hands back is every model round's tokens, summed;
- the request settles at those tokens and at the cost the response carried;
- the response names its request, provider and model; and
- every object the SDK hands back is one its own types describe, and a chat
  stream keeps one id.

The provider is ``parity_provider``, reached over real HTTP. ``PARITY_BACKENDS``
picks what it stands for, ``fake`` when unset: ``fake`` scripts the answers as a
DeepInfra instance (no Responses API of its own, so every endpoint ends at its
chat completions route) and needs no credential; any other ``BACKENDS`` name
forwards each call to that provider and records the tokens it reported.
``otari-parity-live.yml`` runs the live ones on every push to main.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import anthropic
import httpx
import httpx2
import openai
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gateway.core.config import API_ROOT, API_VERSION, MODEL_HEADER, PROVIDER_HEADER, REQUEST_ID_HEADER, GatewayConfig

from . import parity_provider
from .conftest import build_test_client, declared_test_providers, shared_test_config

PROVIDER = "parity"
POLICY = "parity-policy"
# $1 per million tokens each way, so a settled cost is the token total in millionths.
PRICE_PER_MILLION = 1.0


@dataclass(frozen=True)
class Backend:
    """A provider the parity instance stands for: scripted, or forwarded to ``upstream`` with ``key_env``'s key."""

    provider_type: str
    model: str
    upstream: str | None = None
    key_env: str | None = None

    @property
    def live(self) -> bool:
        return self.upstream is not None


BACKENDS: dict[str, Backend] = {
    "fake": Backend("deepinfra", "parity-model"),
    "deepseek": Backend("deepseek", "deepseek-chat", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
}


def _selected_backends() -> list[str]:
    names = [name.strip() for name in os.environ.get("PARITY_BACKENDS", "fake").split(",") if name.strip()]
    if unknown := sorted(set(names) - set(BACKENDS)):
        raise pytest.UsageError(f"PARITY_BACKENDS names unknown backends {unknown}; known: {sorted(BACKENDS)}")
    return names


@pytest.fixture(scope="module", params=_selected_backends())
def backend(request: pytest.FixtureRequest) -> Backend:
    selected = BACKENDS[request.param]
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
        "api_base": provider.root,
        "api_key": os.environ[backend.key_env] if backend.key_env else "parity-key",
    }
    return shared_test_config(postgres_url, providers)


@pytest.fixture
def client(parity_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    """The shared fixtures' client, on a config that declares the parity provider."""
    yield from build_test_client(parity_config)


@pytest.fixture
def model(
    backend: Backend,
    client: TestClient,
    master_key_header: dict[str, str],
    provider: parity_provider.ParityProvider,
    serve_search_on_fake: Callable[..., None],
) -> str:
    """The parity model, priced and served by a stored policy too, with the gateway's search on the fake backend."""
    model = f"{PROVIDER}:{backend.model}"
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
    return model


class ProtocolMismatch(AssertionError):
    """An answer the endpoint's official SDK hands its caller in a shape its own types do not describe."""


@dataclass(frozen=True)
class Answer:
    """What one call through an SDK handed back: its headers, every object it yielded, and the usage among them."""

    headers: Any
    objects: list[Any]
    usage: Any


def _through(client: TestClient, http: Any) -> Any:
    """A client of ``http`` (httpx or httpx2, whichever the SDK is built on) that sends through ``client``."""

    def forward(request: Any) -> Any:
        sent = client.request(
            request.method, str(request.url), headers=request.headers.multi_items(), content=request.read()
        )
        return http.Response(sent.status_code, headers=sent.headers.multi_items(), content=sent.content)

    return http.Client(transport=http.MockTransport(forward))


def _call(client: TestClient, api_key: str, endpoint: str, *, model: str, stream: bool, tools: bool) -> Answer:
    if tools:
        prompt = "Use the web_search tool once to look up the Otari AI gateway, then describe it in one sentence."
    else:
        prompt = "Reply with the single word: ready."
    extra = {"tools": [{"type": "otari_web_search"}]} if tools else None
    messages: Any = [{"role": "user", "content": prompt}]
    raw: Any
    if endpoint == "messages":
        sdk = anthropic.Anthropic(
            base_url=f"{client.base_url}{API_ROOT.removesuffix(f'/{API_VERSION}')}",
            api_key=api_key,
            http_client=_through(client, httpx2),
            max_retries=0,
        )
        raw = sdk.messages.with_raw_response.create(
            model=model, max_tokens=256, messages=messages, stream=stream, extra_body=extra
        )
    else:
        openai_sdk = openai.OpenAI(
            base_url=f"{client.base_url}{API_ROOT}", api_key=api_key, http_client=_through(client, httpx), max_retries=0
        )
        if endpoint == "chat/completions":
            raw = openai_sdk.chat.completions.with_raw_response.create(
                model=model,
                max_tokens=256,
                messages=messages,
                stream=stream,
                stream_options={"include_usage": True} if stream else openai.omit,
                extra_body=extra,
            )
        else:
            raw = openai_sdk.responses.with_raw_response.create(
                model=model, max_output_tokens=256, input=prompt, stream=stream, extra_body=extra
            )
    parsed = raw.parse()
    objects = list(parsed) if stream else [parsed]
    return Answer(raw.headers, objects, _usage(endpoint, objects))


def _usage(endpoint: str, objects: list[Any]) -> Any:
    """The usage the caller reads: the whole response's, or the stream's final report."""
    if endpoint == "chat/completions":
        return next(o.usage for o in reversed(objects) if getattr(o, "usage", None))
    if endpoint == "responses":
        final = next((o.response for o in objects if getattr(o, "type", None) == "response.completed"), objects[0])
        return final.usage
    if len(objects) == 1:
        return objects[0].usage
    # A Messages stream: input tokens on message_start, cumulative usage on the last message_delta.
    start = next(o for o in objects if getattr(o, "type", None) == "message_start").message.usage
    delta = [o.usage for o in objects if getattr(o, "type", None) == "message_delta"][-1]
    return delta.model_copy(update={"input_tokens": delta.input_tokens or start.input_tokens})


def _tokens(usage: Any) -> tuple[int, int]:
    if hasattr(usage, "prompt_tokens"):
        return usage.prompt_tokens, usage.completion_tokens
    return usage.input_tokens, usage.output_tokens


def _check_protocol(endpoint: str, objects: list[Any]) -> None:
    """Raise ``ProtocolMismatch`` where what the SDK handed back is not what its own types describe.

    The SDKs build their objects without validating them, so each is validated
    against its own type here.
    """
    for obj in objects:
        try:
            type(obj).model_validate(obj.to_dict())
        except ValidationError as exc:
            raise ProtocolMismatch(f"the SDK handed back a {type(obj).__name__} its type rejects: {exc}") from exc
        if getattr(obj, "id", "id") == "" or getattr(obj, "created", 1) <= 0:
            raise ProtocolMismatch(f"the SDK handed back a {type(obj).__name__} with no id or creation time")
    if endpoint == "chat/completions" and len(ids := {obj.id for obj in objects}) != 1:
        raise ProtocolMismatch(f"one chat completion carries one id, got {sorted(ids)}")


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


# Protocol mismatches that happen today, keyed by (endpoint, stream, tools). Only a
# ProtocolMismatch, checked after the accounting, counts as the expected failure.
# Strict on the scripted provider, so a fix fails the run until its entry is
# removed; a live provider may not reproduce one, so there a pass is allowed.
_KNOWN_GAPS: dict[tuple[str, bool, bool], str] = {
    ("messages", True, False): "a Messages stream ends with an `event: done` the SDK hands back as a typeless event",
    ("messages", True, True): "a Messages stream ends with an `event: done` the SDK hands back as a typeless event",
}


@pytest.mark.parametrize("tools", [False, True], ids=["no-tools", "gateway-tool-loop"])
@pytest.mark.parametrize("target", ["model", "policy"])
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("endpoint", ["chat/completions", "messages", "responses"])
def test_every_path_reports_and_settles_the_same_request_alike(
    request: pytest.FixtureRequest,
    client: TestClient,
    api_key_obj: dict[str, Any],
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

    answer = _call(
        client, api_key_obj["key"], endpoint, model=model if target == "model" else POLICY, stream=stream, tools=tools
    )

    if backend.live:
        assert provider.rounds, "the request reached the provider"
        # Requests are recorded before they are forwarded, so a follow-up that failed still counts.
        if tools and len(provider.requests) < 2:
            pytest.skip("the live model answered without searching, so no tool loop ran")
    else:
        rounds = 2 if tools else 1
        assert len(provider.rounds) == rounds, f"expected {rounds} model rounds, the provider saw {provider.requests}"
    request_id = answer.headers.get(REQUEST_ID_HEADER)
    assert request_id, "the response names its request"
    assert answer.headers.get(PROVIDER_HEADER) == PROVIDER, "the response names the provider that served it"
    assert answer.headers.get(MODEL_HEADER) == backend.model, "the response names the model that served it"

    expected_input, expected_output = provider.usage()
    assert _tokens(answer.usage) == (expected_input, expected_output), (
        f"the response reports every round's tokens, summed: the provider reported {provider.rounds}"
    )
    cost = getattr(answer.usage, "cost_usd", None)
    assert cost is not None, "the response carries the cost it settled at"
    if not backend.live:
        # A live provider may serve part of the prompt from its cache, at a rate this price does not set.
        assert Decimal(cost) == Decimal(expected_input + expected_output) * Decimal(str(PRICE_PER_MILLION)) / 1_000_000

    settled = _settlement(client, request_id, api_key_header)
    assert (settled["prompt_tokens"], settled["completion_tokens"]) == (expected_input, expected_output), (
        "the request settles at the tokens the response reported"
    )
    assert settled["cost_usd"] is not None
    assert Decimal(settled["cost_usd"]) == Decimal(cost), "the request settles at the cost it reported"

    _check_protocol(endpoint, answer.objects)
