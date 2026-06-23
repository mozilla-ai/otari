"""End-to-end tests for the router seam, driving the real chat route.

Only the provider call (`acompletion`) is mocked; everything else (auth,
dispatch, the routing seam, settlement) runs for real. These tests verify:

- routing disabled (default) and the no-op backend pass the requested model
  through unchanged;
- a backend that reroutes actually changes the model the provider receives, on
  both the non-streaming and streaming standalone paths (this is the behavior
  the seam exists to enable);
- platform mode never consults the router.
"""

import os
from collections.abc import AsyncIterator, Generator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.services.router_backend import (
    NoOpRouterBackend,
    RoutingContext,
    RoutingDecision,
)

REQUESTED_MODEL = "gemini:gemini-2.5-flash"
REROUTE_TARGET = "openai:gpt-4o-mini"


def _completion(model: str = "gemini-2.5-flash") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-e2e",
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class _RerouteBackend:
    """Test backend that always reroutes to a fixed target model."""

    def __init__(self, target: str) -> None:
        self.target = target

    async def route(self, ctx: RoutingContext) -> RoutingDecision:
        return RoutingDecision(ordered_models=[self.target], confidence=0.9, rationale="test reroute")


def _post_chat(client: TestClient, headers: dict[str, str], *, stream: bool = False) -> httpx.Response:
    resp: httpx.Response = client.post(
        "/v1/chat/completions",
        json={
            "model": REQUESTED_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        },
        headers=headers,
    )
    return resp


# ---------------------------------------------------------------------------
# Standalone, non-streaming
# ---------------------------------------------------------------------------


def test_routing_disabled_passes_requested_model_through(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Default config (router_backend='none'): provider gets the requested model."""
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured["model"] = kwargs["model"]
        return _completion()

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = _post_chat(client, api_key_header)

    assert resp.status_code == 200
    assert captured["model"] == REQUESTED_MODEL


def test_noop_backend_passes_requested_model_through(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-op backend echoes the requested model, so the provider call is unchanged."""
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured["model"] = kwargs["model"]
        return _completion()

    monkeypatch.setattr("gateway.api.routes.chat.get_router_backend", lambda config: NoOpRouterBackend())

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = _post_chat(client, api_key_header)

    assert resp.status_code == 200
    assert captured["model"] == REQUESTED_MODEL


def test_reroute_backend_changes_model_sent_to_provider(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerouting backend must change the model the provider actually receives.

    Regression guard: the routed model has to flow into the acompletion call,
    not just into provider-kwargs / pricing / logging.
    """
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured["model"] = kwargs["model"]
        return _completion(model="gpt-4o-mini")

    monkeypatch.setattr(
        "gateway.api.routes.chat.get_router_backend",
        lambda config: _RerouteBackend(REROUTE_TARGET),
    )

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = _post_chat(client, api_key_header)

    assert resp.status_code == 200
    assert captured["model"] == REROUTE_TARGET


def test_router_header_off_opts_a_request_out_of_routing(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Otari-Router: off` skips routing for one request, even with a backend enabled."""
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured["model"] = kwargs["model"]
        return _completion()

    monkeypatch.setattr(
        "gateway.api.routes.chat.get_router_backend",
        lambda config: _RerouteBackend(REROUTE_TARGET),
    )

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = _post_chat(client, {**api_key_header, "Otari-Router": "off"})

    assert resp.status_code == 200
    assert captured["model"] == REQUESTED_MODEL


def test_router_header_invalid_value_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognized `Otari-Router` value is a client error, rejected before dispatch."""
    monkeypatch.setattr(
        "gateway.api.routes.chat.get_router_backend",
        lambda config: _RerouteBackend(REROUTE_TARGET),
    )

    resp = _post_chat(client, {**api_key_header, "Otari-Router": "maybe"})

    assert resp.status_code == 400
    assert "Otari-Router" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Standalone, streaming
# ---------------------------------------------------------------------------


def test_reroute_backend_changes_model_on_streaming_path(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming standalone path must also honor the rerouted model."""
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        captured["model"] = kwargs["model"]

        async def _chunks() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk(
                id="c1", choices=[], created=0, model="gpt-4o-mini", object="chat.completion.chunk", usage=None
            )
            yield ChatCompletionChunk(
                id="c2",
                choices=[],
                created=0,
                model="gpt-4o-mini",
                object="chat.completion.chunk",
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        return _chunks()

    monkeypatch.setattr(
        "gateway.api.routes.chat.get_router_backend",
        lambda config: _RerouteBackend(REROUTE_TARGET),
    )

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = _post_chat(client, api_key_header, stream=True)
        # Drain the SSE body so the stream (and settlement) completes.
        body = resp.text

    assert resp.status_code == 200
    assert "data:" in body
    assert captured["model"] == REROUTE_TARGET


# ---------------------------------------------------------------------------
# Platform mode must not consult the router
# ---------------------------------------------------------------------------


@pytest.fixture
def platform_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    app = create_app(
        GatewayConfig(
            mode="platform",
            platform={"base_url": "http://platform.test/api/v1"},
        )
    )
    with TestClient(app) as client:
        yield client
    reset_config()
    reset_db()


def test_platform_mode_does_not_consult_router(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The platform dispatch path must never call the routing backend factory."""

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "11111111-1111-4111-8111-111111111111",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "11111111-1111-4111-8111-111111111111",
                            "position": 0,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-platform-key",
                            "api_base": "https://api.openai.com/v1",
                            "managed": True,
                        }
                    ],
                },
            )
        return httpx.Response(204)

    def _boom(config: Any) -> Any:
        raise AssertionError("router backend must not be consulted in platform mode")

    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured["model"] = kwargs["model"]
        return _completion(model="gpt-4o-mini")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.get_router_backend", _boom)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", mock_acompletion)

    resp = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    # If the router factory had been consulted, _boom would have raised and the
    # request would have failed. A 200 with the platform-resolved model proves
    # the platform path bypasses the seam entirely.
    assert resp.status_code == 200
    assert captured["model"] == "openai:gpt-4o-mini"


# ---------------------------------------------------------------------------
# Edge cases on the standalone path (mocked provider)
# ---------------------------------------------------------------------------


def test_routed_model_failure_surfaces_error_no_standalone_fallback(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterization: standalone has no fallback, so a routed model that fails
    surfaces an error rather than escalating. Documents the safety-net gap: a
    cheap reroute can turn a working request into a 502 if the cheap model is down.
    """

    async def boom_acompletion(**kwargs: Any) -> ChatCompletion:
        raise RuntimeError("cheap provider is down")

    monkeypatch.setattr(
        "gateway.api.routes.chat.get_router_backend",
        lambda config: _RerouteBackend(REROUTE_TARGET),
    )
    with patch("gateway.api.routes.chat.acompletion", new=boom_acompletion):
        resp = _post_chat(client, api_key_header)

    assert resp.status_code == 502


def test_usage_log_attributes_the_routed_model_not_the_requested_one(
    client: TestClient,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a reroute, cost/usage must be attributed to the model that actually
    ran, not the one the caller asked for.
    """

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        return _completion(model="gpt-4o-mini")

    monkeypatch.setattr(
        "gateway.api.routes.chat.get_router_backend",
        lambda config: _RerouteBackend(REROUTE_TARGET),
    )
    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = _post_chat(client, api_key_header)
    assert resp.status_code == 200

    user_id = api_key_obj["user_id"]
    usage = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage.status_code == 200
    rows = usage.json()
    assert rows, "expected a usage row for the routed call"
    latest = rows[0]
    assert latest["provider"] == "openai"
    assert latest["model"] == "gpt-4o-mini"  # the rerouted model, not gemini:gemini-2.5-flash


def test_request_params_preserved_under_reroute(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rerouting changes only the model; other request fields must pass through."""
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.update(kwargs)
        return _completion(model="gpt-4o-mini")

    monkeypatch.setattr(
        "gateway.api.routes.chat.get_router_backend",
        lambda config: _RerouteBackend(REROUTE_TARGET),
    )
    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": REQUESTED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.0,
                "max_tokens": 32,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    assert captured["model"] == REROUTE_TARGET
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 32


# ---------------------------------------------------------------------------
# Real OpenAI calls through the gateway (no mock); skipped without a key
# ---------------------------------------------------------------------------

_NO_KEY = not os.getenv("OPENAI_API_KEY")


@pytest.mark.skipif(_NO_KEY, reason="needs OPENAI_API_KEY")
def test_real_openai_passthrough_when_routing_disabled(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Default (routing off): a real call to a cheap OpenAI model still works."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o-mini",
            "messages": [{"role": "user", "content": "Reply with the single word: ping"}],
            "max_tokens": 5,
            "temperature": 0,
        },
        headers=api_key_header,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "mini" in body["model"]
    assert body["choices"][0]["message"]["content"]


@pytest.mark.skipif(_NO_KEY, reason="needs OPENAI_API_KEY")
def test_real_openai_reroute_serves_request_from_cheaper_model(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end with live models: a request for gpt-4o, rerouted to gpt-4o-mini,
    is actually answered by gpt-4o-mini (the response reports the mini model).
    """
    monkeypatch.setattr(
        "gateway.api.routes.chat.get_router_backend",
        lambda config: _RerouteBackend("openai:gpt-4o-mini"),
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": [{"role": "user", "content": "Reply with the single word: ping"}],
            "max_tokens": 5,
            "temperature": 0,
        },
        headers=api_key_header,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The caller asked for gpt-4o; the router sent it to gpt-4o-mini, and the
    # provider response reflects the model that actually ran.
    assert "mini" in body["model"], f"expected a mini model, got {body['model']}"
