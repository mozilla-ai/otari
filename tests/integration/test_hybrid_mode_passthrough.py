"""Hybrid-mode integration tests for the pass-through endpoints.

Embeddings, rerank, image generation, audio and moderations run on a hybrid
gateway the way a completion does: the control plane resolves the attempts, and
the gateway reports each outcome back with the token and unit counts it is
priced on.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from typing import Any

import httpx
import pytest
from any_llm.types.audio import Transcription
from any_llm.types.completion import CreateEmbeddingResponse, Embedding, Usage
from any_llm.types.image import Image, ImagesResponse
from any_llm.types.rerank import RerankMeta, RerankResponse, RerankResult, RerankUsage
from fastapi.testclient import TestClient
from openai.types.images_response import Usage as ImageUsage
from openai.types.images_response import UsageInputTokensDetails

from conftest import InstallControlPlane
from gateway.api.deps import reset_config
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.models.routing import RoutingConfig
from gateway.types.moderation import ModerationResponse, ModerationResult

from .conftest import app_for

_AUTH = {"Authorization": "Bearer user_test_token"}


@pytest.fixture
def platform_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    app = app_for(
        GatewayConfig(
            mode="hybrid",
            platform={"base_url": "http://platform.test/api/v1"},
        )
    )

    with TestClient(app) as client:
        yield client

    reset_config()
    reset_db()


def _attempt(attempt_id: str, model: str, *, position: int = 0, provider: str = "openai") -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "position": position,
        "provider": provider,
        "model": model,
        "api_key": f"sk-{attempt_id}",
        "api_base": None,
        "managed": True,
    }


class _ControlPlane:
    """A control plane that resolves to ``attempts`` and records every usage report."""

    def __init__(self, attempts: list[dict[str, Any]], *, cost_usd: str | None = "0.000020") -> None:
        self.attempts = attempts
        self.cost_usd = cost_usd
        self.resolves: list[dict[str, Any]] = []
        self.reports: list[dict[str, Any]] = []

    async def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            self.resolves.append(body)
            return httpx.Response(
                200, json={"request_id": "req-1", "fallback_enabled": True, "attempts": self.attempts}
            )
        self.reports.append(body)
        if body["status"] != "success" or self.cost_usd is None:
            return httpx.Response(204)
        return httpx.Response(
            200,
            json={
                "correlation_id": body["correlation_id"],
                "status": "completed",
                "outcome": "success",
                "cost_usd": self.cost_usd,
                "currency": "USD",
                "usage_status": "reported",
                "pricing": {"source": "managed"},
            },
        )

    def wait_for_reports(self, count: int) -> list[dict[str, Any]]:
        """The reports, once ``count`` have arrived: an earlier attempt's is sent in the background."""
        deadline = time.monotonic() + 5
        while len(self.reports) < count and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.reports


def _embedding_response(prompt_tokens: int = 10) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


def _fake(result: Any, calls: list[dict[str, Any]] | None = None) -> Callable[..., Any]:
    async def call(**kwargs: Any) -> Any:
        if calls is not None:
            calls.append(kwargs)
        return result

    return call


def test_embeddings_runs_on_the_resolved_attempt_and_reports_its_tokens(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    plane = _ControlPlane([_attempt("att-1", "text-embedding-3-small")])
    control_plane_transport(plane)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("gateway.api.routes.embeddings.aembedding", _fake(_embedding_response(12), calls))

    response = platform_client.post(
        f"{API_ROOT}/embeddings",
        json={"model": "openai:text-embedding-3-small", "input": "hello"},
        headers=_AUTH,
    )

    assert response.status_code == 200, response.text
    assert response.headers["Otari-Request-ID"] == "req-1"
    assert response.headers["Otari-Attempt-ID"] == "att-1"
    assert response.json()["usage"] == {
        "prompt_tokens": 12,
        "total_tokens": 12,
        "cost_usd": "0.000020",
        "pricing_source": "managed",
    }
    assert plane.resolves == [{"provider": "openai", "model": "text-embedding-3-small"}]
    assert calls[0]["api_key"] == "sk-att-1"
    assert calls[0]["model"] == "text-embedding-3-small"
    assert plane.reports == [
        {
            "correlation_id": "att-1",
            "status": "success",
            "is_final_attempt": True,
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 0,
                "total_tokens": 12,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        }
    ]


def test_embeddings_without_a_settled_price_carries_no_inline_cost(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    control_plane_transport(_ControlPlane([_attempt("att-1", "text-embedding-3-small")], cost_usd=None))
    monkeypatch.setattr("gateway.api.routes.embeddings.aembedding", _fake(_embedding_response()))

    response = platform_client.post(
        f"{API_ROOT}/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hello"}, headers=_AUTH
    )

    assert response.status_code == 200, response.text
    assert response.json()["usage"] == {"prompt_tokens": 10, "total_tokens": 10}


def test_embeddings_falls_back_and_reports_the_failed_attempt(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    plane = _ControlPlane(
        [
            _attempt("att-a", "embed-english-v3.0", provider="cohere"),
            _attempt("att-b", "text-embedding-3-small", position=1),
        ]
    )
    control_plane_transport(plane)

    async def aembedding(**kwargs: Any) -> CreateEmbeddingResponse:
        if kwargs["provider"].value == "cohere":
            raise RuntimeError("simulated upstream failure")
        return _embedding_response()

    monkeypatch.setattr("gateway.api.routes.embeddings.aembedding", aembedding)

    response = platform_client.post(
        f"{API_ROOT}/embeddings", json={"model": "text-embedding", "input": "hello"}, headers=_AUTH
    )

    assert response.status_code == 200, response.text
    assert response.headers["Otari-Attempt-ID"] == "att-b"
    reports = {report["correlation_id"]: report for report in plane.wait_for_reports(2)}
    assert reports["att-a"] == {
        "correlation_id": "att-a",
        "status": "error",
        "is_final_attempt": False,
        "error_class": "unknown",
    }
    assert reports["att-b"]["status"] == "success"


def test_embeddings_answers_502_and_reports_every_attempt_when_all_fail(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    plane = _ControlPlane(
        [
            _attempt("att-a", "embed-english-v3.0", provider="cohere"),
            _attempt("att-b", "text-embedding-3-small", position=1),
        ]
    )
    control_plane_transport(plane)

    async def aembedding(**_kwargs: Any) -> CreateEmbeddingResponse:
        raise RuntimeError("simulated upstream failure")

    monkeypatch.setattr("gateway.api.routes.embeddings.aembedding", aembedding)

    response = platform_client.post(
        f"{API_ROOT}/embeddings", json={"model": "text-embedding", "input": "hello"}, headers=_AUTH
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "All upstream providers failed"}
    assert [(r["correlation_id"], r["status"], r["is_final_attempt"]) for r in plane.reports] == [
        ("att-a", "error", False),
        ("att-b", "error", True),
    ]


def test_embeddings_refusal_is_answered_as_standalone_answers_it(
    platform_client: TestClient,
    control_plane_transport: InstallControlPlane,
) -> None:
    """An option the provider does not take is a refusal of the request, not a provider failure to fall through on."""
    plane = _ControlPlane(
        [_attempt("att-a", "text-embedding-3-small"), _attempt("att-b", "text-embedding-3-large", position=1)]
    )
    control_plane_transport(plane)

    response = platform_client.post(
        f"{API_ROOT}/embeddings",
        json={"model": "text-embedding", "input": "hello", "input_type": "search_query"},
        headers=_AUTH,
    )

    assert response.status_code == 400
    assert "input_type" in response.json()["detail"]
    assert plane.reports == [{"correlation_id": "att-a", "status": "error", "is_final_attempt": True}]


def test_rerank_reports_the_billed_search_units(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    plane = _ControlPlane([_attempt("att-1", "rerank-v3.5", provider="cohere")])
    control_plane_transport(plane)
    result = RerankResponse(
        id="rerank-1",
        model="rerank-v3.5",
        results=[RerankResult(index=0, relevance_score=0.9)],
        meta=RerankMeta(billed_units={"search_units": 3}),
        usage=RerankUsage(total_tokens=40),
    )
    monkeypatch.setattr("gateway.api.routes.rerank.arerank", _fake(result))

    response = platform_client.post(
        f"{API_ROOT}/rerank",
        json={"model": "cohere:rerank-v3.5", "query": "q", "documents": ["a", "b"]},
        headers=_AUTH,
    )

    assert response.status_code == 200, response.text
    assert response.json()["usage"]["cost_usd"] == "0.000020"
    assert response.json()["usage"]["pricing_source"] == "managed"
    assert plane.reports[0]["usage"] == {
        "prompt_tokens": 40,
        "completion_tokens": 0,
        "total_tokens": 40,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    assert plane.reports[0]["units"] == {"search_units": 3}


def test_image_generation_reports_the_images_returned(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    plane = _ControlPlane([_attempt("att-1", "gpt-image-1")])
    control_plane_transport(plane)
    result = ImagesResponse(
        created=0, data=[Image(url="https://example.com/a.png"), Image(url="https://example.com/b.png")]
    )
    monkeypatch.setattr("gateway.api.routes.images.aimage_generation", _fake(result))

    response = platform_client.post(
        f"{API_ROOT}/images/generations", json={"model": "openai:gpt-image-1", "prompt": "a cat", "n": 2}, headers=_AUTH
    )

    assert response.status_code == 200, response.text
    assert response.headers["Otari-Attempt-ID"] == "att-1"
    # Nobody reads the cost on an image response, so the report is sent without waiting for it.
    [report] = plane.wait_for_reports(1)
    assert "usage" not in report
    assert report["units"] == {"images": 2}


def test_moderation_reports_one_request(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    plane = _ControlPlane([_attempt("att-1", "omni-moderation-latest")])
    control_plane_transport(plane)
    result = ModerationResponse(
        id="modr-1",
        model="omni-moderation-latest",
        results=[ModerationResult(flagged=False, categories={}, category_scores={})],
    )
    monkeypatch.setattr("gateway.api.routes.moderations.amoderation", _fake(result))

    response = platform_client.post(
        f"{API_ROOT}/moderations", json={"model": "openai:omni-moderation-latest", "input": "hello"}, headers=_AUTH
    )

    assert response.status_code == 200, response.text
    [report] = plane.wait_for_reports(1)
    assert "usage" not in report
    assert report["units"] == {"requests": 1}


def test_speech_carries_the_request_and_attempt_ids(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    """Speech returns a response of its own, so the ids travel on the outcome's headers."""
    plane = _ControlPlane([_attempt("att-1", "gpt-4o-mini-tts")])
    control_plane_transport(plane)
    monkeypatch.setattr("gateway.api.routes.audio.aspeech", _fake(b"audio-bytes"))

    response = platform_client.post(
        f"{API_ROOT}/audio/speech",
        json={"model": "openai:gpt-4o-mini-tts", "input": "hello", "voice": "alloy"},
        headers=_AUTH,
    )

    assert response.status_code == 200, response.text
    assert response.content == b"audio-bytes"
    assert response.headers["Otari-Request-ID"] == "req-1"
    assert response.headers["Otari-Attempt-ID"] == "att-1"
    [report] = plane.wait_for_reports(1)
    assert "usage" not in report
    assert report["units"] == {"requests": 1}


def test_a_passthrough_request_needs_a_credential(platform_client: TestClient) -> None:
    response = platform_client.post(
        f"{API_ROOT}/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hi"}
    )

    assert response.status_code == 401


def test_a_control_plane_refusal_is_answered_with_its_status(
    platform_client: TestClient,
    control_plane_transport: InstallControlPlane,
) -> None:
    async def refuse(url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> httpx.Response:
        return httpx.Response(403, json={"detail": "Model not allowed for this key"})

    control_plane_transport(refuse)

    response = platform_client.post(
        f"{API_ROOT}/rerank", json={"model": "cohere:rerank-v3.5", "query": "q", "documents": ["a"]}, headers=_AUTH
    )

    assert response.status_code == 403


def test_transcription_sends_the_whole_upload_to_the_attempt_it_falls_back_to(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    plane = _ControlPlane(
        [_attempt("att-a", "whisper-1", provider="mistral"), _attempt("att-b", "whisper-1", position=1)]
    )
    control_plane_transport(plane)
    received: list[bytes] = []

    async def atranscription(**kwargs: Any) -> Transcription:
        received.append(kwargs["file"].read())
        if kwargs["provider"].value == "mistral":
            raise RuntimeError("simulated upstream failure")
        return Transcription(text="hello")

    monkeypatch.setattr("gateway.api.routes.audio.atranscription", atranscription)

    response = platform_client.post(
        f"{API_ROOT}/audio/transcriptions",
        data={"model": "whisper-1"},
        files={"file": ("a.mp3", b"audio-bytes", "audio/mpeg")},
        headers=_AUTH,
    )

    assert response.status_code == 200, response.text
    assert received == [b"audio-bytes", b"audio-bytes"]
    assert response.headers["Otari-Attempt-ID"] == "att-b"


def test_a_token_billed_image_reports_its_tokens_beside_its_count(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    """The control plane prices gpt-image-1 on its tokens, so they travel with the count."""
    plane = _ControlPlane([_attempt("att-1", "gpt-image-1")])
    control_plane_transport(plane)
    result = ImagesResponse(
        created=0,
        data=[Image(b64_json="aGk=")],
        usage=ImageUsage(
            input_tokens=50,
            output_tokens=4160,
            total_tokens=4210,
            input_tokens_details=UsageInputTokensDetails(image_tokens=0, text_tokens=50),
        ),
    )
    monkeypatch.setattr("gateway.api.routes.images.aimage_generation", _fake(result))

    response = platform_client.post(
        f"{API_ROOT}/images/generations", json={"model": "openai:gpt-image-1", "prompt": "a cat"}, headers=_AUTH
    )

    assert response.status_code == 200, response.text
    [report] = plane.wait_for_reports(1)
    assert report["units"] == {"images": 1}
    assert report["usage"]["prompt_tokens"] == 50
    assert report["usage"]["completion_tokens"] == 4160


@pytest.mark.parametrize(
    ("cost_usd", "source"),
    [("-0.000020", "managed"), ("0.000020", "Managed Price List"), ("0.000020", "x" * 33)],
)
def test_a_malformed_settlement_attaches_no_inline_cost(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
    cost_usd: str,
    source: str,
) -> None:
    """A negative amount or an odd source name from the peer never reaches the caller."""

    async def settle(url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={"request_id": "req-1", "fallback_enabled": True, "attempts": [_attempt("att-1", "e")]},
            )
        return httpx.Response(
            200,
            json={
                "correlation_id": body["correlation_id"],
                "status": "completed",
                "outcome": "success",
                "cost_usd": cost_usd,
                "currency": "USD",
                "usage_status": "reported",
                "pricing": {"source": source},
            },
        )

    control_plane_transport(settle)
    monkeypatch.setattr("gateway.api.routes.embeddings.aembedding", _fake(_embedding_response()))

    response = platform_client.post(
        f"{API_ROOT}/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hi"}, headers=_AUTH
    )

    assert response.status_code == 200, response.text
    assert response.json()["usage"] == {"prompt_tokens": 10, "total_tokens": 10}


def test_an_attempt_naming_a_provider_this_gateway_cannot_reach_falls_through(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    """A newer control plane can name a provider an older gateway predates: a capability gap, not a crash."""
    plane = _ControlPlane(
        [
            _attempt("att-a", "future-embed", provider="brand-new-provider"),
            _attempt("att-b", "text-embedding-3-small", position=1),
        ]
    )
    control_plane_transport(plane)
    monkeypatch.setattr("gateway.api.routes.embeddings.aembedding", _fake(_embedding_response()))

    response = platform_client.post(
        f"{API_ROOT}/embeddings", json={"model": "text-embedding", "input": "hello"}, headers=_AUTH
    )

    assert response.status_code == 200, response.text
    assert response.headers["Otari-Attempt-ID"] == "att-b"
    reports = {report["correlation_id"]: report for report in plane.wait_for_reports(2)}
    assert reports["att-a"] == {
        "correlation_id": "att-a",
        "status": "error",
        "is_final_attempt": False,
        "error_class": "unsupported_provider",
    }


def test_a_lone_attempt_naming_an_unknown_provider_is_a_502_not_a_crash(
    platform_client: TestClient,
    control_plane_transport: InstallControlPlane,
) -> None:
    plane = _ControlPlane([_attempt("att-a", "future-embed", provider="brand-new-provider")])
    control_plane_transport(plane)

    response = platform_client.post(
        f"{API_ROOT}/embeddings", json={"model": "future-embed", "input": "hello"}, headers=_AUTH
    )

    assert response.status_code == 502
    assert plane.reports == [
        {"correlation_id": "att-a", "status": "error", "is_final_attempt": True, "error_class": "unsupported_provider"}
    ]


def test_a_local_routing_policy_name_is_refused_before_the_control_plane_is_asked(
    monkeypatch: pytest.MonkeyPatch,
    control_plane_transport: InstallControlPlane,
) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    plane = _ControlPlane([_attempt("att-1", "text-embedding-3-small")])
    control_plane_transport(plane)
    config = GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://platform.test/api/v1"},
        routing=RoutingConfig.model_validate({"policies": {"smart": {"select": [{"default": "openai:gpt-4o-mini"}]}}}),
    )

    with TestClient(app_for(config)) as client:
        response = client.post(f"{API_ROOT}/embeddings", json={"model": "smart", "input": "hi"}, headers=_AUTH)
    reset_config()
    reset_db()

    assert response.status_code == 400
    assert "cannot be used in hybrid mode" in response.json()["detail"]
    assert plane.resolves == []
