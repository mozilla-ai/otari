"""A deployment provider moved onto a hosted provider keeps serving every path it served, end to end.

Each case stores a deployment provider through ``/provider-credentials`` as the
Deployment providers page does, sends a request that it serves, moves it onto
hosted providers, and sends the same request again. The second request must
reach the same endpoint on the same key: the move changes where the credential
is stored, never what a caller can reach with it. Model discovery is stubbed;
each provider call is captured at the any-llm boundary.
"""

from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.batch import Batch
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Usage,
)
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

from .conftest import shared_test_config

MOVE = f"{API_ROOT}/hosted-providers/migration"
VLLM_BASE = "http://vllm.test/v1"
EU_BASE = "https://eu.test/v1"
EU_KEY = "sk-eu-1234"


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """``openai`` from config.yml, and a policy that falls over from it onto the stored ``my-vllm``."""
    config = shared_test_config(postgres_url, {"openai": {"api_key": "sk-config-openai"}})
    return config.model_copy(
        update={
            "routing": RoutingConfig.model_validate(
                {
                    "policies": {
                        "fast": {
                            "select": [{"default": "openai:gpt-4o"}],
                            "on_failure": ["my-vllm:llama-3"],
                        }
                    }
                }
            )
        }
    )


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    async def _discover(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name,
            models=[
                Model(id=model, object="model", created=0, owned_by=impl_name)
                for model in ("llama-3", "nomic-embed", "gpt-4o-mini")
            ],
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _discover)
    yield


@pytest.fixture
def stored(client: TestClient, master_key_header: dict[str, str]) -> Callable[..., None]:
    """Store a deployment provider as the Deployment providers page does, and the user the requests bill."""
    created = client.post(f"{API_ROOT}/users", json={"user_id": "u1"}, headers=master_key_header)
    assert created.status_code == 200, created.text

    def _store(instance: str, **body: Any) -> None:
        response = client.post(
            f"{API_ROOT}/provider-credentials", json={"instance": instance, **body}, headers=master_key_header
        )
        assert response.status_code == 201, response.text

    return _store


def _move(client: TestClient, headers: dict[str, str]) -> None:
    rows = client.post(MOVE, headers=headers).json()["rows"]
    assert all(row["outcome"] == "migrated" and row["source_removed"] for row in rows), rows
    assert client.get(f"{API_ROOT}/provider-credentials", headers=headers).json() == []


def _completion(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="c",
        object="chat.completion",
        created=0,
        model=model,
        choices=[Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="Hi"))],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat(client: TestClient, headers: dict[str, str], model: str) -> tuple[int, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    async def _call(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        if kwargs["model"] == "openai:gpt-4o":
            request = httpx.Request("POST", "http://upstream")
            raise httpx.HTTPStatusError("down", request=request, response=httpx.Response(503, request=request))
        return _completion(kwargs["model"].split(":", 1)[1])

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_call)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=headers,
        )
    return response.status_code, calls


def _sent_to(call: dict[str, Any]) -> tuple[str | None, str | None]:
    return call.get("api_key"), call.get("api_base")


def test_a_keyless_unpriced_backend_serves_every_model_it_lists(
    client: TestClient, master_key_header: dict[str, str], stored: Callable[..., None]
) -> None:
    stored("my-vllm", provider_type="openai", api_base=VLLM_BASE)
    before, [sent_before] = _chat(client, master_key_header, "my-vllm:llama-3")

    _move(client, master_key_header)
    after, [sent_after] = _chat(client, master_key_header, "my-vllm:llama-3")
    unlisted, calls = _chat(client, master_key_header, "my-vllm:a-model-it-never-listed")

    assert (before, after) == (200, 200)
    assert _sent_to(sent_after) == _sent_to(sent_before)
    assert sent_after["api_base"] == VLLM_BASE
    # Off by default: a model the provider does not list is refused until it is added.
    assert (unlisted, calls) == (424, [])


def test_a_renamed_provider_keeps_its_name_endpoint_and_key(
    client: TestClient, master_key_header: dict[str, str], stored: Callable[..., None]
) -> None:
    stored("openai-eu", provider_type="openai", api_key=EU_KEY, api_base=EU_BASE)
    before, [sent_before] = _chat(client, master_key_header, "openai-eu:gpt-4o-mini")

    _move(client, master_key_header)
    after, [sent_after] = _chat(client, master_key_header, "openai-eu:gpt-4o-mini")

    assert (before, after) == (200, 200)
    assert _sent_to(sent_after) == _sent_to(sent_before) == (EU_KEY, EU_BASE)


def test_a_model_switched_off_on_the_roster_is_refused(
    client: TestClient, master_key_header: dict[str, str], stored: Callable[..., None]
) -> None:
    stored("my-vllm", provider_type="openai", api_base=VLLM_BASE)
    _move(client, master_key_header)
    roster = client.get(f"{API_ROOT}/hosted-providers/my-vllm/models", headers=master_key_header).json()["data"]
    [listed] = [row for row in roster if row["model"] == "llama-3"]
    switched = client.patch(
        f"{API_ROOT}/hosted-providers/my-vllm/models/{listed['id']}", json={"enabled": False}, headers=master_key_header
    )
    assert switched.status_code == 200, switched.text

    status, calls = _chat(client, master_key_header, "my-vllm:llama-3")

    assert (status, calls) == (424, [])


def test_a_policy_falls_over_onto_a_moved_provider(
    client: TestClient,
    master_key_header: dict[str, str],
    stored: Callable[..., None],
    db_session_factory: Callable[[], Session],
) -> None:
    stored("my-vllm", provider_type="openai", api_base=VLLM_BASE)
    before, calls_before = _chat(client, master_key_header, "fast")

    _move(client, master_key_header)
    after, calls_after = _chat(client, master_key_header, "fast")

    assert (before, after) == (200, 200)
    assert [call["model"] for call in calls_after] == ["openai:gpt-4o", "openai:llama-3"]
    assert _sent_to(calls_after[1]) == _sent_to(calls_before[1])
    session = db_session_factory()
    try:
        served = session.execute(
            select(UsageLog.provider, UsageLog.deployment_paid).where(UsageLog.status == "success")
        ).all()
    finally:
        session.close()
    # Served on the deployment's own credential once moved, and recorded so.
    assert sorted(served) == [("my-vllm", False), ("my-vllm", True)]


def test_embeddings_reach_a_moved_provider(
    client: TestClient, master_key_header: dict[str, str], stored: Callable[..., None]
) -> None:
    stored("my-vllm", provider_type="openai", api_base=VLLM_BASE)
    embedded = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1], index=0, object="embedding")],
        model="nomic-embed",
        object="list",
        usage=Usage(prompt_tokens=3, total_tokens=3),
    )

    def _embed() -> tuple[int, dict[str, Any]]:
        with patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock(return_value=embedded)) as call:
            response = client.post(
                f"{API_ROOT}/embeddings",
                json={"model": "my-vllm:nomic-embed", "input": "hi", "user": "u1"},
                headers=master_key_header,
            )
        return response.status_code, dict(call.await_args.kwargs) if call.await_args else {}

    before, sent_before = _embed()
    _move(client, master_key_header)
    after, sent_after = _embed()

    assert (before, after) == (200, 200)
    assert _sent_to(sent_after) == _sent_to(sent_before)


class _SupportsBatch:
    SUPPORTS_BATCH = True


def test_a_batch_runs_and_settles_on_a_moved_provider(
    client: TestClient,
    master_key_header: dict[str, str],
    stored: Callable[..., None],
    db_session_factory: Callable[[], Session],
) -> None:
    stored("openai-eu", provider_type="openai", api_key=EU_KEY, api_base=EU_BASE)
    _move(client, master_key_header)
    # A batch is accounted once something prices it.
    priced = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "openai-eu:gpt-4o-mini", "input_price_per_million": 1, "output_price_per_million": 2},
        headers=master_key_header,
    )
    assert priced.status_code == 200, priced.text
    batch = Batch(
        id="batch_eu",
        object="batch",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        status="completed",
        created_at=0,
        input_file_id="file-1",
    )
    body = {
        "model": "openai-eu:gpt-4o-mini",
        "requests": [{"custom_id": "r1", "body": {"messages": [{"role": "user", "content": "Hi"}]}}],
        "user": "u1",
    }
    results = AsyncMock()
    results.return_value.results = [
        type("Line", (), {"custom_id": "r1", "result": _completion("gpt-4o-mini"), "error": None})()
    ]

    with (
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
        patch("gateway.api.routes.batches.acreate_batch", new=AsyncMock(return_value=batch)) as create,
        patch("gateway.api.routes.batches.aretrieve_batch", new=AsyncMock(return_value=batch)) as retrieve,
        patch("gateway.api.routes.batches.aretrieve_batch_results", new=results),
    ):
        created = client.post(f"{API_ROOT}/batches", json=body, headers=master_key_header)
        status = client.get(f"{API_ROOT}/batches/batch_eu", params={"provider": "openai-eu"}, headers=master_key_header)
        settled = client.get(
            f"{API_ROOT}/batches/batch_eu/results", params={"provider": "openai-eu"}, headers=master_key_header
        )

    assert (created.status_code, status.status_code, settled.status_code) == (200, 200, 200), settled.text
    assert created.json()["provider"] == "openai-eu"
    for call in (create, retrieve, results):
        assert call.await_args is not None
        assert _sent_to(dict(call.await_args.kwargs)) == (EU_KEY, EU_BASE)
    session = db_session_factory()
    try:
        rows = session.execute(
            select(UsageLog.provider, UsageLog.deployment_paid, UsageLog.cost).where(
                UsageLog.endpoint == "/v1/batches/results"
            )
        ).all()
    finally:
        session.close()
    [(provider, deployment_paid, cost)] = rows
    assert (provider, deployment_paid) == ("openai-eu", True)
    assert cost is not None and cost > 0
