"""A deployment provider the retired Deployment providers page stored keeps serving every path, end to end.

Each case leaves a row in the table that page wrote, as an upgrade finds it,
then starts the gateway, which moves the row onto a hosted provider of the same
name, and sends the request the row served. It must reach the same endpoint on
the same key: the move changes where the credential is stored, never what a
caller can reach with it. Model discovery is stubbed; each provider call is
captured at the any-llm boundary.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
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
from gateway.models.providers import HostedProvider, ProviderCredential
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import encrypt_secret, generate_secret_key

from .conftest import build_test_client, shared_test_config

VLLM_BASE = "http://vllm.test/v1"
EU_BASE = "https://eu.test/v1"
EU_KEY = "sk-eu-1234"
MASTER = {"Otari-Key": "Bearer test-master-key"}
VLLM = {"my-vllm": {"provider_type": "openai", "api_base": VLLM_BASE}}
EU = {"openai-eu": {"provider_type": "openai", "api_key": EU_KEY, "api_base": EU_BASE}}

Upgrade = Callable[[dict[str, dict[str, Any]]], AbstractContextManager[TestClient]]


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
def upgraded(test_config: GatewayConfig, db_session_factory: Callable[[], Session]) -> Upgrade:
    """Leave deployment providers in the retired table, then start the gateway, which moves them."""

    @contextmanager
    def _start(rows: dict[str, dict[str, Any]]) -> Iterator[TestClient]:
        with db_session_factory() as db:
            for instance, fields in rows.items():
                api_key = fields.get("api_key")
                db.add(
                    ProviderCredential(
                        instance=instance,
                        provider_type=fields["provider_type"],
                        api_base=fields.get("api_base"),
                        encrypted_api_key=encrypt_secret(api_key) if api_key else None,
                        last4=api_key[-4:] if api_key else None,
                        client_args={},
                        updated_at=datetime.now(UTC),
                    )
                )
            db.commit()
        for client in build_test_client(test_config):
            created = client.post(f"{API_ROOT}/users", json={"user_id": "u1"}, headers=MASTER)
            assert created.status_code == 200, created.text
            yield client

    return _start


def test_startup_moves_every_row_and_leaves_none_behind(
    upgraded: Upgrade, db_session_factory: Callable[[], Session]
) -> None:
    with upgraded({**VLLM, **EU}):
        pass

    with db_session_factory() as db:
        assert db.execute(select(ProviderCredential)).scalars().all() == []
        hosted = {row.name: row for row in db.execute(select(HostedProvider)).scalars()}
    assert set(hosted) == {"my-vllm", "openai-eu"}
    assert not any(row.serves_unlisted_models for row in hosted.values())


def test_a_row_startup_cannot_move_is_left_in_place(
    upgraded: Upgrade, db_session_factory: Callable[[], Session]
) -> None:
    """Named after another provider, so a hosted provider cannot keep its name; it stays for the operator."""
    with upgraded({"groq": {"provider_type": "openai", "api_key": "sk-groq-1234"}}) as client:
        [row] = client.get(f"{API_ROOT}/hosted-providers/migration", headers=MASTER).json()["rows"]

    assert row["outcome"] == "invalid"
    with db_session_factory() as db:
        assert [r.instance for r in db.execute(select(ProviderCredential)).scalars()] == ["groq"]


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


def _sent_to(call: Mapping[str, Any]) -> tuple[str | None, str | None]:
    return call.get("api_key"), call.get("api_base")


def test_a_keyless_unpriced_backend_serves_every_model_it_lists(upgraded: Upgrade) -> None:
    with upgraded(VLLM) as client:
        status, [sent] = _chat(client, MASTER, "my-vllm:llama-3")
        unlisted, calls = _chat(client, MASTER, "my-vllm:a-model-it-never-listed")

    assert status == 200
    assert _sent_to(sent) == ("otari-no-key-required", VLLM_BASE)
    # Off by default: a model the provider does not list is refused until it is added.
    assert (unlisted, calls) == (424, [])


def test_a_renamed_provider_keeps_its_name_endpoint_and_key(upgraded: Upgrade) -> None:
    with upgraded(EU) as client:
        status, [sent] = _chat(client, MASTER, "openai-eu:gpt-4o-mini")

    assert status == 200
    assert _sent_to(sent) == (EU_KEY, EU_BASE)


def test_a_model_switched_off_on_the_roster_is_refused(upgraded: Upgrade) -> None:
    with upgraded(VLLM) as client:
        offered = client.post(
            f"{API_ROOT}/hosted-providers/my-vllm/models", json={"model": "retired-model"}, headers=MASTER
        )
        assert offered.status_code == 201, offered.text
        switched = client.patch(
            f"{API_ROOT}/hosted-providers/my-vllm/models/{offered.json()['id']}",
            json={"enabled": False},
            headers=MASTER,
        )
        assert switched.status_code == 200, switched.text

        status, calls = _chat(client, MASTER, "my-vllm:retired-model")

    assert (status, calls) == (424, [])


def test_a_policy_falls_over_onto_a_moved_provider(
    upgraded: Upgrade, db_session_factory: Callable[[], Session]
) -> None:
    with upgraded(VLLM) as client:
        status, calls = _chat(client, MASTER, "fast")

    assert status == 200
    assert [call["model"] for call in calls] == ["openai:gpt-4o", "openai:llama-3"]
    assert _sent_to(calls[1]) == ("otari-no-key-required", VLLM_BASE)
    with db_session_factory() as db:
        served = db.execute(
            select(UsageLog.provider, UsageLog.deployment_paid).where(UsageLog.status == "success")
        ).all()
    # Served on the deployment's own credential, and recorded so.
    assert [tuple(row) for row in served] == [("my-vllm", True)]


def test_embeddings_reach_a_moved_provider(upgraded: Upgrade) -> None:
    embedded = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1], index=0, object="embedding")],
        model="nomic-embed",
        object="list",
        usage=Usage(prompt_tokens=3, total_tokens=3),
    )

    with (
        upgraded(VLLM) as client,
        patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock(return_value=embedded)) as call,
    ):
        response = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "my-vllm:nomic-embed", "input": "hi", "user": "u1"},
            headers=MASTER,
        )

    assert response.status_code == 200, response.text
    assert call.await_args is not None
    assert _sent_to(call.await_args.kwargs) == ("otari-no-key-required", VLLM_BASE)


class _SupportsBatch:
    SUPPORTS_BATCH = True


def test_a_batch_runs_and_settles_on_a_moved_provider(
    upgraded: Upgrade, db_session_factory: Callable[[], Session]
) -> None:
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
        upgraded(EU) as client,
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
        patch("gateway.api.routes.batches.acreate_batch", new=AsyncMock(return_value=batch)) as create,
        patch("gateway.api.routes.batches.aretrieve_batch", new=AsyncMock(return_value=batch)) as retrieve,
        patch("gateway.api.routes.batches.aretrieve_batch_results", new=results),
    ):
        # A batch is accounted once something prices it.
        priced = client.post(
            f"{API_ROOT}/pricing",
            json={"model_key": "openai-eu:gpt-4o-mini", "input_price_per_million": 1, "output_price_per_million": 2},
            headers=MASTER,
        )
        assert priced.status_code == 200, priced.text
        created = client.post(f"{API_ROOT}/batches", json=body, headers=MASTER)
        status = client.get(f"{API_ROOT}/batches/batch_eu", params={"provider": "openai-eu"}, headers=MASTER)
        settled = client.get(f"{API_ROOT}/batches/batch_eu/results", params={"provider": "openai-eu"}, headers=MASTER)

    assert (created.status_code, status.status_code, settled.status_code) == (200, 200, 200), settled.text
    assert created.json()["provider"] == "openai-eu"
    for call in (create, retrieve, results):
        assert call.await_args is not None
        assert _sent_to(call.await_args.kwargs) == (EU_KEY, EU_BASE)
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


def test_a_policy_or_an_alias_may_target_a_moved_provider(upgraded: Upgrade) -> None:
    """The name was accepted while the deployment provider served it, so it still is."""
    spec = {"select": [{"default": "openai:gpt-4o"}], "on_failure": ["my-vllm:llama-3"]}
    with upgraded(VLLM) as client:
        policy = client.post(f"{API_ROOT}/routing/policies", json={"name": "local", "spec": spec}, headers=MASTER)
        alias = client.post(f"{API_ROOT}/aliases", json={"name": "house", "target": "my-vllm:llama-3"}, headers=MASTER)
        unknown = client.post(f"{API_ROOT}/aliases", json={"name": "nope", "target": "my-nas:llama-3"}, headers=MASTER)

    assert (policy.status_code, alias.status_code) == (200, 200), (policy.text, alias.text)
    assert unknown.status_code == 400
