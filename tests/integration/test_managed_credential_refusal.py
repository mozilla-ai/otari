"""A model offered only on managed credentials is refused by name where they do not serve.

Managed (deployment-paid) credentials serve only a single-model Chat
Completions, Messages or Responses request. The pass-through endpoints, batches
and a routing policy with fallbacks never ask ``ModelProviderPort`` to serve,
so a model the deployment offers there is refused with
``managed_credential_unsupported`` rather than the ``provider_not_configured``
that tells the caller to declare a provider it cannot. A plain build, whose
core adapter offers nothing, answers exactly as before.
"""

import concurrent.futures
import itertools
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.error_codes import (
    ERROR_CODE_HEADER,
    MANAGED_CREDENTIAL_UNSUPPORTED,
    PROVIDER_NOT_CONFIGURED,
)
from gateway.models.budgets import RESERVATION_ACTIVE, BudgetReservation
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog

from .conftest import build_test_client

MODULE = "probe_managed_refusal"
PROBE_BOOTSTRAP = """
import asyncio

from gateway.container import Container
from gateway.ports.model_provider_port import HostedCredential, ModelProviderPort

RESOLVES = []
FAULTS = set()


class Fleet:
    def __init__(self, session):
        del session

    async def resolve_hosted_credential(self, *, organization_id, workspace_id, provider, model):
        RESOLVES.append((provider, model))
        return HostedCredential(api_key="fleet-key", api_base=None, response_provider=provider)

    async def get_hosted_models(self, *, organization_id):
        if "cancel" in FAULTS:
            raise asyncio.CancelledError
        return {"openai": None}


def register(container: Container) -> None:
    container.bind(ModelProviderPort, Fleet)
"""
MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_NAMES = (f"{MODULE}_{n}" for n in itertools.count())

ROUTING = {
    "policies": {
        "managed_only": {"select": [{"default": "openai:gpt-5-mini"}], "on_failure": ["openai:gpt-5-nano"]},
        "managed_then_declared": {
            "select": [{"default": "openai:gpt-5-mini"}],
            "on_failure": ["anthropic:claude-haiku-4-5"],
        },
    }
}


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _config(postgres_url: str, bootstrap: str | None, *, require_pricing: bool = False) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=require_pricing,
        model_discovery=False,
        providers={"anthropic": {"api_key": "sk-ant"}},
        aliases={"house-embed": "openai:text-embedding-3-small"},
        routing=RoutingConfig.model_validate(ROUTING),
        bootstrap=bootstrap,
    )


def _overlay_client(
    postgres_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, require_pricing: bool = False
) -> Generator[tuple[TestClient, ModuleType]]:
    name = next(_NAMES)
    (tmp_path / f"{name}.py").write_text(PROBE_BOOTSTRAP)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    try:
        config = _config(postgres_url, f"{name}:register", require_pricing=require_pricing)
        for client in build_test_client(config):
            yield client, sys.modules[name]
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def managed(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, ModuleType]]:
    yield from _overlay_client(postgres_url, tmp_path, monkeypatch)


@pytest.fixture
def managed_priced(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, ModuleType]]:
    """The overlay gateway with ``require_pricing`` on, and nothing priced."""
    yield from _overlay_client(postgres_url, tmp_path, monkeypatch, require_pricing=True)


@pytest.fixture
def plain(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    yield from build_test_client(_config(postgres_url, None))


def _key(client: TestClient) -> dict[str, str]:
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 100.0}, headers=MASTER)
    assert budget.status_code == status.HTTP_200_OK, budget.text
    user = client.post(
        f"{API_ROOT}/users", json={"user_id": "caller", "budget_id": budget.json()["budget_id"]}, headers=MASTER
    )
    assert user.status_code == status.HTTP_200_OK, user.text
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "caller", "user_id": "caller"}, headers=MASTER)
    assert created.status_code == status.HTTP_200_OK, created.text
    return {API_KEY_HEADER: f"Bearer {created.json()['key']}"}


def _embeddings(
    client: TestClient, headers: dict[str, str], model: str = "openai:text-embedding-3-small"
) -> tuple[Any, AsyncMock]:
    provider = AsyncMock(side_effect=AssertionError("the provider must not be called"))
    with patch("gateway.api.routes.embeddings.aembedding", new=provider):
        response = client.post(f"{API_ROOT}/embeddings", json={"model": model, "input": "hi"}, headers=headers)
    return response, provider


def _batch(client: TestClient, headers: dict[str, str]) -> tuple[Any, AsyncMock]:
    provider = AsyncMock(side_effect=AssertionError("the provider must not be called"))
    with patch("gateway.api.routes.batches.acreate_batch", new=provider):
        response = client.post(
            f"{API_ROOT}/batches",
            json={"model": "openai:gpt-5-mini", "requests": [{"custom_id": "a", "body": {"messages": []}}]},
            headers=headers,
        )
    return response, provider


def _chat(client: TestClient, headers: dict[str, str], model: str, provider: Callable[..., Any]) -> Any:
    with patch("gateway.api.routes.chat.acompletion", new=provider):
        return client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
            headers=headers,
        )


def _active_reservations(make_session: Callable[[], Session]) -> int:
    with make_session() as db:
        return len(list(db.scalars(select(BudgetReservation).where(BudgetReservation.status == RESERVATION_ACTIVE))))


def _rows(make_session: Callable[[], Session], endpoint: str) -> list[UsageLog]:
    with make_session() as db:
        return list(db.scalars(select(UsageLog).where(UsageLog.endpoint == endpoint)))


def _assert_managed(response: Any, selector: str, surface: str) -> None:
    assert response.status_code == status.HTTP_424_FAILED_DEPENDENCY, response.text
    assert response.headers[ERROR_CODE_HEADER] == MANAGED_CREDENTIAL_UNSUPPORTED
    assert response.json()["detail"] == (
        f"'{selector}' is offered on this deployment's managed credentials only through Chat Completions, "
        f"Messages and Responses with a single model, not through {surface}. "
        "Use one of those, or add your own provider key."
    )


def test_embeddings_for_a_managed_model_is_refused_by_name(
    managed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = managed
    response, provider = _embeddings(client, _key(client))

    _assert_managed(response, "openai:text-embedding-3-small", "/v1/embeddings")
    provider.assert_not_called()
    assert probe.RESOLVES == []
    assert _active_reservations(db_session_factory) == 0
    rows = _rows(db_session_factory, "/v1/embeddings")
    assert [(row.status_code, row.error_message) for row in rows] == [(424, response.json()["detail"])]


def test_a_batch_for_a_managed_model_is_refused_by_name(
    managed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = managed
    response, provider = _batch(client, _key(client))

    _assert_managed(response, "openai:gpt-5-mini", "batches")
    provider.assert_not_called()
    assert probe.RESOLVES == []
    # Refused before anything is reserved, as the batch route's other refusals are.
    assert _active_reservations(db_session_factory) == 0


def test_a_policy_whose_every_candidate_is_managed_is_refused_by_name(
    managed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = managed
    provider = AsyncMock(side_effect=AssertionError("the provider must not be called"))
    response = _chat(client, _key(client), "managed_only", provider)

    assert response.status_code == status.HTTP_424_FAILED_DEPENDENCY, response.text
    assert response.headers[ERROR_CODE_HEADER] == MANAGED_CREDENTIAL_UNSUPPORTED
    # Named by the policy the caller wrote, once, and never by the targets behind it.
    _assert_managed(response, "managed_only", "a routing policy with fallbacks")
    assert "gpt-5" not in response.text
    assert "openai" not in response.text
    provider.assert_not_called()
    assert probe.RESOLVES == []
    assert _active_reservations(db_session_factory) == 0


def test_a_policy_with_a_declared_candidate_is_served_on_it(managed: tuple[TestClient, ModuleType]) -> None:
    client, probe = managed
    calls: list[str] = []

    async def provider(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs["model"])
        return ChatCompletion(
            id="chatcmpl-1",
            object="chat.completion",
            created=0,
            model="claude-haiku-4-5",
            choices=[
                Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")
            ],
            usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    response = _chat(client, _key(client), "managed_then_declared", provider)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert calls == ["anthropic:claude-haiku-4-5"]
    assert probe.RESOLVES == []


def test_a_plain_build_answers_provider_not_configured(
    plain: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(plain)

    embeddings, provider = _embeddings(plain, headers)
    assert embeddings.status_code == status.HTTP_424_FAILED_DEPENDENCY, embeddings.text
    assert embeddings.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    assert embeddings.json()["detail"].startswith("No credential is configured for provider 'openai'.")
    provider.assert_not_called()

    batch, provider = _batch(plain, headers)
    assert batch.status_code == status.HTTP_424_FAILED_DEPENDENCY, batch.text
    assert batch.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    provider.assert_not_called()

    policy = _chat(plain, headers, "managed_only", AsyncMock(side_effect=AssertionError))
    assert policy.status_code == status.HTTP_424_FAILED_DEPENDENCY, policy.text
    assert policy.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    assert "openai" in policy.text
    assert _active_reservations(db_session_factory) == 0


def test_an_aliased_managed_model_is_named_by_its_alias(
    managed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, _probe = managed
    response, provider = _embeddings(client, _key(client), "house-embed")

    _assert_managed(response, "house-embed", "/v1/embeddings")
    assert "openai" not in response.text
    assert "text-embedding" not in response.text
    provider.assert_not_called()
    assert _active_reservations(db_session_factory) == 0


def test_an_unpriced_managed_model_is_refused_as_managed_not_unpriced(
    managed_priced: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = managed_priced
    response, provider = _embeddings(client, _key(client))

    _assert_managed(response, "openai:text-embedding-3-small", "/v1/embeddings")
    provider.assert_not_called()
    assert probe.RESOLVES == []
    assert _active_reservations(db_session_factory) == 0
    rows = _rows(db_session_factory, "/v1/embeddings")
    assert [(row.status_code, row.error_message) for row in rows] == [(424, response.json()["detail"])]


def test_a_cancelled_listing_releases_the_pass_through_reservation(
    managed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = managed
    headers = _key(client)
    probe.FAULTS.add("cancel")
    # The test client runs the app on a portal thread, so the cancellation arrives as the futures one.
    with pytest.raises(concurrent.futures.CancelledError):
        _embeddings(client, headers)
    assert _active_reservations(db_session_factory) == 0
