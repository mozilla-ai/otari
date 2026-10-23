"""A ``providers:`` entry of config.yml is copied onto a hosted provider of its name at startup, once.

The entry's ``${VAR}`` values are read at that start, and the hosted provider
serves the name from then on. An entry a hosted provider cannot hold keeps
serving from config.yml. Model discovery is stubbed; the provider call is
captured at the any-llm boundary.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.providers import ConfiguredProviderImport, HostedProvider
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import decrypt_secret, generate_secret_key

from .conftest import build_test_client, shared_test_config

MASTER = {"Otari-Key": "Bearer test-master-key"}
EU_KEY = "sk-eu-from-the-environment"
EU_BASE = "https://eu.test/v1"
Start = Callable[[], AbstractContextManager[TestClient]]


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """One entry a hosted provider can hold, and one it cannot (it declares its own model list)."""
    config = shared_test_config(
        postgres_url,
        {
            "openai-eu": {"provider_type": "openai", "api_key": EU_KEY, "api_base": EU_BASE},
            "local": {"provider_type": "openai", "api_base": "http://local.test/v1", "models": ["llama-3"]},
        },
    )
    return config.model_copy(update={"import_configured_providers": True})


LISTED: list[str] = ["gpt-4o"]


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    async def _lists(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name,
            models=[Model(id=model, object="model", created=0, owned_by=impl_name) for model in LISTED],
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _lists)
    yield
    LISTED[:] = ["gpt-4o"]


@pytest.fixture
def start(test_config: GatewayConfig) -> Start:
    @contextmanager
    def _start() -> Iterator[TestClient]:
        for client in build_test_client(test_config):
            yield client

    return _start


def _chat(client: TestClient, model: str) -> tuple[int, dict[str, Any]]:
    completion = ChatCompletion(
        id="c",
        object="chat.completion",
        created=0,
        model="gpt-4o",
        choices=[Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="Hi"))],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    created = client.post(f"{API_ROOT}/users", json={"user_id": "u1"}, headers=MASTER)
    assert created.status_code in (200, 409), created.text
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=completion)) as call:
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=MASTER,
        )
    return response.status_code, dict(call.await_args.kwargs) if call.await_args else {}


def test_an_entry_is_copied_onto_a_hosted_provider_and_served_by_it(
    start: Start, test_config: GatewayConfig, db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    with start() as client:
        status, sent = _chat(client, "openai-eu:gpt-4o")
        assert "openai-eu" not in test_config.providers
        # An entry a hosted provider cannot hold keeps serving from config.yml.
        assert "local" in test_config.providers

    assert status == 200
    assert (sent["api_key"], sent["api_base"]) == (EU_KEY, EU_BASE)
    with db_session_factory() as db:
        row = db.execute(select(HostedProvider).where(HostedProvider.name == "openai-eu")).scalar_one()
        assert (row.provider, row.api_base, row.serves_unlisted_models) == ("openai", EU_BASE, False)
        assert row.encrypted_api_key is not None and decrypt_secret(row.encrypted_api_key) == EU_KEY
        assert db.execute(select(ConfiguredProviderImport.name)).scalars().all() == ["openai-eu"]


def test_a_copied_entry_keeps_its_models_in_the_catalog(start: Start, monkeypatch: pytest.MonkeyPatch) -> None:
    """The copy is no longer discovered as a configured instance, and nothing prices its models."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    with start() as client:
        listed = client.get(f"{API_ROOT}/models", headers=MASTER)
        catalog = client.get(f"{API_ROOT}/catalog/models", headers=MASTER)

    assert listed.status_code == 200, listed.text
    assert "openai-eu:gpt-4o" in {model["id"] for model in listed.json()["data"]}
    assert catalog.status_code == 200, catalog.text
    assert "openai-eu:gpt-4o" in {selector for model in catalog.json()["models"] for selector in model["selectors"]}


def test_a_copied_entry_removed_on_the_page_stays_removed(
    start: Start, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    with start() as client:
        removed = client.delete(f"{API_ROOT}/hosted-providers/openai-eu", headers=MASTER)
        assert removed.status_code == 200, removed.text

    with start() as client:
        listed = client.get(f"{API_ROOT}/hosted-providers", headers=MASTER).json()["data"]
        assert "openai-eu" not in {row["name"] for row in listed}
        assert "openai-eu" not in test_config.providers


def test_without_a_secret_key_the_entry_keeps_serving_from_config(
    start: Start, test_config: GatewayConfig, db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    with start() as client:
        status, sent = _chat(client, "openai-eu:gpt-4o")
        assert "openai-eu" in test_config.providers

    assert status == 200
    assert sent["api_key"] == EU_KEY
    with db_session_factory() as db:
        assert db.execute(select(HostedProvider)).scalars().all() == []


def test_an_entry_whose_provider_lists_nothing_keeps_serving_from_config(
    start: Start, test_config: GatewayConfig, db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy would serve none of its models, so none is kept, and the next start asks again."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    LISTED.clear()

    with start() as client:
        status, sent = _chat(client, "openai-eu:gpt-4o")
        assert "openai-eu" in test_config.providers

    assert (status, sent["api_key"]) == (200, EU_KEY)
    with db_session_factory() as db:
        assert db.execute(select(HostedProvider)).scalars().all() == []
        assert db.execute(select(ConfiguredProviderImport)).scalars().all() == []
