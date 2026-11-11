"""Memory's models run on the deployment's hosted providers when ``config.yml`` declares none that serves them.

The provider call is captured at the any-llm boundary and never made.
"""

import uuid
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import create_unit_of_work
from gateway.models.tenancy import Workspace
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.services.memory import MemoryModels
from gateway.services.memory._models import MemoryModelError
from gateway.services.secret_box import generate_secret_key

from .conftest import shared_test_config

KEY = "sk-live-hosted-memory-1234"
EMBEDDING_MODEL = "openai:text-embedding-3-small"


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """No provider declared, so only a hosted one can serve memory's models."""
    return shared_test_config(postgres_url, {}).model_copy(
        update={"memory_enabled": True, "memory_embedding_model": EMBEDDING_MODEL}
    )


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A developer's own shell key would credential the ladder and skip the hosted provider."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


def _hosted_openai(client: TestClient, headers: dict[str, str], **extra: Any) -> None:
    body = {"provider": "openai", "api_key": KEY, "api_base": "https://hosted.test/v1", "serves_unlisted_models": True}
    response = client.post(f"{API_ROOT}/hosted-providers", json={**body, **extra}, headers=headers)
    assert response.status_code == 201, response.text


def _embed(client: TestClient) -> dict[str, Any]:
    """Embed one text as the consolidator would, on a Unit of Work of its own; what the provider was sent."""
    app = cast(FastAPI, client.app)
    sent: dict[str, Any] = {}

    async def fake(model: str, inputs: list[str], **kwargs: Any) -> SimpleNamespace:
        sent.update(kwargs, model=model)
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])], usage=None)

    async def run() -> None:
        async with create_unit_of_work() as uow:
            port = app.state.container.resolve(ModelProviderPort, None, uow=uow)
            await MemoryModels(app.state.config, port).embed(["deploy"])

    portal = client.portal
    assert portal is not None
    with patch("gateway.services.memory._models.aembedding", new=fake):
        portal.call(run)
    return sent


def test_a_hosted_provider_serves_memorys_models_on_its_own_credential(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _hosted_openai(client, master_key_header)

    sent = _embed(client)

    assert sent["api_key"] == KEY
    assert sent["api_base"] == "https://hosted.test/v1"
    assert sent["model"] == "text-embedding-3-small"


def test_a_hosted_provider_restricted_to_an_organization_does_not_serve_memorys_models(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    with db_session_factory() as session:
        workspace = session.scalars(select(Workspace)).first()
        assert workspace is not None
        organization_id: uuid.UUID = workspace.organization_id
    _hosted_openai(client, master_key_header, restricted_to_organization_id=str(organization_id))

    with pytest.raises(MemoryModelError):
        _embed(client)


def test_with_no_provider_to_serve_them_memorys_models_fail_without_a_call(client: TestClient) -> None:
    with pytest.raises(MemoryModelError):
        assert _embed(client) == {}
