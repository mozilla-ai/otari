"""Tests for the GET /v1/providers endpoint."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.core.config import GatewayConfig
from gateway.core.database import get_db
from gateway.main import create_app

from .conftest import _run_alembic_migrations, _to_async_url


@pytest.fixture
def providers_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        providers={
            "openai": {"api_key": "sk-fake-for-test"},
            "anthropic": {"api_key": "sk-ant-fake-for-test"},
        },
    )


@pytest.fixture
def providers_client(providers_config: GatewayConfig) -> Generator[TestClient]:
    _run_alembic_migrations(providers_config.database_url)
    async_engine = create_async_engine(_to_async_url(providers_config.database_url), pool_pre_ping=True)
    async_session_factory = async_sessionmaker(async_engine, expire_on_commit=False)
    app = create_app(providers_config)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with async_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        try:
            asyncio.run(async_engine.dispose())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(async_engine.dispose())
            loop.close()


@pytest.fixture
def master_header() -> dict[str, str]:
    return {"Authorization": "Bearer test-master-key"}


def test_providers_lists_configured_providers(
    providers_client: TestClient,
    master_header: dict[str, str],
) -> None:
    """Every configured provider is returned, sorted by instance name."""
    resp = providers_client.get("/v1/providers", headers=master_header)
    assert resp.status_code == 200

    providers = resp.json()["providers"]
    instances = [p["instance"] for p in providers]
    assert instances == ["anthropic", "openai"]


def test_providers_carries_metadata_and_capabilities(
    providers_client: TestClient,
    master_header: dict[str, str],
) -> None:
    """OpenAI reports a display name, doc link, pricing links, and capabilities."""
    resp = providers_client.get("/v1/providers", headers=master_header)
    providers = {p["instance"]: p for p in resp.json()["providers"]}

    openai = providers["openai"]
    assert openai["provider_type"] == "openai"
    assert openai["name"] == "OpenAI"
    assert openai["doc_url"].startswith("http")
    assert openai["pricing_urls"]
    assert openai["env_key"] == "OPENAI_API_KEY"

    caps = openai["capabilities"]
    assert caps["vision"] is True
    assert caps["list_models"] is True
    # Every curated flag is present in the response shape.
    assert set(caps) == {
        "streaming",
        "reasoning",
        "vision",
        "pdf",
        "embeddings",
        "image_generation",
        "audio",
        "rerank",
        "responses_api",
        "moderation",
        "list_models",
    }


def test_providers_requires_master_key(
    providers_client: TestClient,
) -> None:
    """The endpoint describes gateway config, so it is master-key gated."""
    resp = providers_client.get("/v1/providers")
    assert resp.status_code in (401, 403)


def test_models_carry_context_window(
    providers_client: TestClient,
    master_header: dict[str, str],
) -> None:
    """A priced model the dataset knows reports its context window."""
    post = providers_client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_header,
    )
    assert post.status_code == 200

    resp = providers_client.get("/v1/models/openai:gpt-4o", headers=master_header)
    assert resp.status_code == 200
    body: dict[str, Any] = resp.json()
    assert body["context_window"] == 128000
