"""Integration tests for model auto-discovery in the GET /v1/models endpoint."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.db import Base, get_db
from gateway.main import create_app
from gateway.services.model_discovery_service import get_model_cache

from .conftest import _run_alembic_migrations, _to_async_url


def _make_openai_model(model_id: str, owned_by: str = "openai", created: int = 1700000000) -> dict[str, Any]:
    """Build a dict that Model.model_validate accepts."""
    return {"id": model_id, "object": "model", "created": created, "owned_by": owned_by}


@pytest.fixture
def discovery_config(postgres_url: str) -> GatewayConfig:
    """Config with a fake openai provider and discovery enabled."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        model_discovery=True,
        model_cache_ttl_seconds=300,
        providers={"openai": {"api_key": "sk-fake-for-test"}},
    )


@pytest.fixture
def no_discovery_config(postgres_url: str) -> GatewayConfig:
    """Config with discovery disabled."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-fake-for-test"}},
    )


def _make_client(config: GatewayConfig) -> Generator[TestClient]:
    _run_alembic_migrations(config.database_url)

    from sqlalchemy import create_engine, text

    engine = create_engine(config.database_url, pool_pre_ping=True)
    async_engine = create_async_engine(_to_async_url(config.database_url), pool_pre_ping=True)
    async_session_factory = async_sessionmaker(async_engine, expire_on_commit=False)
    app = create_app(config)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with async_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        # Dispose the async engine to avoid leaking connections/tasks.
        asyncio.run(async_engine.dispose())
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()
        engine.dispose()


@pytest.fixture
def discovery_client(discovery_config: GatewayConfig) -> Generator[TestClient]:
    """Test client with model discovery enabled."""
    # Clear the module-level model cache before each test.
    get_model_cache().clear()
    yield from _make_client(discovery_config)


@pytest.fixture
def no_discovery_client(no_discovery_config: GatewayConfig) -> Generator[TestClient]:
    """Test client with model discovery disabled."""
    get_model_cache().clear()
    yield from _make_client(no_discovery_config)


@pytest.fixture
def discovery_master_header() -> dict[str, str]:
    return {API_KEY_HEADER: "Bearer test-master-key"}


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


def test_list_models_with_discovery(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Discovered models appear in GET /v1/models."""
    from any_llm.types.model import Model

    fake_models = [
        Model(**_make_openai_model("gpt-4o")),
        Model(**_make_openai_model("gpt-4o-mini")),
    ]

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "openai:gpt-4o" in ids
    assert "openai:gpt-4o-mini" in ids


def test_list_models_discovery_disabled(
    no_discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """When discovery is disabled, only pricing-table models appear."""
    resp = no_discovery_client.get("/v1/models", headers=discovery_master_header)
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_list_models_pricing_enrichment(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Discovered models are enriched with pricing data when available."""
    from any_llm.types.model import Model

    fake_models = [Model(**_make_openai_model("gpt-4o"))]

    # First, set pricing for this model.
    resp = discovery_client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=discovery_master_header,
    )
    assert resp.status_code == 200

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    models = resp.json()["data"]
    gpt4 = next(m for m in models if m["id"] == "openai:gpt-4o")
    assert gpt4["pricing"] is not None
    assert gpt4["pricing"]["input_price_per_million"] == 2.5
    assert gpt4["pricing"]["output_price_per_million"] == 10.0


def test_list_models_no_pricing_returns_null(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Discovered models without pricing have pricing=None."""
    from any_llm.types.model import Model

    fake_models = [Model(**_make_openai_model("gpt-4o-mini"))]

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    models = resp.json()["data"]
    gpt4_mini = next(m for m in models if m["id"] == "openai:gpt-4o-mini")
    assert gpt4_mini["pricing"] is None


def test_list_models_pricing_only_still_appears(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Models only in the pricing table still appear even when discovery returns nothing."""
    # Add a pricing-only model (for a provider not in config.providers).
    resp = discovery_client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:legacy-model",
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
        },
        headers=discovery_master_header,
    )
    assert resp.status_code == 200

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "openai:legacy-model" in ids


def test_list_models_provider_filter(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """The ?provider= query param filters results to a single provider."""
    from any_llm.types.model import Model

    fake_models = [Model(**_make_openai_model("gpt-4o"))]

    # Add a pricing entry for a different provider.
    discovery_client.post(
        "/v1/pricing",
        json={
            "model_key": "anthropic:claude-3-opus",
            "input_price_per_million": 15.0,
            "output_price_per_million": 75.0,
        },
        headers=discovery_master_header,
    )

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models?provider=openai", headers=discovery_master_header)

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    # OpenAI models should appear; Anthropic should not.
    assert "openai:gpt-4o" in ids
    assert all("anthropic:" not in mid for mid in ids)


def test_list_models_discovery_error_graceful(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """When discovery fails for a provider, the endpoint still succeeds."""
    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            side_effect=ConnectionError("upstream unreachable"),
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    assert resp.json()["object"] == "list"


def test_get_model_from_discovery_cache(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """GET /v1/models/{id} finds a model that exists only in the discovery cache."""
    from any_llm.types.model import Model

    fake_models = [Model(**_make_openai_model("gpt-4o"))]

    # Pre-populate the discovery cache.
    cache = get_model_cache()
    cache.set("openai", fake_models)

    resp = discovery_client.get("/v1/models/openai:gpt-4o", headers=discovery_master_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "openai:gpt-4o"
    assert data["owned_by"] == "openai"


def test_get_model_not_found_with_discovery(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """GET /v1/models/{id} returns 404 when not in cache or pricing table."""
    resp = discovery_client.get(
        "/v1/models/openai:nonexistent-model",
        headers=discovery_master_header,
    )
    assert resp.status_code == 404


def test_list_models_sorted(
    discovery_client: TestClient,
    discovery_master_header: dict[str, str],
) -> None:
    """Discovered models are sorted alphabetically by id."""
    from any_llm.types.model import Model

    fake_models = [
        Model(**_make_openai_model("gpt-4o")),
        Model(**_make_openai_model("dall-e-3")),
        Model(**_make_openai_model("gpt-3.5-turbo")),
    ]

    with (
        patch(
            "gateway.services.model_discovery_service._supports_list_models",
            return_value=True,
        ),
        patch(
            "gateway.services.model_discovery_service.alist_models",
            new_callable=AsyncMock,
            return_value=fake_models,
        ),
    ):
        resp = discovery_client.get("/v1/models", headers=discovery_master_header)

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert ids == sorted(ids)
