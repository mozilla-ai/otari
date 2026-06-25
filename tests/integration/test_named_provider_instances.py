"""End-to-end routing for named provider instances (mozilla-ai/otari#213).

Two instances that share an implementation (real OpenAI plus a self-hosted
OpenAI-compatible backend) must route to any-llm with the right implementation
and the right per-instance credentials, while pricing/usage key on the instance
name.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.db import Base, get_db
from gateway.main import create_app

from .conftest import _run_alembic_migrations, build_async_session_override


class _MockCompletionError(Exception):
    """Raised to short-circuit the mocked acompletion after capturing kwargs."""


@pytest.fixture
def multi_instance_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        providers={
            "openai": {
                "api_key": "sk-real-openai",
                "api_base": "https://api.openai.com/v1",
            },
            "home_lab": {
                "provider_type": "openai",
                "api_base": "https://box.ts.net/v1",
                "api_key": "home-lab-token",
            },
        },
    )


@pytest.fixture
def client(multi_instance_config: GatewayConfig) -> Generator[TestClient]:
    _run_alembic_migrations(multi_instance_config.database_url)
    engine = create_engine(multi_instance_config.database_url, pool_pre_ping=True)
    app = create_app(multi_instance_config)
    override_get_db, dispose_override = build_async_session_override(multi_instance_config.database_url)
    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        dispose_override()
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()


def _create_user(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post(
        "/v1/users",
        json={"user_id": "test-user", "alias": "Test User"},
        headers=headers,
    )
    assert response.status_code == 200


def _post_chat(client: TestClient, headers: dict[str, str], model: str) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise _MockCompletionError

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "user": "test-user",
            },
            headers=headers,
        )
    assert captured, f"acompletion was never called for model {model!r}"
    return captured


@pytest.mark.asyncio
async def test_named_instance_routes_to_implementation_with_its_credentials(client: TestClient) -> None:
    headers = {API_KEY_HEADER: "Bearer test-master-key"}
    _create_user(client, headers)

    captured = _post_chat(client, headers, "home_lab:deepseek-v4-flash")

    # any-llm sees the implementation, never the instance name.
    assert captured["model"] == "openai:deepseek-v4-flash"
    # ...routed with the home_lab instance's own credentials.
    assert captured["api_base"] == "https://box.ts.net/v1"
    assert captured["api_key"] == "home-lab-token"


@pytest.mark.asyncio
async def test_real_provider_instance_still_routes_normally(client: TestClient) -> None:
    headers = {API_KEY_HEADER: "Bearer test-master-key"}
    _create_user(client, headers)

    captured = _post_chat(client, headers, "openai:gpt-4o")

    assert captured["model"] == "openai:gpt-4o"
    assert captured["api_base"] == "https://api.openai.com/v1"
    assert captured["api_key"] == "sk-real-openai"


@pytest.mark.asyncio
async def test_two_instances_of_same_impl_do_not_collide(client: TestClient) -> None:
    headers = {API_KEY_HEADER: "Bearer test-master-key"}
    _create_user(client, headers)

    real = _post_chat(client, headers, "openai:gpt-4o")
    local = _post_chat(client, headers, "home_lab:gpt-4o")

    assert real["api_key"] == "sk-real-openai"
    assert local["api_key"] == "home-lab-token"
    # Same model name on both, but each carries its own instance credentials.
    assert real["model"] == local["model"] == "openai:gpt-4o"


@pytest.mark.asyncio
async def test_pricing_round_trip_for_instance_key(client: TestClient) -> None:
    """Pricing for an instance-scoped key can be set and then read back (no 500).

    Regression: the pricing read path split the key via any-llm, which raises
    AnyLLMError for an instance name, surfacing as a 500.
    """
    headers = {API_KEY_HEADER: "Bearer test-master-key"}

    set_resp = client.post(
        "/v1/pricing",
        json={
            "model_key": "home_lab:deepseek-v4-flash",
            "input_price_per_million": 0.0,
            "output_price_per_million": 0.0,
        },
        headers=headers,
    )
    assert set_resp.status_code == 200
    assert set_resp.json()["model_key"] == "home_lab:deepseek-v4-flash"

    get_resp = client.get("/v1/pricing/home_lab:deepseek-v4-flash", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["model_key"] == "home_lab:deepseek-v4-flash"

    # The legacy slash form resolves to the same stored colon key.
    slash_resp = client.get("/v1/pricing/home_lab/deepseek-v4-flash/history", headers=headers)
    assert slash_resp.status_code == 200
    assert slash_resp.json()[0]["model_key"] == "home_lab:deepseek-v4-flash"
