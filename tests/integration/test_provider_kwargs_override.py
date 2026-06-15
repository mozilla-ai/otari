"""Tests for provider kwargs not overriding user request fields."""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from any_llm import LLMProvider
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.api.routes.chat import get_provider_kwargs
from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.db import Base, get_db
from gateway.main import create_app

from .conftest import _run_alembic_migrations, build_async_session_override


class _MockCompletionError(Exception):
    """Exception raised to short-circuit mock acompletion calls."""


@pytest.fixture
def config_with_model_in_provider(postgres_url: str) -> GatewayConfig:
    """Create a test configuration where the provider config contains a 'model' key."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        providers={
            "openai": {
                "api_key": "test-openai-key",
                "model": "provider-default-model",
            }
        },
    )


@pytest.fixture
def client_with_model_in_provider(config_with_model_in_provider: GatewayConfig) -> Generator[TestClient]:
    """Create a test client whose provider config contains a conflicting 'model' key."""
    _run_alembic_migrations(config_with_model_in_provider.database_url)
    engine = create_engine(config_with_model_in_provider.database_url, pool_pre_ping=True)
    app = create_app(config_with_model_in_provider)
    override_get_db, dispose_override = build_async_session_override(config_with_model_in_provider.database_url)
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


def test_provider_kwargs_do_not_contain_model() -> None:
    """Test that provider kwargs from a typical config don't include request-level fields."""
    config = GatewayConfig(
        database_url="postgresql://localhost/test",
        master_key="test",
        require_pricing=False,
        providers={
            "openai": {
                "api_key": "sk-test-key",
            },
        },
    )

    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI)
    assert "api_key" in kwargs
    assert "model" not in kwargs
    assert "messages" not in kwargs
    assert "stream" not in kwargs


@pytest.mark.asyncio
async def test_user_model_not_overridden_by_provider_config(
    client_with_model_in_provider: TestClient,
) -> None:
    """Verify the user's model value is not overwritten when provider config also contains 'model'."""
    captured_kwargs: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> None:
        captured_kwargs.update(kwargs)
        raise _MockCompletionError

    master_key_header = {API_KEY_HEADER: "Bearer test-master-key"}

    response = client_with_model_in_provider.post(
        "/v1/users",
        json={"user_id": "test-user", "alias": "Test User"},
        headers=master_key_header,
    )
    assert response.status_code == 200

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        client_with_model_in_provider.post(
            "/v1/chat/completions",
            json={
                "model": "openai:gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "user": "test-user",
            },
            headers=master_key_header,
        )

    assert captured_kwargs["model"] == "openai:gpt-4"
    assert captured_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert captured_kwargs["api_key"] == "test-openai-key"


@pytest.mark.asyncio
async def test_unset_optional_fields_do_not_override_provider_defaults(
    client_with_model_in_provider: TestClient,
) -> None:
    """Verify that optional request fields the user didn't set don't appear in kwargs."""
    captured_kwargs: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> None:
        captured_kwargs.update(kwargs)
        raise _MockCompletionError

    master_key_header = {API_KEY_HEADER: "Bearer test-master-key"}

    create_user = client_with_model_in_provider.post(
        "/v1/users",
        json={"user_id": "test-user", "alias": "Test User"},
        headers=master_key_header,
    )
    assert create_user.status_code == 200

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        client_with_model_in_provider.post(
            "/v1/chat/completions",
            json={
                "model": "openai:gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "user": "test-user",
            },
            headers=master_key_header,
        )

    # The call must be reached for the absence checks below to mean anything.
    assert captured_kwargs, "acompletion was never called"
    # The user didn't set temperature, so it should not be in the kwargs
    # (exclude_unset=True prevents None from overwriting provider defaults)
    assert "temperature" not in captured_kwargs
    assert "max_tokens" not in captured_kwargs
    assert "tools" not in captured_kwargs


@pytest.mark.asyncio
async def test_completion_params_reach_provider(
    client_with_model_in_provider: TestClient,
) -> None:
    """Every set completion param must reach ``acompletion``, not be dropped before the call.

    Regression guard for the silent-param-drop bug: ``reasoning_effort``
    (mozilla-ai/otari#150) and the standard OpenAI params (#152) were dropped
    because the hand-maintained schema omitted them. The schema is now derived
    from any-llm's ``CompletionParams`` and forwarded via ``model_dump``, so this
    asserts the forwarding contract end to end rather than only at the schema.
    """
    captured_kwargs: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> None:
        captured_kwargs.update(kwargs)
        raise _MockCompletionError

    params: dict[str, Any] = {
        "reasoning_effort": "high",
        "seed": 42,
        "stop": ["STOP"],
        "presence_penalty": 0.5,
        "frequency_penalty": 0.25,
        "n": 2,
        "parallel_tool_calls": False,
        "logprobs": True,
        "top_logprobs": 3,
        "logit_bias": {"123": -1.0},
    }

    master_key_header = {API_KEY_HEADER: "Bearer test-master-key"}

    create_user = client_with_model_in_provider.post(
        "/v1/users",
        json={"user_id": "test-user", "alias": "Test User"},
        headers=master_key_header,
    )
    assert create_user.status_code == 200

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        response = client_with_model_in_provider.post(
            "/v1/chat/completions",
            json={
                "model": "openai:gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "user": "test-user",
                **params,
            },
            headers=master_key_header,
        )

    # The mock raises after capturing, so the call is reached but the request errors out.
    assert captured_kwargs, f"acompletion was never called (status {response.status_code})"
    for name, value in params.items():
        assert captured_kwargs.get(name) == value, f"{name} was dropped before the provider call"
