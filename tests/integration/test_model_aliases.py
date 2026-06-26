"""End-to-end behavior for model aliases (mozilla-ai/otari#228).

An alias is a display name (e.g. ``myopusmodel``) that maps to a real selector
(``provider:model`` or a named ``instance:model``). A request naming the alias
routes to the target with the target's credentials, billing keys on the target,
and the alias is what appears in GET /v1/models and in the response ``model``.
"""

from collections.abc import AsyncIterator, Generator
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.db import Base, get_db
from gateway.main import create_app

from .conftest import _run_alembic_migrations, build_async_session_override


class _MockCompletionError(Exception):
    """Raised to short-circuit the mocked acompletion after capturing kwargs."""


@pytest.fixture
def alias_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        # Discovery off so the listing exposes only the curated alias names plus
        # any explicitly-priced models, not the full provider catalog.
        model_discovery=False,
        providers={
            "anthropic": {"api_key": "sk-ant"},
            "home_lab": {
                "provider_type": "openai",
                "api_base": "https://box.ts.net/v1",
                "api_key": "home-lab-token",
            },
        },
        aliases={
            "myopusmodel": "anthropic:claude-opus-4",
            "housemodel": "home_lab:qwen3",
        },
    )


@pytest.fixture
def client(alias_config: GatewayConfig) -> Generator[TestClient]:
    _run_alembic_migrations(alias_config.database_url)
    engine = create_engine(alias_config.database_url, pool_pre_ping=True)
    app = create_app(alias_config)
    override_get_db, dispose_override = build_async_session_override(alias_config.database_url)
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


HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}


def _create_user(client: TestClient) -> None:
    resp = client.post("/v1/users", json={"user_id": "test-user", "alias": "Test User"}, headers=HEADERS)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def test_list_models_shows_aliases_only_with_discovery_off(client: TestClient) -> None:
    resp = client.get("/v1/models", headers=HEADERS)
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    # Only the curated alias names are exposed; the real provider/model is hidden.
    assert ids == {"myopusmodel", "housemodel"}


def test_alias_listing_owned_by_is_gateway(client: TestClient) -> None:
    resp = client.get("/v1/models", headers=HEADERS)
    entry = next(m for m in resp.json()["data"] if m["id"] == "myopusmodel")
    assert entry["owned_by"] == "otari"
    assert entry["object"] == "model"


def test_alias_listing_surfaces_target_pricing(client: TestClient) -> None:
    # Pricing is configured on the real target; the alias inherits it in listing.
    client.post(
        "/v1/pricing",
        json={
            "model_key": "anthropic:claude-opus-4",
            "input_price_per_million": 15.0,
            "output_price_per_million": 75.0,
        },
        headers=HEADERS,
    )
    resp = client.get("/v1/models", headers=HEADERS)
    entry = next(m for m in resp.json()["data"] if m["id"] == "myopusmodel")
    assert entry["pricing"]["input_price_per_million"] == 15.0
    assert entry["pricing"]["output_price_per_million"] == 75.0


def test_provider_filter_excludes_aliases(client: TestClient) -> None:
    # A ?provider= filter asks for one provider's real models and must not leak
    # the alias mapping.
    resp = client.get("/v1/models", params={"provider": "anthropic"}, headers=HEADERS)
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert "myopusmodel" not in ids


def test_get_model_by_alias(client: TestClient) -> None:
    resp = client.get("/v1/models/housemodel", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "housemodel"
    assert data["owned_by"] == "otari"


# ---------------------------------------------------------------------------
# Request routing
# ---------------------------------------------------------------------------


def _post_chat_capture(client: TestClient, model: str) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise _MockCompletionError

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "user": "test-user"},
            headers=HEADERS,
        )
    assert captured, f"acompletion was never called for {model!r}"
    return captured


@pytest.mark.asyncio
async def test_alias_routes_to_provider_target(client: TestClient) -> None:
    _create_user(client)
    captured = _post_chat_capture(client, "myopusmodel")
    # any-llm sees the real model, with the target provider's credentials.
    assert captured["model"] == "anthropic:claude-opus-4"
    assert captured["api_key"] == "sk-ant"


@pytest.mark.asyncio
async def test_alias_routes_to_named_instance_target(client: TestClient) -> None:
    _create_user(client)
    captured = _post_chat_capture(client, "housemodel")
    # Alias -> named instance -> implementation, with the instance's credentials.
    assert captured["model"] == "openai:qwen3"
    assert captured["api_base"] == "https://box.ts.net/v1"
    assert captured["api_key"] == "home-lab-token"


@pytest.mark.asyncio
async def test_response_model_echoes_alias(client: TestClient) -> None:
    _create_user(client)

    mock_response = ChatCompletion(
        id="chatcmpl-alias",
        object="chat.completion",
        created=0,
        model="claude-opus-4",  # what the provider returns
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        return mock_response

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "myopusmodel", "messages": [{"role": "user", "content": "Hi"}], "user": "test-user"},
            headers=HEADERS,
        )
    assert resp.status_code == 200
    # The caller sees the alias they sent, not the underlying model name.
    assert resp.json()["model"] == "myopusmodel"


@pytest.mark.asyncio
async def test_streaming_response_model_echoes_alias(client: TestClient) -> None:
    _create_user(client)

    async def chunk_stream() -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            id="chatcmpl-alias",
            object="chat.completion.chunk",
            created=0,
            model="claude-opus-4",  # what the provider streams back
            choices=[
                ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason="stop")
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def mock_acompletion(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        return chunk_stream()

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "myopusmodel",
                "messages": [{"role": "user", "content": "Hi"}],
                "user": "test-user",
                "stream": True,
            },
            headers=HEADERS,
        )
    assert resp.status_code == 200
    body = resp.text
    # The streamed chunks carry the alias, never the underlying model name.
    assert "myopusmodel" in body
    assert "claude-opus-4" not in body
