"""End-to-end behavior for model aliases (mozilla-ai/otari#228).

An alias is a display name (e.g. ``myopusmodel``) that maps to a real selector
(``provider:model`` or a named ``instance:model``). A request naming the alias
routes to the target with the target's credentials, billing keys on the target,
and the alias is what appears in GET /v1/models and in the response ``model``.
"""

from collections.abc import AsyncIterator, Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Usage,
)
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.db import Base, get_db
from gateway.main import create_app
from gateway.types.moderation import ModerationResponse, ModerationResult

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
# require_pricing rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/embeddings", {"input": "hello"}),
        ("/v1/rerank", {"query": "q", "documents": ["a", "b"]}),
        ("/v1/images/generations", {"prompt": "a cat"}),
    ],
)
def test_unpriced_alias_402_echoes_alias_not_target(
    client: TestClient,
    alias_config: GatewayConfig,
    path: str,
    payload: dict[str, Any],
) -> None:
    # The app holds this config object, so the gate reads the flipped flag. With
    # no pricing on the target, the 402 must name the alias: echoing
    # "{instance}:{model}" would hand back the string aliases exist to hide.
    alias_config.require_pricing = True
    _create_user(client)

    resp = client.post(path, json={"model": "myopusmodel", "user": "test-user", **payload}, headers=HEADERS)

    assert resp.status_code == 402
    detail = resp.json()["detail"]
    assert "myopusmodel" in detail
    assert "claude-opus-4" not in detail
    assert "anthropic" not in detail


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


def test_legacy_form_pricing_row_still_prices_alias(client: TestClient, alias_config: GatewayConfig) -> None:
    # Keys are canonicalized on write, but a row predating that may still use the
    # legacy "provider/model" separator. It must price the alias and must not put
    # the target back in the listing: withholding it while failing to match the
    # alias would show the price nowhere.
    engine = create_engine(alias_config.database_url)
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO model_pricing "
                    "(model_key, effective_at, input_price_per_million, output_price_per_million, "
                    " created_at, updated_at) "
                    "VALUES ('anthropic/claude-opus-4', NOW(), 15.0, 75.0, NOW(), NOW())"
                ),
            )
            conn.commit()
    finally:
        engine.dispose()

    resp = client.get("/v1/models", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert {m["id"] for m in data} == {"myopusmodel", "housemodel"}
    entry = next(m for m in data if m["id"] == "myopusmodel")
    assert entry["pricing"]["input_price_per_million"] == 15.0

    # The single-model endpoint reads the same row through its filtered query.
    single = client.get("/v1/models/myopusmodel", headers=HEADERS)
    assert single.json()["pricing"]["input_price_per_million"] == 15.0


def test_get_model_alias_surfaces_target_pricing(client: TestClient) -> None:
    client.post(
        "/v1/pricing",
        json={
            "model_key": "home_lab:qwen3",
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
        },
        headers=HEADERS,
    )
    resp = client.get("/v1/models/housemodel", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["pricing"] == {"input_price_per_million": 1.0, "output_price_per_million": 2.0}


def test_alias_listing_reports_where_its_price_came_from(client: TestClient) -> None:
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
    data = resp.json()["data"]
    # The target is priced in the database, so the alias says so. The unpriced
    # alias reports "none" rather than inheriting the other one's source.
    assert next(m for m in data if m["id"] == "myopusmodel")["pricing_source"] == "configured"
    assert next(m for m in data if m["id"] == "housemodel")["pricing_source"] == "none"


def test_pricing_an_alias_is_rejected(client: TestClient) -> None:
    # Pricing keys on the resolved target, so a row stored under the alias name
    # is never read. Accepting the write would report success and silently
    # change nothing about what the caller is billed.
    resp = client.post(
        "/v1/pricing",
        json={
            "model_key": "myopusmodel",
            "input_price_per_million": 99.0,
            "output_price_per_million": 99.0,
        },
        headers=HEADERS,
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "alias" in detail
    # The error names the target, so the caller knows what to price instead.
    assert "anthropic:claude-opus-4" in detail

    # Nothing was written: the alias is still unpriced, not priced at 99.
    listing = client.get("/v1/models", headers=HEADERS)
    assert next(m for m in listing.json()["data"] if m["id"] == "myopusmodel")["pricing"] is None
    assert client.get("/v1/pricing/myopusmodel", headers=HEADERS).status_code == 404


def test_pricing_the_alias_target_is_still_accepted(client: TestClient) -> None:
    # The rejection is scoped to alias names only; the target it points at is the
    # supported way to price an aliased model.
    resp = client.post(
        "/v1/pricing",
        json={
            "model_key": "anthropic:claude-opus-4",
            "input_price_per_million": 15.0,
            "output_price_per_million": 75.0,
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200


def test_pricing_on_alias_target_does_not_expose_it(client: TestClient) -> None:
    # Aliasing a model forces a pricing entry on the real target (billing keys
    # there), and that entry must not put the hidden name back in the listing.
    client.post(
        "/v1/pricing",
        json={
            "model_key": "home_lab:qwen3",
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
        },
        headers=HEADERS,
    )
    resp = client.get("/v1/models", headers=HEADERS)
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert ids == {"myopusmodel", "housemodel"}


def test_pricing_on_unaliased_model_still_lists_it(client: TestClient) -> None:
    # Only alias targets are withheld; a priced model nothing points at is still
    # listed, preserving the pricing-only listing behavior.
    client.post(
        "/v1/pricing",
        json={
            "model_key": "anthropic:claude-haiku-4",
            "input_price_per_million": 1.0,
            "output_price_per_million": 5.0,
        },
        headers=HEADERS,
    )
    resp = client.get("/v1/models", headers=HEADERS)
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert ids == {"myopusmodel", "housemodel", "anthropic:claude-haiku-4"}


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


# ---------------------------------------------------------------------------
# Non-chat surfaces whose responses carry a ``model`` field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embeddings_response_model_echoes_alias(client: TestClient) -> None:
    _create_user(client)

    mock_response = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2], index=0, object="embedding")],
        model="qwen3",  # what the provider returns
        object="list",
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )

    with patch("gateway.api.routes.embeddings.aembedding", new_callable=AsyncMock, return_value=mock_response):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "housemodel", "input": "hello", "user": "test-user"},
            headers=HEADERS,
        )
    assert resp.status_code == 200
    # The caller sees the alias they sent, not the underlying model name.
    assert resp.json()["model"] == "housemodel"


@pytest.mark.asyncio
async def test_moderations_response_model_echoes_alias(client: TestClient) -> None:
    _create_user(client)

    mock_response = ModerationResponse(
        id="modr-alias",
        model="claude-opus-4",  # what the provider returns
        results=[
            ModerationResult(
                flagged=False,
                categories={"violence": False},
                category_scores={"violence": 0.01},
            )
        ],
    )

    with patch("gateway.api.routes.moderations.amoderation", new_callable=AsyncMock, return_value=mock_response):
        resp = client.post(
            "/v1/moderations",
            json={"model": "myopusmodel", "input": "hello", "user": "test-user"},
            headers=HEADERS,
        )
    assert resp.status_code == 200
    # The caller sees the alias they sent, not the underlying model name.
    assert resp.json()["model"] == "myopusmodel"
