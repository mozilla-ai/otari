"""A stream whose provider reports usage on its finish chunk still carries its cost.

OpenAI ends a stream with a usage-only chunk after the one that finishes the
choice. Cohere (through any-llm) has no such chunk: usage rides on the chunk
whose choice carries ``finish_reason``, and the chunk's ``model`` is any-llm's
placeholder rather than the model that served. These pin that such a stream's
final usage carries ``cost_usd`` and ``pricing_source`` from the price of the
model the request resolved to, named directly or through a routing policy.
"""

import json
import time
from collections.abc import AsyncIterator, Callable, Generator
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog

from .conftest import build_test_client

_HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}
_MODEL = "cohere:command-r7b-12-2024"
_POLICY = "cohere-first"
_USER = "cohere-stream-user"
# $1.00 per million tokens each way, so 100k in plus 200k out settles at $0.30.
_USAGE = CompletionUsage(prompt_tokens=100_000, completion_tokens=200_000, total_tokens=300_000)


@pytest.fixture
def cohere_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"cohere": {"api_key": "co-test"}},
        routing=RoutingConfig.model_validate({"policies": {_POLICY: {"select": [{"default": _MODEL}]}}}),
    )


@pytest.fixture
def cohere_client(cohere_config: GatewayConfig) -> Generator[TestClient]:
    yield from build_test_client(cohere_config)


@pytest.fixture
def make_session(cohere_config: GatewayConfig) -> Generator[Callable[[], Session]]:
    engine = create_engine(cohere_config.database_url, pool_pre_ping=True)
    try:
        yield sessionmaker(bind=engine)
    finally:
        engine.dispose()


def _cohere_chunks() -> list[ChatCompletionChunk]:
    """The shape any-llm's Cohere stream has: no usage-only chunk, and a placeholder model."""
    return [
        ChatCompletionChunk(
            id="chatcmpl-cohere",
            object="chat.completion.chunk",
            created=0,
            model="cohere-model",
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason=None)],
        ),
        ChatCompletionChunk(
            id="chatcmpl-cohere",
            object="chat.completion.chunk",
            created=0,
            model="cohere-model",
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
            usage=_USAGE,
        ),
    ]


def _stream(client: TestClient, model: str) -> Any:
    async def _acompletion(**_kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def _chunks() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in _cohere_chunks():
                yield chunk

        return _chunks()

    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _acompletion
        return client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": _USER, "stream": True},
            headers=_HEADERS,
        )


def _logged_cost(make_session: Callable[[], Session], *, timeout: float = 3.0) -> Decimal | None:
    """The request's row cost, polled with a fresh session so a background log writer cannot race the read."""
    deadline = time.monotonic() + timeout
    while True:
        with make_session() as db:
            row = db.execute(select(UsageLog).where(UsageLog.user_id == _USER)).scalar_one_or_none()
            if row is not None:
                return row.cost
        assert time.monotonic() < deadline, "the usage row was never written"
        time.sleep(0.1)


@pytest.mark.parametrize("model", [_MODEL, _POLICY], ids=["direct", "routing-policy"])
def test_a_finish_chunk_with_usage_carries_the_organization_priced_cost(
    cohere_client: TestClient, make_session: Callable[[], Session], model: str
) -> None:
    created = cohere_client.post(
        f"{API_ROOT}/organizations/me/pricing",
        json={"model_key": _MODEL, "input_price_per_million": 1.0, "output_price_per_million": 1.0},
        headers=_HEADERS,
    )
    assert created.status_code == 201, created.text
    user = cohere_client.post(f"{API_ROOT}/users", json={"user_id": _USER}, headers=_HEADERS)
    assert user.status_code == 200, user.text

    response = _stream(cohere_client, model)

    assert response.status_code == 200, response.text
    frames = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert "usage" not in chunks[0] or chunks[0]["usage"] is None
    finish = chunks[-1]
    assert finish["choices"][0]["finish_reason"] == "stop"
    assert finish["usage"]["cost_usd"] == "0.300000"
    assert finish["usage"]["pricing_source"] == "organization"
    assert _logged_cost(make_session) == Decimal("0.300000")
