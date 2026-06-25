"""End-to-end tests for the require_pricing gate (F3) and its precedence.

These build a client with ``require_pricing=True`` (the production default; the
shared ``client`` fixture turns it off for the legacy suite). They cover the 402
route branch — exercised nowhere else — and verify that user/blocked/budget
rejections (404/403) take precedence over the missing-pricing rejection (402),
i.e. budget is reserved before the pricing gate is enforced.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.db import Base, get_db
from gateway.main import create_app

from .conftest import _run_alembic_migrations, _to_async_url

_MASTER_HEADER = {API_KEY_HEADER: "Bearer test-master-key"}
_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def strict_pricing_client(postgres_url: str) -> Generator[TestClient]:
    """TestClient for a gateway with require_pricing=True (fail-closed).

    Default pricing is disabled so these tests exercise the missing-pricing gate
    in isolation: otherwise genai-prices would price well-known models (gpt-4o)
    and the 402 branch would never be reached.
    """
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=True,
        default_pricing=False,
    )
    _run_alembic_migrations(postgres_url)
    engine = create_engine(postgres_url, pool_pre_ping=True)
    async_engine = create_async_engine(_to_async_url(postgres_url), pool_pre_ping=True)
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
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()
        try:
            asyncio.run(async_engine.dispose())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(async_engine.dispose())
            loop.close()


def _chat(client: TestClient, *, model: str, user: str) -> int:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": _MESSAGES, "user": user},
        headers=_MASTER_HEADER,
    )
    return int(resp.status_code)


def test_unpriced_model_rejected_with_402(strict_pricing_client: TestClient) -> None:
    """An unpriced model is rejected with 402 when require_pricing is on (F3)."""
    strict_pricing_client.post("/v1/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    assert _chat(strict_pricing_client, model="openai:gpt-4o", user="priced-user") == 402


def test_priced_model_passes_the_gate(strict_pricing_client: TestClient) -> None:
    """A priced model clears the pricing gate (no 402); any later failure is a provider error."""
    strict_pricing_client.post("/v1/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    strict_pricing_client.post(
        "/v1/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=_MASTER_HEADER,
    )
    assert _chat(strict_pricing_client, model="openai:gpt-4o", user="priced-user") != 402


def test_blocked_user_takes_precedence_over_missing_pricing(strict_pricing_client: TestClient) -> None:
    """A blocked user gets 403, not 402 — budget/state is checked before pricing."""
    strict_pricing_client.post(
        "/v1/users", json={"user_id": "blocked-user", "blocked": True}, headers=_MASTER_HEADER
    )
    assert _chat(strict_pricing_client, model="openai:gpt-4o", user="blocked-user") == 403


def test_missing_user_takes_precedence_over_missing_pricing(strict_pricing_client: TestClient) -> None:
    """A nonexistent user gets 404, not 402 — user existence is checked before pricing."""
    assert _chat(strict_pricing_client, model="openai:gpt-4o", user="ghost-user") == 404
