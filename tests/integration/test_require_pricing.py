"""End-to-end tests for the require_pricing gate (F3) and its precedence.

These build a client with ``require_pricing=True`` (the production default; the
shared ``client`` fixture turns it off for the legacy suite). They cover the 402
route branch — exercised nowhere else — and verify that user/blocked/budget
rejections (404/403) take precedence over the missing-pricing rejection (402),
i.e. budget is reserved before the pricing gate is enforced.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

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


def test_missing_pricing_rejection_is_recorded_in_the_usage_log(strict_pricing_client: TestClient) -> None:
    """A 402 rejection is logged as an error row so an operator can see dropped traffic.

    Regression for #317: the gate refunded the reservation and raised without
    writing anything, so requests dropped for missing pricing were invisible to
    every admin view (the activity log, the error rate, the pricing alarm's
    count). Cost stays null: nothing was spent.
    """
    c = strict_pricing_client
    c.post("/v1/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    assert _chat(c, model="openai:gpt-4o", user="priced-user") == 402

    rows = c.get("/v1/usage", params={"status": "error"}, headers=_MASTER_HEADER).json()
    assert len(rows) == 1
    # The resolved target, not the request selector: the same form every success
    # row on this pipeline uses, so a model's failures and successes group
    # together under one key in the activity log, the model filter (an exact
    # match on this column), and usage-by-model. #449 logged the full selector
    # here, which split them; #465 unified it.
    assert rows[0]["model"] == "gpt-4o"
    assert rows[0]["provider"] == "openai"
    assert rows[0]["endpoint"] == "/v1/chat/completions"
    assert rows[0]["user_id"] == "priced-user"
    assert rows[0]["status"] == "error"
    assert rows[0]["cost"] is None
    assert "pricing" in rows[0]["error_message"].lower()
    # Always an enforced row, never an imported-looking one: the dashboard treats
    # counts_toward_budget=False as imported and offers those to bulk delete and
    # set-price, which must never reach a row the gateway wrote itself.
    assert rows[0]["counts_toward_budget"] is True

    # The dashboard's live "N failed in the last hour" signal reads this count.
    count = c.get("/v1/usage/count", params={"status": "error"}, headers=_MASTER_HEADER).json()
    assert count["total"] == 1
    # And it reads it scoped to gateway traffic, which these rows must satisfy or
    # the alarm would undercount its own rejections.
    scoped = c.get(
        "/v1/usage/count", params={"status": "error", "source": "gateway"}, headers=_MASTER_HEADER
    ).json()
    assert scoped["total"] == 1


def test_passthrough_missing_pricing_rejection_is_recorded_too(strict_pricing_client: TestClient) -> None:
    """The pass-through gate (embeddings, images, rerank) records its 402 as well,
    so the failure count covers every rejected request, not only chat."""
    c = strict_pricing_client
    c.post("/v1/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    resp = c.post(
        "/v1/embeddings",
        json={"model": "openai:text-embedding-3-small", "input": "hi", "user": "priced-user"},
        headers=_MASTER_HEADER,
    )
    assert resp.status_code == 402

    rows = c.get("/v1/usage", params={"status": "error"}, headers=_MASTER_HEADER).json()
    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/v1/embeddings"
    assert rows[0]["cost"] is None
    assert rows[0]["counts_toward_budget"] is True
    # Bare model with the instance in `provider`, the one form both scaffolds now
    # use for every row they write (see the chat assertion above).
    assert rows[0]["model"] == "text-embedding-3-small"
    assert rows[0]["provider"] == "openai"


def test_budget_exempt_key_writes_no_pricing_rejection_row(strict_pricing_client: TestClient) -> None:
    """A budget-exempt key skips the gate, so no missing-pricing row is written.

    The gate is guarded by ``not budget_exempt``: an exempt key is never debited,
    so there is no budget for the require_pricing safety gate to protect. Nothing
    else pins that, so pin it here. (Whether a rejection row could ever carry
    counts_toward_budget=False is pinned separately, per gate, in
    test_gateway_rejection_logging.py; the shared writer fixes the flag at True.)

    Structured as an A/B against an enforced key rather than as a bare "no row
    appeared": this fixture configures no providers, so an exempt request fails
    before any provider call and logs nothing either way. Without the control, the
    negative would pass even if the gate had stopped writing rows entirely.
    """
    c = strict_pricing_client
    c.post("/v1/users", json={"user_id": "gate-user"}, headers=_MASTER_HEADER)

    def issue_key(name: str, *, exempt: bool) -> str:
        body = {"key_name": name, "user_id": "gate-user", "exclude_from_budget": exempt}
        return str(c.post("/v1/keys", json=body, headers=_MASTER_HEADER).json()["key"])

    def chat(key: str) -> int:
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "openai:gpt-4o", "messages": _MESSAGES},
            headers={API_KEY_HEADER: f"Bearer {key}"},
        )
        return int(resp.status_code)

    def pricing_rejections() -> list[dict[str, Any]]:
        rows = c.get("/v1/usage", params={"status": "error"}, headers=_MASTER_HEADER).json()
        return [r for r in rows if "No pricing is configured" in (r["error_message"] or "")]

    # Control: an enforced key hits the gate and leaves exactly one row, carrying
    # the enforced flag. This is what makes the negative below meaningful.
    assert chat(issue_key("enforced", exempt=False)) == 402
    control = pricing_rejections()
    assert len(control) == 1
    assert control[0]["counts_toward_budget"] is True

    # The exempt key skips the gate, so it adds no rejection row of its own.
    assert chat(issue_key("exempt", exempt=True)) != 402
    assert len(pricing_rejections()) == 1


def test_priced_model_passes_the_gate(strict_pricing_client: TestClient) -> None:
    """A priced model clears the pricing gate (no 402); any later failure is a provider error."""
    strict_pricing_client.post("/v1/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    strict_pricing_client.post(
        "/v1/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=_MASTER_HEADER,
    )
    assert _chat(strict_pricing_client, model="openai:gpt-4o", user="priced-user") != 402


def test_budget_exempt_key_skips_pricing_gate(strict_pricing_client: TestClient) -> None:
    """A key exempt from budget clears the require_pricing gate: unpriced usage is
    allowed through (logged with cost=null), not 402, since there is no budget to
    protect. The call then fails as a provider error, never a 402."""
    c = strict_pricing_client
    c.post("/v1/users", json={"user_id": "exempt-user"}, headers=_MASTER_HEADER)
    key = c.post(
        "/v1/keys",
        json={"key_name": "exempt", "user_id": "exempt-user", "exclude_from_budget": True},
        headers=_MASTER_HEADER,
    ).json()["key"]
    resp = c.post(
        "/v1/chat/completions",
        json={"model": "openai:gpt-4o", "messages": _MESSAGES},
        headers={API_KEY_HEADER: f"Bearer {key}"},
    )
    assert resp.status_code != 402


def test_blocked_user_takes_precedence_over_missing_pricing(strict_pricing_client: TestClient) -> None:
    """A blocked user gets 403, not 402 — budget/state is checked before pricing."""
    strict_pricing_client.post(
        "/v1/users", json={"user_id": "blocked-user", "blocked": True}, headers=_MASTER_HEADER
    )
    assert _chat(strict_pricing_client, model="openai:gpt-4o", user="blocked-user") == 403


def test_missing_user_takes_precedence_over_missing_pricing(strict_pricing_client: TestClient) -> None:
    """A nonexistent user gets 404, not 402 — user existence is checked before pricing."""
    assert _chat(strict_pricing_client, model="openai:gpt-4o", user="ghost-user") == 404
