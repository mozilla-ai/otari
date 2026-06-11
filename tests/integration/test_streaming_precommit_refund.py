"""Regression tests for the streaming pre-commit reservation leak.

A streaming request whose upstream provider call fails *before the first chunk*
("pre-commit") must release the budget pre-debit hold, the same as the
non-streaming handlers do. Previously ``/v1/messages`` and ``/v1/responses``
logged the error but left ``users.reserved`` inflated (a leak that was never
released, since a budget-period reset zeroes ``spend`` but not ``reserved``).
``/v1/chat/completions`` already refunded here; it is covered too as a parity
guard so all three endpoints stay consistent.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from gateway.core.config import GatewayConfig
from gateway.models.entities import UsageLog, User

from .conftest import MODEL_NAME

_PRICING = {"input_price_per_million": 2.5, "output_price_per_million": 10.0}


def _seed_budgeted_user(client: TestClient, headers: dict[str, str], user_id: str) -> None:
    budget = client.post("/v1/budgets", json={"max_budget": 100.0}, headers=headers)
    assert budget.status_code == 200
    budget_id = budget.json()["budget_id"]
    created = client.post(
        "/v1/users",
        json={"user_id": user_id, "budget_id": budget_id},
        headers=headers,
    )
    assert created.status_code == 200


def _configure_pricing(client: TestClient, headers: dict[str, str], model_key: str) -> None:
    res = client.post("/v1/pricing", json={"model_key": model_key, **_PRICING}, headers=headers)
    assert res.status_code == 200


def _user_state(test_config: GatewayConfig, user_id: str) -> tuple[float, float]:
    engine = create_engine(test_config.database_url)
    session_local = sessionmaker(bind=engine)
    db = session_local()
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        assert user is not None
        return float(user.spend), float(user.reserved)
    finally:
        db.close()
        engine.dispose()


def _poll_usage_logs(test_config: GatewayConfig, user_id: str, *, timeout: float = 3.0) -> list[str]:
    engine = create_engine(test_config.database_url)
    session_local = sessionmaker(bind=engine)
    deadline = time.time() + timeout
    try:
        while True:
            db = session_local()
            try:
                rows = db.query(UsageLog).filter(UsageLog.user_id == user_id).all()
                if rows or time.time() > deadline:
                    return [r.status for r in rows]
            finally:
                db.close()
            time.sleep(0.1)
    finally:
        engine.dispose()


def test_chat_streaming_precommit_error_refunds(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    user_id = "precommit-chat"
    _seed_budgeted_user(client, master_key_header, user_id)
    _configure_pricing(client, master_key_header, MODEL_NAME)

    async def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("simulated upstream failure before first chunk")

    with patch("gateway.api.routes.chat.acompletion", side_effect=_boom):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "user": user_id,
                "stream": True,
            },
            headers=master_key_header,
        )

    assert response.status_code == 502
    spend, reserved = _user_state(test_config, user_id)
    assert spend == pytest.approx(0.0)
    assert reserved == pytest.approx(0.0)
    assert _poll_usage_logs(test_config, user_id) == ["error"]


def test_messages_streaming_precommit_error_refunds(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    user_id = "precommit-messages"
    _seed_budgeted_user(client, master_key_header, user_id)
    _configure_pricing(client, master_key_header, MODEL_NAME)

    async def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("simulated upstream failure before first event")

    with patch("gateway.api.routes.messages.amessages", side_effect=_boom):
        response = client.post(
            "/v1/messages",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
                "stream": True,
                "metadata": {"user_id": user_id},
            },
            headers=master_key_header,
        )

    assert response.status_code == 500
    spend, reserved = _user_state(test_config, user_id)
    assert spend == pytest.approx(0.0)
    assert reserved == pytest.approx(0.0)  # hold refunded (was leaked before the fix)
    assert _poll_usage_logs(test_config, user_id) == ["error"]


def test_responses_streaming_precommit_error_refunds(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    user_id = "precommit-responses"
    _seed_budgeted_user(client, master_key_header, user_id)
    # openai supports the Responses API; price it so the hold is genuinely
    # non-zero before the failure refunds it.
    _configure_pricing(client, master_key_header, "openai:gpt-4o-mini")

    async def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("simulated upstream failure before first event")

    with patch("gateway.api.routes.responses.aresponses", side_effect=_boom):
        response = client.post(
            "/v1/responses",
            json={"model": "openai:gpt-4o-mini", "input": "hi", "stream": True, "user": user_id},
            headers=master_key_header,
        )

    assert response.status_code == 502
    spend, reserved = _user_state(test_config, user_id)
    assert spend == pytest.approx(0.0)
    assert reserved == pytest.approx(0.0)  # hold refunded (was leaked before the fix)
    assert _poll_usage_logs(test_config, user_id) == ["error"]
