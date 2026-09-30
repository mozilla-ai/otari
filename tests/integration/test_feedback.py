"""Feedback authentication, explicit payloads, and disabled deployments."""

import json
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.api import deps
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.tenancy import DashboardSession, User
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.feedback import FeedbackService

from .conftest import build_test_client


@pytest.fixture
def test_config(test_config: GatewayConfig) -> GatewayConfig:
    return test_config.model_copy(update={"feedback_enabled": True})


@pytest.fixture
def deliveries(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    calls: list[httpx.Request] = []

    def receive(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(204)

    monkeypatch.setattr(deps, "FeedbackService", partial(FeedbackService, transport=httpx.MockTransport(receive)))
    return calls


def test_operator_submission_is_explicit(
    client: TestClient, test_config: GatewayConfig, deliveries: list[httpx.Request]
) -> None:
    assert client.get(f"{API_ROOT}/bootstrap").json()["feedback_enabled"] is True
    assert not deliveries
    assert client.post(f"{API_ROOT}/auth/session", json={"master_key": test_config.master_key}).status_code == 200
    response = client.post(
        f"{API_ROOT}/feedback",
        json={"message": " Useful "},
        headers={"Referer": "https://private.test/page", "X-Forwarded-For": "192.0.2.1"},
    )
    assert response.status_code == 204
    assert len(deliveries) == 1
    request = deliveries[0]
    assert str(request.url) == f"https://api.otari.ai{API_ROOT}/feedback/submissions"
    assert json.loads(request.content) == {"message": "Useful"}
    assert not {"authorization", "x-api-key", "cookie", "referer", "origin", "x-forwarded-for"} & set(request.headers)


def test_ordinary_member_can_submit(
    client: TestClient, db_session: Session, deliveries: list[httpx.Request], master_key_header: dict[str, str]
) -> None:
    added = client.post(
        f"{API_ROOT}/organizations/me/members", json={"email": "member@example.com"}, headers=master_key_header
    )
    assert added.status_code == 201, added.text
    user = db_session.scalars(select(User).where(col(User.email) == "member@example.com")).one()
    token = "feedback-member-session"
    db_session.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user.id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    db_session.commit()
    client.cookies.set(SESSION_COOKIE_NAME, token)
    assert client.post(f"{API_ROOT}/feedback", json={"message": "A member's idea"}).status_code == 204
    assert json.loads(deliveries[0].content) == {"message": "A member's idea"}
    assert (
        client.post(
            f"{API_ROOT}/feedback", json={"message": "forged"}, headers={"Sec-Fetch-Site": "cross-site"}
        ).status_code
        == 401
    )
    assert len(deliveries) == 1


def test_unauthenticated_and_api_key_callers_are_refused(
    client: TestClient, api_key_header: dict[str, str], deliveries: list[httpx.Request]
) -> None:
    for headers in ({}, api_key_header):
        assert client.post(f"{API_ROOT}/feedback", json={"message": "Idea"}, headers=headers).status_code in (401, 403)
    assert not deliveries


@pytest.mark.parametrize(
    "body",
    [
        {"message": " "},
        {"message": "a" * 4001},
        {"message": "ok", "url": "private"},
        {"message": "ok", "email": "private@example.com"},
        {"message": "\ud800"},
    ],
)
def test_invalid_feedback_does_not_leave_gateway(
    client: TestClient, master_key_header: dict[str, str], deliveries: list[httpx.Request], body: dict[str, Any]
) -> None:
    response = client.post(
        f"{API_ROOT}/feedback",
        content=json.dumps(body),
        headers={**master_key_header, "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "private" not in response.text
    assert not deliveries


def test_each_caller_is_limited_to_five_sends_per_window(
    client: TestClient,
    db_session: Session,
    master_key_header: dict[str, str],
    deliveries: list[httpx.Request],
) -> None:
    for _ in range(5):
        sent = client.post(f"{API_ROOT}/feedback", json={"message": "Idea"}, headers=master_key_header)
        assert sent.status_code == 204
    refused = client.post(f"{API_ROOT}/feedback", json={"message": "One more"}, headers=master_key_header)
    assert refused.status_code == 429
    assert 1 <= int(refused.headers["Retry-After"]) <= 600
    assert len(deliveries) == 5

    # A body that never leaves the gateway spends nothing, and the limit is the
    # caller's own: a member signed in on the same gateway still gets through.
    assert client.post(f"{API_ROOT}/feedback", json={"message": " "}, headers=master_key_header).status_code == 422
    added = client.post(
        f"{API_ROOT}/organizations/me/members", json={"email": "member@example.com"}, headers=master_key_header
    )
    assert added.status_code == 201, added.text
    user = db_session.scalars(select(User).where(col(User.email) == "member@example.com")).one()
    token = "feedback-limit-member"
    db_session.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user.id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    db_session.commit()
    client.cookies.set(SESSION_COOKIE_NAME, token)
    assert client.post(f"{API_ROOT}/feedback", json={"message": "Mine"}).status_code == 204
    assert len(deliveries) == 6


def test_chunked_body_limit(
    client: TestClient, master_key_header: dict[str, str], deliveries: list[httpx.Request]
) -> None:
    response = client.post(
        f"{API_ROOT}/feedback",
        content=iter([b"x" * 20000, b"y" * 20000]),
        headers={**master_key_header, "Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert not deliveries


def test_disabled_feedback_is_not_mounted(test_config: GatewayConfig, deliveries: list[httpx.Request]) -> None:
    config = test_config.model_copy(update={"feedback_enabled": False})
    for client in build_test_client(config):
        assert client.get(f"{API_ROOT}/bootstrap").json()["feedback_enabled"] is False
        assert client.post(f"{API_ROOT}/feedback", json={"message": "Idea"}).status_code == 404
    assert not deliveries
