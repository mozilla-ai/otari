"""A dashboard session cannot mint itself the deployment master key.

``rotate_master_key`` hands the caller the deployment credential itself, which is what
``hand_over_bootstrap_operator`` and the bootstrap operator's password link assume a session
cannot hold. From a session it is therefore held to the handover's proof: only the identity the
master key acts as, and only after re-proving its password. A header master key is that
credential already, so it is exempt (``test_settings_endpoint`` covers that path).
"""

import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import create_app
from gateway.models.tenancy import DashboardSession, User
from gateway.services import master_key_service
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.password_service import hash_password

ROTATE = f"{API_ROOT}/settings/master-key/rotate"

T = TypeVar("T")


def _with_session(database_url: str, work: Callable[[Session], T]) -> T:
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            return work(session)
    finally:
        engine.dispose()


def _mint_session(database_url: str, user_id: uuid.UUID, token: str) -> None:
    def work(session: Session) -> None:
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=user_id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        )
        session.commit()

    _with_session(database_url, work)


def _bootstrap_operator_id(database_url: str) -> uuid.UUID:
    return _with_session(database_url, lambda session: session.query(User).one().id)


def _add_superuser(database_url: str, *, email: str) -> uuid.UUID:
    def work(session: Session) -> uuid.UUID:
        user = User(email=email, full_name="Ada", is_superuser=True)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.id

    return _with_session(database_url, work)


def _set_password(database_url: str, user_id: uuid.UUID, password: str) -> None:
    def work(session: Session) -> None:
        session.query(User).filter(col(User.id) == user_id).update({"hashed_password": hash_password(password)})
        session.commit()

    _with_session(database_url, work)


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, str]]:
    tokens = iter(["otari-mk-first", "otari-mk-second"])
    monkeypatch.setattr(master_key_service, "generate_master_key", lambda: next(tokens))
    config = GatewayConfig(database_url=f"sqlite:///{tmp_path / 'rotation.db'}", require_pricing=False)
    with TestClient(create_app(config)) as client:
        # One master-key request provisions the tenancy root and its bootstrap operator.
        provisioned = client.get(f"{API_ROOT}/organizations/me", headers={"Authorization": "Bearer otari-mk-first"})
        assert provisioned.status_code == 200, provisioned.text
        yield client, config.database_url


def test_a_non_bootstrap_operator_session_cannot_rotate(app: tuple[TestClient, str]) -> None:
    client, database_url = app
    other = _add_superuser(database_url, email="ada@example.com")
    _mint_session(database_url, other, "otari-sess-ada")
    client.cookies.set(SESSION_COOKIE_NAME, "otari-sess-ada")
    try:
        response = client.post(ROTATE, json={"current_password": "irrelevant"})
    finally:
        client.cookies.clear()
    assert response.status_code == 400, response.text
    assert "bootstrap operator" in response.json()["detail"]


def test_the_bootstrap_operator_session_must_prove_its_password(app: tuple[TestClient, str]) -> None:
    client, database_url = app
    operator = _bootstrap_operator_id(database_url)
    _mint_session(database_url, operator, "otari-sess-operator")
    client.cookies.set(SESSION_COOKIE_NAME, "otari-sess-operator")
    try:
        # The account has no password yet, so there is nothing to prove.
        assert client.post(ROTATE, json={}).status_code == 400
        _set_password(database_url, operator, "correct horse battery")
        assert client.post(ROTATE, json={"current_password": "wrong"}).status_code == 400
        rotated = client.post(ROTATE, json={"current_password": "correct horse battery"})
        assert rotated.status_code == 200, rotated.text
        assert rotated.json()["master_key"].startswith("otari-mk-")
    finally:
        client.cookies.clear()
