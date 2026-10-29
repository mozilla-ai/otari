"""The Subagent router operator API over HTTP.

Who reaches it and how it refuses, that no response body ever carries key
material, and what each route does to the rows: the first router added is
on, a later one waits, switching one on switches the others off, a price
change keeps the key, and with every row gone the configured model answers
again. The decision call itself is covered by
``test_agent_recommenders_recommend.py``.
"""

import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User
from gateway.repositories.routing import AgentRecommenderRepository
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.secret_box import generate_secret_key

PATH = f"{API_ROOT}/routing/recommenders"
KEY = "sk-or-stored-router-1234"
SECOND_KEY = "ts-second-router-5678"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    """A TypeSafe decisions provider, so the configured decision model resolves and is listed."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        decision_providers={"typesafe": {"api_key": "ts-secret"}},
        agent_recommender_model="typesafe:jev-latest",
    )


def _create(client: TestClient, headers: dict[str, str], **fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "provider": "openrouter",
        "model": "stored-decider",
        "api_key": KEY,
        "input_price_per_million": 20,
    }
    body.update(fields)
    response = client.post(PATH, json=body, headers=headers)
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


def _rows(client: TestClient, headers: dict[str, str]) -> list[dict[str, Any]]:
    response = client.get(PATH, headers=headers)
    assert response.status_code == 200, response.text
    rows: list[dict[str, Any]] = response.json()["data"]
    return rows


def _session_for(session_factory: Callable[[], Session], *, email: str) -> str:
    """A member of the default organization holding a live dashboard session, and its cookie."""
    session = session_factory()
    try:
        organization_id = session.query(Organization).filter(col(Organization.slug) == "default").one().id
        user = User(email=email, full_name="Ada", active_organization_id=organization_id)
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(
            OrganizationMember(organization_id=organization_id, user_id=user.id, role="member", status="active")
        )
        token = f"otari-sess-{email}"
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=user.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        )
        session.commit()
        return token
    finally:
        session.close()


def test_with_nothing_stored_the_configured_model_is_listed(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.get(PATH, headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "data": [],
        "configured": {"selector": "typesafe:jev-latest", "provider": "typesafe", "model": "jev-latest"},
    }


def test_anyone_but_the_operator_finds_no_such_surface(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """404 rather than 403, like /api/v1/admin, on a read and on a write; a bare API key is not even that."""
    token = _session_for(db_session_factory, email="ada@example.com")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        read = client.get(PATH)
        write = client.post(PATH, json={"provider": "openrouter", "model": "m", "api_key": KEY}, headers={})
    finally:
        client.cookies.clear()

    assert read.status_code == 404
    assert write.status_code == 404
    assert client.get(PATH).status_code == 401
    assert _rows(client, master_key_header) == []


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"api_key": None}, "API key is required"),
        ({"api_base": "http://proxy.example"}, "must use https"),
        ({"provider": "nope"}, "provider must be one of"),
        ({"provider": "llamacpp", "api_key": None}, "API base is required"),
    ],
)
def test_a_router_that_could_not_work_is_refused(
    client: TestClient, master_key_header: dict[str, str], fields: dict[str, Any], problem: str
) -> None:
    body: dict[str, Any] = {"provider": "openrouter", "model": "m", "api_key": KEY, "input_price_per_million": 20}
    body.update(fields)

    response = client.post(PATH, json=body, headers=master_key_header)

    assert response.status_code == 400, response.text
    assert problem in response.json()["detail"]
    assert KEY not in response.text
    assert _rows(client, master_key_header) == []


def test_a_negative_price_or_a_margin_is_unprocessable(client: TestClient, master_key_header: dict[str, str]) -> None:
    base = {"provider": "openrouter", "model": "m", "api_key": KEY}

    negative = client.post(PATH, json={**base, "input_price_per_million": -1}, headers=master_key_header)
    margin = client.post(
        PATH, json={**base, "input_price_per_million": 1, "margin_percent": 20}, headers=master_key_header
    )

    assert negative.status_code == 422
    assert margin.status_code == 422


def test_the_first_router_added_is_switched_on_and_a_later_one_waits(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    first = _create(client, master_key_header)
    second = _create(client, master_key_header, provider="typesafe", model="second-decider", api_key=SECOND_KEY)

    assert first["enabled"] is True
    assert second["enabled"] is False
    assert {key: first[key] for key in ("provider", "model", "api_base", "api_key_last4", "has_api_key")} == {
        "provider": "openrouter",
        "model": "stored-decider",
        "api_base": None,
        "api_key_last4": "1234",
        "has_api_key": True,
    }
    assert (first["input_price_per_million"], first["output_price_per_million"]) == (20.0, 0.0)
    assert [(row["model"], row["enabled"]) for row in _rows(client, master_key_header)] == [
        ("stored-decider", True),
        ("second-decider", False),
    ]


def test_a_new_price_keeps_the_stored_key_and_the_other_price(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    created = _create(client, master_key_header, output_price_per_million=5)
    before = _ciphertext(db_session_factory)

    response = client.patch(f"{PATH}/{created['id']}", json={"input_price_per_million": 30}, headers=master_key_header)

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["input_price_per_million"], body["output_price_per_million"]) == (30.0, 5.0)
    assert (body["has_api_key"], body["api_key_last4"]) == (True, "1234")
    assert _ciphertext(db_session_factory) == before


def test_a_new_provider_or_address_needs_its_own_key_and_the_same_address_keeps_it(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = _create(client, master_key_header, api_base="https://proxy.example/api")
    url = f"{PATH}/{created['id']}"

    new_provider = client.patch(url, json={"provider": "typesafe"}, headers=master_key_header)
    new_base = client.patch(url, json={"api_base": "https://other.example"}, headers=master_key_header)
    same_base = client.patch(url, json={"api_base": "https://proxy.example/api"}, headers=master_key_header)

    assert new_provider.status_code == 400 and "API key is required" in new_provider.text
    assert new_base.status_code == 400 and "API key is required" in new_base.text
    assert same_base.status_code == 200, same_base.text
    assert same_base.json()["has_api_key"] is True


def test_switching_one_router_on_switches_the_other_off(client: TestClient, master_key_header: dict[str, str]) -> None:
    first = _create(client, master_key_header)
    second = _create(client, master_key_header, provider="typesafe", model="second-decider", api_key=SECOND_KEY)

    switched = client.patch(f"{PATH}/{second['id']}", json={"enabled": True}, headers=master_key_header)

    assert switched.status_code == 200, switched.text
    assert switched.json()["enabled"] is True
    assert [(row["id"], row["enabled"]) for row in _rows(client, master_key_header)] == [
        (first["id"], False),
        (second["id"], True),
    ]


def test_two_switched_on_at_the_same_moment_is_a_conflict(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The partial unique index is the last word, and its refusal is a 409 rather than a 500."""
    _create(client, master_key_header)
    second = _create(client, master_key_header, provider="typesafe", model="second-decider", api_key=SECOND_KEY)

    async def nothing_switched_off(self: AgentRecommenderRepository, recommender_id: uuid.UUID) -> None:
        del self, recommender_id

    monkeypatch.setattr(AgentRecommenderRepository, "switch_off_all_but", nothing_switched_off)
    response = client.patch(f"{PATH}/{second['id']}", json={"enabled": True}, headers=master_key_header)

    assert response.status_code == 409, response.text
    assert "switched on at the same moment" in response.json()["detail"]
    assert [row["enabled"] for row in _rows(client, master_key_header)] == [True, False]


def test_a_router_that_is_not_there_is_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    missing = uuid.uuid4()

    assert client.patch(f"{PATH}/{missing}", json={"enabled": True}, headers=master_key_header).status_code == 404
    assert client.delete(f"{PATH}/{missing}", headers=master_key_header).status_code == 404


def test_removing_every_router_hands_back_to_the_configured_model(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    first = _create(client, master_key_header)
    second = _create(client, master_key_header, provider="typesafe", model="second-decider", api_key=SECOND_KEY)

    assert client.delete(f"{PATH}/{first['id']}", headers=master_key_header).status_code == 204
    assert client.delete(f"{PATH}/{second['id']}", headers=master_key_header).status_code == 204
    assert client.delete(f"{PATH}/{second['id']}", headers=master_key_header).status_code == 404

    listed = client.get(PATH, headers=master_key_header).json()
    assert listed["data"] == []
    assert listed["configured"]["selector"] == "typesafe:jev-latest"


def test_the_keys_are_stored_encrypted_and_never_answered(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    created = _create(client, master_key_header)
    repriced = client.patch(f"{PATH}/{created['id']}", json={"output_price_per_million": 1}, headers=master_key_header)
    listed = client.get(PATH, headers=master_key_header)

    for body in (repriced.text, listed.text):
        assert KEY not in body
        assert "encrypted" not in body
    stored = _ciphertext(db_session_factory)
    assert stored is not None
    assert KEY not in stored


def test_a_key_too_short_to_show_is_still_known_to_be_stored(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = _create(client, master_key_header, api_key="abcd")

    assert (created["api_key_last4"], created["has_api_key"]) == (None, True)


def _ciphertext(session_factory: Callable[[], Session]) -> str | None:
    session = session_factory()
    try:
        value = session.execute(text("SELECT encrypted_api_key FROM agent_recommenders")).scalar_one()
        return None if value is None else str(value)
    finally:
        session.close()
