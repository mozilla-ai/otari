"""The subagent recommenders an operator adds, and the recommendation each one answers at its own price.

``/api/v1/routing/recommenders`` stores a decision model with the deployment's
own key for it and a price per million of its tokens. The row switched on
answers ``POST /api/v1/routing/recommend``, held at the most one call can cost
and charged for the tokens it used; with no row added, ``agent_recommender_model``
answers, and with rows added and every one off, nothing does. The upstream call
is stubbed at ``request_decision`` where the core recommender calls it.

A recommender an overlay binds over the core one, to set a price of its own,
is covered at the end: it wraps the real adapter and doubles what it charges.
"""

import itertools
import sys
import uuid
from collections.abc import Callable, Generator, Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.routing.recommend import RECOMMENDATION_QUESTION
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client

PATH = f"{API_ROOT}/routing/recommenders"
RECOMMEND = f"{API_ROOT}/routing/recommend"
KEY = "or-live-secret-9876"
SPAWN: dict[str, Any] = {
    "harness": "claude-code",
    "session_id": "bf05abe2-5ff2-4eb0-8459-ab578d5c9468",
    "tool_use_id": "toolu_01",
    "agent_type": "Explore",
    "description": "List files",
    "prompt": "List the files under src/ and report what each module does.",
    "parent_model": "claude-opus-5",
    "requested_model": None,
}
ANSWER: dict[str, Any] = {
    "model": "gpt-5-mini",
    "answers": {RECOMMENDATION_QUESTION: {"type": "choice", "choice": "haiku"}},
    "usage": {"input_tokens": 1_000, "output_tokens": 10},
}
ROUTER: dict[str, Any] = {
    "provider": "openrouter",
    "model": "openai/gpt-5-mini",
    "api_key": KEY,
    "input_price_per_million": 2,
    "output_price_per_million": 10,
}
# 1,000 input tokens at $2 and 10 output at $10 per million.
TOKEN_PRICE = 0.0021


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    """A TypeSafe decision provider for ``agent_recommender_model`` to name while no row is added."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        decision_providers={"typesafe": {"api_key": "ts-secret"}},
    )


def _mock_decision(answer: dict[str, Any] | None = None) -> Any:
    return patch(
        "gateway.adapters.agent_model_recommender_adapter.request_decision",
        AsyncMock(return_value=answer if answer is not None else ANSWER),
    )


def _add(client: TestClient, headers: dict[str, str], **overrides: Any) -> dict[str, Any]:
    response = client.post(PATH, json={**ROUTER, **overrides}, headers=headers)
    assert response.status_code == status.HTTP_201_CREATED, response.text
    assert KEY not in response.text
    body: dict[str, Any] = response.json()
    return body


def _switch_on(client: TestClient, headers: dict[str, str], recommender_id: str) -> None:
    response = client.patch(f"{PATH}/{recommender_id}", json={"enabled": True}, headers=headers)
    assert response.status_code == status.HTTP_200_OK, response.text


def _usage_rows(client: TestClient, headers: dict[str, str], user_id: str) -> list[dict[str, Any]]:
    rows = client.get(f"{API_ROOT}/usage", params={"user_id": user_id, "endpoint": "/v1/decisions"}, headers=headers)
    assert rows.status_code == status.HTTP_200_OK, rows.text
    return [dict(row) for row in rows.json()]


# --- The operator's surface --------------------------------------------------


def test_an_operator_adds_lists_changes_and_removes_a_recommender(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    added = _add(client, master_key_header)

    assert added["enabled"] is False
    assert (added["has_api_key"], added["api_key_last4"]) == (True, "9876")
    assert (added["input_price_per_million"], added["output_price_per_million"]) == (2.0, 10.0)

    listed = client.get(PATH, headers=master_key_header)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    assert KEY not in listed.text
    body = listed.json()
    assert [row["id"] for row in body["data"]] == [added["id"]]
    assert body["decision_providers"] == [
        {"provider": "typesafe", "self_hosted": False},
        {"provider": "openrouter", "self_hosted": False},
        {"provider": "llamacpp", "self_hosted": True},
    ]

    # A price change re-sends no key, and the stored one is kept.
    repriced = client.patch(f"{PATH}/{added['id']}", json={"output_price_per_million": 12.5}, headers=master_key_header)
    assert repriced.status_code == status.HTTP_200_OK, repriced.text
    assert repriced.json()["output_price_per_million"] == 12.5
    assert (repriced.json()["has_api_key"], repriced.json()["api_key_last4"]) == (True, "9876")

    removed = client.delete(f"{PATH}/{added['id']}", headers=master_key_header)
    assert removed.status_code == status.HTTP_204_NO_CONTENT, removed.text
    assert client.get(PATH, headers=master_key_header).json()["data"] == []


def test_switching_one_on_switches_the_others_off(client: TestClient, master_key_header: dict[str, str]) -> None:
    first = _add(client, master_key_header, enabled=True)
    second = _add(client, master_key_header, model="openai/gpt-5-nano")

    _switch_on(client, master_key_header, second["id"])

    rows = {row["id"]: row["enabled"] for row in client.get(PATH, headers=master_key_header).json()["data"]}
    assert rows == {first["id"]: False, second["id"]: True}


@pytest.mark.parametrize(
    ("body", "complaint"),
    [
        pytest.param({"input_price_per_million": 0, "output_price_per_million": 0}, "above zero", id="free"),
        pytest.param({"provider": "openai"}, "not a supported decision provider", id="unknown-provider"),
        pytest.param({"api_key": None}, "api_key is required", id="no-key"),
        pytest.param({"api_base": "http://proxy.example"}, "https", id="key-over-http"),
        pytest.param({"provider": "llamacpp", "api_key": None}, "api_base is required", id="server-with-no-address"),
    ],
)
def test_a_recommender_that_could_not_work_is_refused(
    client: TestClient, master_key_header: dict[str, str], body: dict[str, Any], complaint: str
) -> None:
    response = client.post(PATH, json={**ROUTER, **body}, headers=master_key_header)

    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text
    assert complaint in response.json()["detail"]
    assert client.get(PATH, headers=master_key_header).json()["data"] == []


def test_a_negative_price_is_unprocessable(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(PATH, json={**ROUTER, "input_price_per_million": -1}, headers=master_key_header)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_a_new_address_needs_the_key_again(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The key is sent to the API base, so it does not follow the row to another one."""
    added = _add(client, master_key_header)

    moved = client.patch(
        f"{PATH}/{added['id']}", json={"api_base": "https://elsewhere.example"}, headers=master_key_header
    )
    assert moved.status_code == status.HTTP_400_BAD_REQUEST, moved.text
    assert "api_key is required" in moved.json()["detail"]

    with_key = client.patch(
        f"{PATH}/{added['id']}",
        json={"api_base": "https://elsewhere.example", "api_key": "or-other-secret-4321"},
        headers=master_key_header,
    )
    assert with_key.status_code == status.HTTP_200_OK, with_key.text
    assert (with_key.json()["api_base"], with_key.json()["api_key_last4"]) == ("https://elsewhere.example", "4321")


def test_an_unknown_recommender_is_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    unknown = f"{PATH}/{uuid.uuid4()}"

    assert client.patch(unknown, json={"enabled": True}, headers=master_key_header).status_code == 404
    assert client.delete(unknown, headers=master_key_header).status_code == 404


def _member_cookie(client: TestClient, master: dict[str, str], factory: Callable[[], Session]) -> str:
    """A signed-in member of the default organization who does not operate the deployment."""
    assert client.get(f"{API_ROOT}/organizations/me", headers=master).status_code == 200
    session = factory()
    try:
        organization = session.query(Organization).filter(col(Organization.slug) == "default").one()
        workspace = session.query(Workspace).filter(col(Workspace.organization_id) == organization.id).first()
        assert workspace is not None
        user = User(email="member@example.com", full_name="Member", active_organization_id=organization.id)
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(OrganizationMember(organization_id=organization.id, user_id=user.id, role="owner", status="active"))
        session.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="member", status="active"))
        token = "otari-sess-member"
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=user.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        )
        session.commit()
    finally:
        session.close()
    return token


def _row_count(factory: Callable[[], Session]) -> int:
    session = factory()
    try:
        return int(session.execute(text("SELECT count(*) FROM agent_recommenders")).scalar_one())
    finally:
        session.close()


def test_only_the_deployments_operator_reaches_the_surface(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """An API key gets the master-key check's 401, and a signed-in non-operator the 404 /admin answers."""
    added = _add(client, master_key_header)
    cookie = _member_cookie(client, master_key_header, db_session_factory)
    calls: list[tuple[str, str, dict[str, Any] | None]] = [
        ("GET", PATH, None),
        ("POST", PATH, ROUTER),
        ("PATCH", f"{PATH}/{added['id']}", {"enabled": True}),
        ("DELETE", f"{PATH}/{added['id']}", None),
    ]

    for method, url, body in calls:
        assert client.request(method, url, json=body).status_code == 401
        assert client.request(method, url, json=body, headers=api_key_header).status_code == 401
        client.cookies.set(SESSION_COOKIE_NAME, cookie)
        try:
            refused = client.request(method, url, json=body)
        finally:
            client.cookies.clear()
        assert refused.status_code == 404, f"{method} {url}: {refused.text}"

    assert _row_count(db_session_factory) == 1
    assert client.get(PATH, headers=master_key_header).json()["data"][0]["enabled"] is False


@pytest.mark.parametrize("mode", ["standalone", "hosted"])
def test_both_topologies_serve_the_surface(
    test_config: GatewayConfig, clean_database: None, master_key_header: dict[str, str], mode: str
) -> None:
    for client in build_test_client(test_config.model_copy(update={"mode": mode})):
        assert client.get(PATH, headers=master_key_header).status_code == status.HTTP_200_OK


# --- The recommendation it answers -------------------------------------------


def test_the_row_switched_on_answers_at_its_token_price(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    _add(client, master_key_header, enabled=True)

    with _mock_decision() as mock:
        response = client.post(RECOMMEND, json=SPAWN, headers=api_key_header)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["model"] == "haiku"
    provider, sent = mock.await_args.args
    assert (provider.provider, provider.api_key, sent["model"]) == ("openrouter", KEY, "openai/gpt-5-mini")
    (row,) = _usage_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["status"], row["provider"], row["model"]) == ("success", "openrouter", "openai/gpt-5-mini")
    assert float(row["cost"]) == pytest.approx(TOKEN_PRICE)


def test_a_create_lands_off_and_answers_once_switched_on(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    """The dashboard's submit: the row lands off, the edition saves what it adds, then the row is switched on."""
    added = _add(client, master_key_header)

    with _mock_decision() as mock:
        refused = client.post(RECOMMEND, json=SPAWN, headers=api_key_header)
    assert refused.status_code == status.HTTP_400_BAD_REQUEST, refused.text
    mock.assert_not_awaited()

    _switch_on(client, master_key_header, added["id"])
    with _mock_decision():
        answered = client.post(RECOMMEND, json=SPAWN, headers=api_key_header)
    assert answered.status_code == status.HTTP_200_OK, answered.text


def test_with_every_row_switched_off_nothing_answers(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    """The configured model is not a fallback for a choice the operator made in the dashboard."""
    added = _add(client, master_key_header, enabled=True)
    switched_off = client.patch(f"{PATH}/{added['id']}", json={"enabled": False}, headers=master_key_header)
    assert switched_off.status_code == status.HTTP_200_OK, switched_off.text

    with _mock_decision() as mock:
        response = client.post(RECOMMEND, json=SPAWN, headers=api_key_header)

    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text
    assert "switched on" in response.json()["detail"]
    mock.assert_not_awaited()


def test_with_no_row_added_the_configured_model_answers(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    added = _add(client, master_key_header, enabled=True)
    assert client.delete(f"{PATH}/{added['id']}", headers=master_key_header).status_code == 204

    with _mock_decision() as mock:
        response = client.post(RECOMMEND, json=SPAWN, headers=api_key_header)

    assert response.status_code == status.HTTP_200_OK, response.text
    provider, sent = mock.await_args.args
    assert (provider.name, sent["model"]) == ("typesafe", "jev-latest")
    (row,) = _usage_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["provider"], row["model"]) == ("typesafe", "jev-latest")


# --- An edition's own price --------------------------------------------------

MODULE = "doubling_recommender"
DOUBLING_BOOTSTRAP = """
from dataclasses import replace

from gateway.container import Container
from gateway.ports.agent_model_recommender_port import AgentModelRecommenderPort

QUOTED = []


class Doubling:
    def __init__(self, inner):
        self.inner = inner

    def quote(self):
        quote = self.inner.quote()
        QUOTED.append(self.inner.quoted_recommender_id)
        return replace(quote, charge=None if quote.charge is None else quote.charge * 2)

    async def recommend(self, spawn, *, candidates):
        recommendation = await self.inner.recommend(spawn, candidates=candidates)
        usage = recommendation.usage
        charge = None if usage.charge is None else usage.charge * 2
        return replace(recommendation, usage=replace(usage, charge=charge))


def register(container: Container) -> None:
    core = dict(container.bindings())[AgentModelRecommenderPort]
    container.bind(AgentModelRecommenderPort, lambda session: Doubling(core(session)))
"""
_NAMES = (f"{MODULE}_{n}" for n in itertools.count())


@pytest.fixture
def doubled(
    test_config: GatewayConfig, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, ModuleType]]:
    name = next(_NAMES)
    (tmp_path / f"{name}.py").write_text(DOUBLING_BOOTSTRAP)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    try:
        for client in build_test_client(test_config.model_copy(update={"bootstrap": f"{name}:register"})):
            yield client, sys.modules[name]
    finally:
        sys.modules.pop(name, None)


def test_a_recommender_wrapped_over_the_core_one_sets_its_own_price(
    doubled: tuple[TestClient, ModuleType], master_key_header: dict[str, str]
) -> None:
    client, probe = doubled
    added = _add(client, master_key_header, enabled=True)
    key = client.post(f"{API_ROOT}/keys", json={"key_name": "doubled"}, headers=master_key_header).json()

    with _mock_decision():
        response = client.post(RECOMMEND, json=SPAWN, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert response.status_code == status.HTTP_200_OK, response.text
    assert probe.QUOTED == [uuid.UUID(added["id"])]
    (row,) = _usage_rows(client, master_key_header, key["user_id"])
    assert Decimal(str(row["cost"])) == Decimal(str(TOKEN_PRICE)) * 2
