"""A service key mints keys bound to one of its end users, for jobs it does not trust with its own.

A minted key bills its end user whatever the request names, is capped by that end
user's budget and pooled under the service key's ceiling, and stops working with
its expiry, its revocation, or the service key it came from.
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.api.deps import build_api_key_service, get_unit_of_work
from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember
from gateway.models.tenancy import User as Identity
from gateway.models.usage import UsageLog
from gateway.models.users import User
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

from .conftest import MODEL_NAME, _to_async_url

END_USER_KEYS = f"{API_ROOT}/end-user-keys"
KEYS = f"{API_ROOT}/keys"


async def _mock_acompletion(**_kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-x",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat(client: TestClient, headers: dict[str, str], *, user: str | None = None, **body: Any) -> Any:
    payload: dict[str, Any] = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], **body}
    if user is not None:
        payload["user"] = user
    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _mock_acompletion
        return client.post(f"{API_ROOT}/chat/completions", json=payload, headers=headers)


def _bearer(secret: str) -> dict[str, str]:
    return {API_KEY_HEADER: f"Bearer {secret}"}


def _setup(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A price for the model, and two per-run budgets: one request each, or five."""
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 1.0, "output_price_per_million": 1.0},
        headers=master_key_header,
    )
    for budget_id, limit in (("run-one", 1), ("run-five", 5), ("run-other", 5)):
        created = client.put(
            f"{API_ROOT}/budgets/{budget_id}",
            json={"budget_duration_sec": 86400, "request_limit": limit},
            headers=master_key_header,
        )
        assert created.status_code in (200, 201), created.text


def _service_key(
    client: TestClient, master_key_header: dict[str, str], owner: str = "agent", **settings: Any
) -> tuple[str, dict[str, str]]:
    body = {
        "key_name": owner,
        "user_id": owner,
        "is_service_key": True,
        "end_user_budget_ids": ["run-one", "run-five"],
        "end_user_budget_id": "run-five",
        "max_child_ttl_sec": 3600,
        **settings,
    }
    created = client.post(KEYS, json=body, headers=master_key_header)
    assert created.status_code == 200, created.text
    return created.json()["id"], _bearer(created.json()["key"])


def _mint(client: TestClient, service: dict[str, str], external_id: str, **fields: Any) -> Any:
    return client.post(
        END_USER_KEYS, json={"external_id": external_id, "expires_in_sec": 600, **fields}, headers=service
    )


def _minted(client: TestClient, service: dict[str, str], external_id: str, **fields: Any) -> tuple[str, dict[str, str]]:
    response = _mint(client, service, external_id, **fields)
    assert response.status_code == 201, response.text
    return response.json()["id"], _bearer(response.json()["key"])


def _end_user(db_session: Session, external_id: str, owner: str = "agent") -> User:
    db_session.expire_all()
    return db_session.query(User).filter(User.parent_user_id == owner, User.external_id == external_id).one()


def _set_key(db_session: Session, key_id: str, *, expires_at: datetime | None) -> None:
    db_session.query(APIKey).filter(APIKey.id == key_id).update({APIKey.expires_at: expires_at})
    db_session.commit()


# =============================================================================
# Who may mint, list and revoke
# =============================================================================


def test_a_service_key_mints_lists_and_revokes_its_keys(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header)

    minted = _mint(client, service, "run-1", budget_id="run-one", tags={"agent": "bug-fix"}, key_name="run 1")
    assert minted.status_code == 201, minted.text
    body = minted.json()
    assert body["key"].startswith("tk-")
    assert body | {"external_id": "run-1", "budget_id": "run-one", "tags": {"agent": "bug-fix"}} == body
    assert body["key_name"] == "run 1"
    assert body["end_user_id"] == _end_user(db_session, "run-1").user_id
    expires_at = datetime.fromisoformat(body["expires_at"])
    assert timedelta(seconds=590) < expires_at - datetime.now(UTC) <= timedelta(seconds=600)
    assert _chat(client, _bearer(body["key"])).status_code == 200

    other_id, _ = _minted(client, service, "run-2")
    listed = client.get(END_USER_KEYS, headers=service)
    assert listed.status_code == 200, listed.text
    assert [row["id"] for row in listed.json()] == [body["id"], other_id]
    assert "key" not in listed.json()[0]
    one = client.get(END_USER_KEYS, params={"external_id": "run-2"}, headers=service)
    assert [row["id"] for row in one.json()] == [other_id]

    revoked = client.patch(f"{END_USER_KEYS}/{body['id']}", json={"is_active": False}, headers=service)
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["is_active"] is False
    refused = _chat(client, _bearer(body["key"]))
    assert refused.status_code == 401
    assert [row["id"] for row in client.get(END_USER_KEYS, headers=service).json()] == [other_id]
    with_inactive = client.get(END_USER_KEYS, params={"include_inactive": True}, headers=service)
    assert [row["id"] for row in with_inactive.json()] == [body["id"], other_id]
    reactivate = client.patch(f"{END_USER_KEYS}/{body['id']}", json={"is_active": True}, headers=service)
    assert reactivate.status_code == 422


def test_only_a_service_key_allowed_to_mint_reaches_the_routes(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header)
    _, no_ttl = _service_key(client, master_key_header, owner="other", max_child_ttl_sec=None)
    plain = client.post(KEYS, json={"user_id": "plain"}, headers=master_key_header).json()["key"]
    _, child = _minted(client, service, "run-1")

    assert _mint(client, no_ttl, "run-1").status_code == 403
    assert _mint(client, _bearer(plain), "run-1").status_code == 403
    assert _mint(client, child, "run-1").status_code == 403
    assert client.get(END_USER_KEYS, headers=child).status_code == 403
    assert _mint(client, master_key_header, "run-1").status_code == 401
    assert client.get(END_USER_KEYS, headers=master_key_header).status_code == 401

    session = db_session_factory()
    try:
        organization_id = session.query(Organization).filter(col(Organization.slug) == "default").one().id
        operator = Identity(email="op@example.com", full_name="Op", active_organization_id=organization_id)
        operator.is_superuser = True
        session.add(operator)
        session.commit()
        session.refresh(operator)
        session.add(
            OrganizationMember(organization_id=organization_id, user_id=operator.id, role="owner", status="active")
        )
        session.add(
            DashboardSession(
                token_hash=hash_session_token("otari-sess-op"),
                user_id=operator.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )
        session.commit()
    finally:
        session.close()
    client.cookies.set(SESSION_COOKIE_NAME, "otari-sess-op")
    try:
        assert client.post(END_USER_KEYS, json={"external_id": "x", "expires_in_sec": 60}).status_code == 401
        assert client.get(END_USER_KEYS).status_code == 401
    finally:
        client.cookies.clear()


def test_a_mint_stays_within_the_service_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header, allowed_models=[MODEL_NAME])

    too_long = _mint(client, service, "run-1", expires_in_sec=3601)
    assert too_long.status_code == 400
    assert "max_child_ttl_sec" in too_long.json()["detail"]
    off_list = _mint(client, service, "run-1", budget_id="run-other")
    assert off_list.status_code == 403
    assert off_list.headers["Otari-Error-Code"] == "end_user_budget_not_allowed"
    wider = _mint(client, service, "run-1", allowed_models=["gemini:*"])
    assert wider.status_code == 400
    too_many_tags = _mint(client, service, "run-1", tags={f"k{n}": "v" for n in range(17)})
    assert too_many_tags.status_code == 422
    assert _mint(client, service, "run-1", allowed_models=[MODEL_NAME]).status_code == 201


def test_a_service_key_reaches_only_the_keys_it_minted(client: TestClient, master_key_header: dict[str, str]) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header)
    _, sibling = _service_key(client, master_key_header, owner="agent")
    child_id, _ = _minted(client, service, "run-1")

    assert client.get(END_USER_KEYS, headers=sibling).json() == []
    assert client.patch(f"{END_USER_KEYS}/{child_id}", json={"is_active": False}, headers=sibling).status_code == 404


# =============================================================================
# The request path
# =============================================================================


def test_a_minted_key_bills_its_end_user_whatever_the_body_names(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header, reject_user_mismatch=True)
    _, child = _minted(client, service, "run-1", tags={"agent": "bug-fix", "run": "1"})

    named = _chat(client, child, user='{"device_id":"telemetry"}', metadata={"run": "spoofed", "team": "x"})
    assert named.status_code == 200, named.text

    end_user = _end_user(db_session, "run-1")
    assert end_user.current_requests == 1
    rows = db_session.query(UsageLog).filter(UsageLog.user_id == end_user.user_id).all()
    assert len(rows) == 1
    assert rows[0].tags == {"agent": "bug-fix", "run": "1", "team": "x"}
    owner = db_session.query(User).filter(User.user_id == "agent").one()
    assert owner.current_requests == 0
    assert db_session.query(User).filter(User.external_id == '{"device_id":"telemetry"}').count() == 0


def test_a_minted_key_is_refused_by_its_end_users_budget(client: TestClient, master_key_header: dict[str, str]) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header)
    _, child = _minted(client, service, "run-1", budget_id="run-one")

    assert _chat(client, child).status_code == 200
    refused = _chat(client, child)
    assert refused.status_code == 403, refused.text


def test_minted_keys_are_pooled_under_the_service_keys_ceiling(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    created = client.put(
        f"{API_ROOT}/budgets/agent-day",
        json={"budget_duration_sec": 86400, "request_limit": 2},
        headers=master_key_header,
    )
    assert created.status_code in (200, 201), created.text
    ceiling = client.post(
        f"{API_ROOT}/scoped-budgets",
        json={"scope_type": "api_token", "scope_id": service_id, "budget_id": "agent-day"},
        headers=master_key_header,
    )
    assert ceiling.status_code == 200, ceiling.text
    _, first = _minted(client, service, "run-1")
    _, second = _minted(client, service, "run-2")

    assert _chat(client, first).status_code == 200
    assert _chat(client, second).status_code == 200
    refused = _chat(client, second)
    assert refused.status_code == 403, refused.text
    assert "API key" in refused.text


def test_a_minted_key_reaches_only_the_models_its_service_key_does(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    _, child = _minted(client, service, "run-1")
    assert _chat(client, child).status_code == 200

    narrowed = client.patch(
        f"{KEYS}/{service_id}", json={"allowed_models": ["gemini:other-model"]}, headers=master_key_header
    )
    assert narrowed.status_code == 200, narrowed.text
    assert _chat(client, child).status_code == 403


# =============================================================================
# Lifecycle
# =============================================================================


def test_a_minted_key_outlives_neither_its_expiry_nor_its_service_key(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    expired_id, expired = _minted(client, service, "run-1")
    _, live = _minted(client, service, "run-2")
    _set_key(db_session, expired_id, expires_at=datetime.now(UTC) - timedelta(seconds=1))

    assert _chat(client, expired).status_code == 401
    assert _chat(client, live).status_code == 200

    _set_key(db_session, service_id, expires_at=datetime.now(UTC) - timedelta(seconds=1))
    assert _chat(client, live).status_code == 401
    _set_key(db_session, service_id, expires_at=None)
    assert _chat(client, live).status_code == 200

    client.patch(f"{KEYS}/{service_id}", json={"is_active": False}, headers=master_key_header)
    assert _chat(client, live).status_code == 401
    client.patch(f"{KEYS}/{service_id}", json={"is_active": True}, headers=master_key_header)
    assert _chat(client, live).status_code == 200

    blocked = client.patch(f"{API_ROOT}/users/agent", json={"blocked": True}, headers=master_key_header)
    assert blocked.status_code == 200, blocked.text
    assert _chat(client, live).status_code == 401
    client.patch(f"{API_ROOT}/users/agent", json={"blocked": False}, headers=master_key_header)

    assert client.delete(f"{KEYS}/{service_id}", headers=master_key_header).status_code == 204
    assert _chat(client, live).status_code == 401
    db_session.expire_all()
    assert db_session.query(APIKey).filter(APIKey.parent_key_id == service_id).count() == 0


def test_the_sweep_deletes_expired_minted_keys_and_keeps_their_attribution(
    client: TestClient, master_key_header: dict[str, str], db_session: Session, postgres_url: str
) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header)
    old_id, old = _minted(client, service, "run-1")
    recent_id, _ = _minted(client, service, "run-2")
    live_id, _ = _minted(client, service, "run-3")
    assert _chat(client, old).status_code == 200
    _set_key(db_session, old_id, expires_at=datetime.now(UTC) - timedelta(days=31))
    _set_key(db_session, recent_id, expires_at=datetime.now(UTC) - timedelta(days=1))

    async def sweep() -> int:
        engine = create_async_engine(_to_async_url(postgres_url))
        try:
            async with async_sessionmaker(engine, expire_on_commit=False)() as session:
                return await build_api_key_service(get_unit_of_work(session)).sweep_expired_end_user_keys(30 * 86400)
        finally:
            await engine.dispose()

    assert asyncio.run(sweep()) == 1
    db_session.expire_all()
    remaining = {key.id for key in db_session.query(APIKey).filter(APIKey.parent_key_id.is_not(None))}
    assert remaining == {recent_id, live_id}
    end_user = _end_user(db_session, "run-1")
    rows = db_session.query(UsageLog).filter(UsageLog.user_id == end_user.user_id).all()
    assert [(row.api_key_id, row.user_id) for row in rows] == [(None, end_user.user_id)]
    read = client.get(f"{API_ROOT}/end-users", params={"external_id": "run-1"}, headers=service)
    assert read.json()[0]["current_requests"] == 1


# =============================================================================
# The operator's view
# =============================================================================


def test_the_key_list_leaves_minted_keys_out_unless_asked(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    child_id, _ = _minted(client, service, "run-1")

    default = {row["id"] for row in client.get(KEYS, headers=master_key_header).json()}
    assert service_id in default
    assert child_id not in default
    everything = client.get(KEYS, params={"include_children": True}, headers=master_key_header).json()
    child = next(row for row in everything if row["id"] == child_id)
    assert child["parent_key_id"] == service_id

    listed = client.get(f"{KEYS}/{service_id}/end-user-keys", headers=master_key_header)
    assert [row["id"] for row in listed.json()] == [child_id]
    revoked = client.patch(
        f"{KEYS}/{service_id}/end-user-keys/{child_id}", json={"is_active": False}, headers=master_key_header
    )
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["is_active"] is False


def test_a_minted_key_cannot_be_widened_by_an_update(client: TestClient, master_key_header: dict[str, str]) -> None:
    _setup(client, master_key_header)
    _, service = _service_key(client, master_key_header)
    child_id, _ = _minted(client, service, "run-1")

    for change in ({"is_service_key": True}, {"max_child_ttl_sec": 60}, {"expires_at": None}):
        refused = client.patch(f"{KEYS}/{child_id}", json=change, headers=master_key_header)
        assert refused.status_code == 400, (change, refused.text)
    assert (
        client.patch(f"{KEYS}/{child_id}", json={"key_name": "renamed"}, headers=master_key_header).status_code == 200
    )


def test_max_child_ttl_sec_needs_a_service_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    _setup(client, master_key_header)
    refused = client.post(KEYS, json={"user_id": "plain", "max_child_ttl_sec": 60}, headers=master_key_header)
    assert refused.status_code == 400
    service_id, _ = _service_key(client, master_key_header)
    turned_off = client.patch(f"{KEYS}/{service_id}", json={"is_service_key": False}, headers=master_key_header)
    assert turned_off.status_code == 200, turned_off.text
    assert turned_off.json()["max_child_ttl_sec"] is None


def test_a_minted_key_takes_no_ceiling_of_its_own(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Nothing removes a ceiling when a minted key goes, by deletion or the sweep, so none is placed."""
    _setup(client, master_key_header)
    service_id, service = _service_key(client, master_key_header)
    child_id, _ = _minted(client, service, "run-1")

    deployment = client.post(
        f"{API_ROOT}/scoped-budgets",
        json={"scope_type": "api_token", "scope_id": child_id, "budget_id": "run-five"},
        headers=master_key_header,
    )
    assert deployment.status_code == 400, deployment.text

    budget = client.post(
        f"{API_ROOT}/organizations/me/budgets",
        json={"name": "Per key", "max_budget": 10.0, "reset_alignment": "calendar_month"},
        headers=master_key_header,
    )
    assert budget.status_code == 201, budget.text
    organization = client.post(
        f"{API_ROOT}/organizations/me/spend-ceilings",
        json={"scope_type": "api_token", "scope_id": child_id, "budget_id": budget.json()["budget_id"]},
        headers=master_key_header,
    )
    assert organization.status_code == 400, organization.text
    on_service = client.post(
        f"{API_ROOT}/organizations/me/spend-ceilings",
        json={"scope_type": "api_token", "scope_id": service_id, "budget_id": budget.json()["budget_id"]},
        headers=master_key_header,
    )
    assert on_service.status_code == 201, on_service.text
