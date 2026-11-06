"""The calling API key reads its own budget and usage, and nothing beyond them."""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from conftest import seed_workspace_id
from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.api_keys import APIKey
from gateway.models.budgets import ScopedBudget
from gateway.models.usage import UsageLog
from gateway.models.users import User
from gateway.repositories.tenancy import OrganizationMemberRepository, WorkspaceMemberRepository
from gateway.repositories.users_repository import get_or_create_attribution_user
from gateway.services.budgets import BudgetScopeRequest
from gateway.services.budgets._scoped_enforcement import get_scope_ceilings

from .tenancy_helpers import budget_service, create_budget, create_member, create_organization, create_workspace


def _key(client: TestClient, master_key_header: dict[str, str], *, user_id: str, **fields: Any) -> dict[str, Any]:
    response = client.post(
        f"{API_ROOT}/keys", json={"key_name": f"key-{user_id}", "user_id": user_id} | fields, headers=master_key_header
    )
    assert response.status_code == 200, response.text
    created: dict[str, Any] = response.json()
    return created


def _auth(key: dict[str, Any]) -> dict[str, str]:
    return {API_KEY_HEADER: f"Bearer {key['key']}"}


def _user_on_budget(client: TestClient, master_key_header: dict[str, str], user_id: str, **budget: Any) -> str:
    created = client.post(f"{API_ROOT}/budgets", json=budget, headers=master_key_header)
    assert created.status_code == 200, created.text
    budget_id: str = created.json()["budget_id"]
    response = client.post(
        f"{API_ROOT}/users", json={"user_id": user_id, "budget_id": budget_id}, headers=master_key_header
    )
    assert response.status_code == 200, response.text
    return budget_id


def _spend(db_session_factory: Callable[[], Session], user_id: str, spend: str) -> None:
    with db_session_factory() as db:
        db.execute(update(User).where(User.user_id == user_id).values(spend=Decimal(spend)))
        db.commit()


def test_a_key_reads_what_its_users_budget_leaves(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _user_on_budget(client, master_key_header, "volunteer", max_budget=0.5, budget_duration_sec=30 * 86400)
    key = _key(client, master_key_header, user_id="volunteer")
    _spend(db_session_factory, "volunteer", "0.31")

    response = client.get(f"{API_ROOT}/key/budget", headers=_auth(key))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["budgets_enforced"] is True
    assert body["remaining_usd"] == 0.19
    [limit] = body["limits"]
    assert limit["scope"] == "user"
    assert limit["provider"] is None
    assert (limit["limit_usd"], limit["used_usd"], limit["remaining_usd"]) == (0.5, 0.31, 0.19)
    assert limit["resets_at"] is not None


def test_a_spent_budget_reads_as_nothing_left_and_the_gate_refuses(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    test_messages: list[dict[str, str]],
) -> None:
    """What the read reports and what the gate does agree."""
    _user_on_budget(client, master_key_header, "spent", max_budget=0.5)
    key = _key(client, master_key_header, user_id="spent")
    _spend(db_session_factory, "spent", "0.5")
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=master_key_header,
    )

    assert client.get(f"{API_ROOT}/key/budget", headers=_auth(key)).json()["remaining_usd"] == 0.0
    refused = client.post(
        f"{API_ROOT}/chat/completions", json={"model": "openai:gpt-4o", "messages": test_messages}, headers=_auth(key)
    )
    assert refused.status_code == 403
    assert refused.headers["Otari-Error-Code"] == "budget_exceeded"


def test_an_ended_period_reads_as_the_fresh_one(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _user_on_budget(client, master_key_header, "rolled", max_budget=1.0, budget_duration_sec=86400)
    key = _key(client, master_key_header, user_id="rolled")
    with db_session_factory() as db:
        db.execute(
            update(User)
            .where(User.user_id == "rolled")
            .values(spend=Decimal("0.9"), next_budget_reset_at=datetime.now(UTC) - timedelta(hours=1))
        )
        db.commit()

    [limit] = client.get(f"{API_ROOT}/key/budget", headers=_auth(key)).json()["limits"]

    assert limit["used_usd"] == 0.0
    assert limit["remaining_usd"] == 1.0
    assert datetime.fromisoformat(limit["resets_at"]) > datetime.now(UTC)


def test_a_hold_in_flight_counts_as_used(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """The gate refuses on spend plus holds, so the read counts both, and keeps the difference exact."""
    _user_on_budget(client, master_key_header, "holding", max_budget=0.3, budget_duration_sec=86400)
    key = _key(client, master_key_header, user_id="holding")
    with db_session_factory() as db:
        db.execute(
            update(User).where(User.user_id == "holding").values(spend=Decimal("0.05"), reserved=Decimal("0.05"))
        )
        db.commit()

    [limit] = client.get(f"{API_ROOT}/key/budget", headers=_auth(key)).json()["limits"]

    assert (limit["used_usd"], limit["remaining_usd"]) == (0.1, 0.2)


def test_an_ended_period_keeps_the_holds_still_in_flight(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """Rolling a window zeroes recorded spend and leaves holds, so the fresh window starts with them."""
    _user_on_budget(client, master_key_header, "rolled-holding", max_budget=1.0, budget_duration_sec=86400)
    key = _key(client, master_key_header, user_id="rolled-holding")
    with db_session_factory() as db:
        db.execute(
            update(User)
            .where(User.user_id == "rolled-holding")
            .values(
                spend=Decimal("0.9"),
                reserved=Decimal("0.1"),
                next_budget_reset_at=datetime.now(UTC) - timedelta(hours=1),
            )
        )
        db.commit()

    [limit] = client.get(f"{API_ROOT}/key/budget", headers=_auth(key)).json()["limits"]

    assert (limit["used_usd"], limit["remaining_usd"]) == (0.1, 0.9)


def test_a_shared_ceiling_reports_only_what_it_leaves(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """The key's own ceiling is shown in full; the workspace's spend is the team's, so only its room is."""
    key = _key(client, master_key_header, user_id="member")
    with db_session_factory() as db:
        workspace_id = str(seed_workspace_id(db))
    caps = {"token_limit": 1000, "request_limit": 10}
    own = client.post(f"{API_ROOT}/budgets", json={"max_budget": 2.0, **caps}, headers=master_key_header)
    team = client.post(f"{API_ROOT}/budgets", json={"max_budget": 10.0, **caps}, headers=master_key_header)
    for scope_type, scope_id, budget_id in (
        ("api_token", key["id"], own.json()["budget_id"]),
        ("workspace", workspace_id, team.json()["budget_id"]),
    ):
        response = client.post(
            f"{API_ROOT}/scoped-budgets",
            json={"scope_type": scope_type, "scope_id": scope_id, "budget_id": budget_id},
            headers=master_key_header,
        )
        assert response.status_code in (200, 201), response.text

    body = client.get(f"{API_ROOT}/key/budget", headers=_auth(key)).json()

    by_scope = {limit["scope"]: limit for limit in body["limits"]}
    assert [limit["scope"] for limit in body["limits"]] == ["api_token", "workspace"]
    assert (by_scope["api_token"]["limit_usd"], by_scope["api_token"]["remaining_usd"]) == (2.0, 2.0)
    for axis in ("tokens", "requests"):
        assert by_scope["api_token"][f"limit_{axis}"] == caps[f"{axis[:-1]}_limit"]
        assert by_scope["api_token"][f"used_{axis}"] == 0
        assert by_scope["api_token"][f"remaining_{axis}"] == caps[f"{axis[:-1]}_limit"]
    for field in ("limit_usd", "used_usd", "limit_tokens", "used_tokens", "limit_requests", "used_requests"):
        assert by_scope["workspace"][field] is None, field
    assert by_scope["workspace"]["remaining_usd"] == 10.0
    assert by_scope["workspace"]["remaining_tokens"] == 1000
    assert by_scope["workspace"]["remaining_requests"] == 10
    assert body["remaining_usd"] == 2.0


def test_a_budget_exempt_key_is_not_gated(client: TestClient, master_key_header: dict[str, str]) -> None:
    _user_on_budget(client, master_key_header, "exempt", max_budget=0.5)
    key = _key(client, master_key_header, user_id="exempt", exclude_from_budget=True)

    body = client.get(f"{API_ROOT}/key/budget", headers=_auth(key)).json()

    assert body == {"budgets_enforced": False, "remaining_usd": None, "limits": []}


def test_only_an_api_key_may_ask(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The master key has no key of its own to answer for."""
    assert client.get(f"{API_ROOT}/key/budget").status_code == 401
    assert client.get(f"{API_ROOT}/key/budget", headers=master_key_header).status_code == 401
    assert client.get(f"{API_ROOT}/key/usage/summary", headers=master_key_header).status_code == 401


def _row(workspace_id: Any, api_key_id: str, cost: str, role: str, app: str = "claim-check") -> UsageLog:
    return UsageLog(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        api_key_id=api_key_id,
        timestamp=datetime.now(UTC) - timedelta(minutes=5),
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        status="success",
        cost=Decimal(cost),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        tags={"app": app, "role": role},
        request_group_id=str(uuid.uuid4()),
    )


def test_a_key_summarizes_its_own_usage_by_tag(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    mine = _key(client, master_key_header, user_id="one")
    theirs = _key(client, master_key_header, user_id="two")
    with db_session_factory() as db:
        workspace_id = seed_workspace_id(db)
        db.add_all(
            [
                _row(workspace_id, mine["id"], "0.10", "research"),
                _row(workspace_id, mine["id"], "0.20", "jury"),
                _row(workspace_id, mine["id"], "0.05", "jury"),
                _row(workspace_id, mine["id"], "0.50", "jury", app="other"),
                _row(workspace_id, theirs["id"], "1.00", "jury"),
            ]
        )
        db.commit()

    response = client.get(
        f"{API_ROOT}/key/usage/summary",
        params={"group_by_tag": "role", "tag": "app:claim-check", "dimensions": "none"},
        headers=_auth(mine),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["totals"]["request_count"] == 3
    assert round(body["totals"]["cost"], 6) == 0.35
    assert {row["key"]: round(row["cost"], 6) for row in body["by_tag"]} == {"jury": 0.25, "research": 0.1}

    unfiltered = client.get(f"{API_ROOT}/key/usage/summary", params={"dimensions": "none"}, headers=_auth(mine))
    assert unfiltered.json()["totals"]["request_count"] == 4
    other = client.get(f"{API_ROOT}/key/usage/summary", params={"dimensions": "none"}, headers=_auth(theirs))
    assert other.json()["totals"]["request_count"] == 1


def test_a_key_does_not_read_a_sibling_keys_series(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """Two keys of one user: the scope is the key, not the user it bills."""
    mine = _key(client, master_key_header, user_id="series-owner")
    theirs = _key(client, master_key_header, user_id="series-owner", key_name="sibling")
    with db_session_factory() as db:
        workspace_id = seed_workspace_id(db)
        db.add_all([_row(workspace_id, mine["id"], "0.10", "jury"), _row(workspace_id, theirs["id"], "1.00", "jury")])
        db.commit()

    response = client.get(f"{API_ROOT}/key/usage/series", params={"group_by": "api_key_id"}, headers=_auth(mine))

    assert response.status_code == 200, response.text
    assert [group["key"] for group in response.json()["groups"]] == [mine["id"]]


def test_a_malformed_tag_filter_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    key = _key(client, master_key_header, user_id="tags")

    response = client.get(f"{API_ROOT}/key/usage/summary", params={"tag": "no-colon"}, headers=_auth(key))

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_a_members_key_lists_the_ceilings_the_gate_checks(async_db: AsyncSession) -> None:
    """The read resolves a key's scopes on its own, so it is held to the gate's resolution here."""
    organization = await create_organization(async_db, slug="claim-check")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    volunteer = await create_member(async_db, organization, role="member", full_name="Volunteer")
    workspace = await create_workspace(async_db, organization, name="Checks", owner=owner)
    workspace_member = await WorkspaceMemberRepository(async_db).create(
        workspace_id=workspace.id, user_id=volunteer.id, role="member"
    )
    organization_member = await OrganizationMemberRepository(async_db).get_by_organization_and_user(
        organization.id, volunteer.id
    )
    assert organization_member is not None
    await get_or_create_attribution_user(async_db, user_id=str(volunteer.id), alias="Volunteer")
    key = APIKey(id="volunteer-key", key_hash="hash-volunteer", workspace_id=workspace.id, user_id=str(volunteer.id))
    async_db.add(key)
    for scope_type, scope_id in (
        ("workspace_member", str(workspace_member.id)),
        ("org_member", str(organization_member.id)),
        ("api_token", key.id),
        ("workspace", str(workspace.id)),
        ("organization", str(organization.id)),
    ):
        async_db.add(
            ScopedBudget(
                scope_type=scope_type, scope_id=scope_id, budget_id=await create_budget(async_db, max_budget=5.0)
            )
        )
    await async_db.commit()

    gate = await get_scope_ceilings(async_db, user_id=str(volunteer.id), scope=BudgetScopeRequest(api_key=key))
    read = await budget_service(async_db).key_budget(key, strategy=None)

    assert [limit.scope for limit in read.limits] == [
        "workspace_member",
        "org_member",
        "api_token",
        "workspace",
        "organization",
    ]
    assert sorted(limit.scope for limit in read.limits) == sorted(ceiling.scope_type for ceiling in gate)
