"""Personal access tokens read the management API as their owner, never write, and back the MCP server."""

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.api.deps.auth import build_personal_access_token_service
from gateway.core.config import API_ROOT
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import PersonalAccessTokenSessionEndedError
from gateway.models.identity import BlockedEmail, PersonalAccessToken
from gateway.models.tenancy import OrganizationMember, User
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

from .conftest import MODEL_NAME
from .test_organization_member_keys import _World
from .test_organization_member_keys import world as world

_TOKENS = f"{API_ROOT}/auth/access-tokens"
_MCP_HEADERS = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}


def _session(world: _World, who: str) -> dict[str, str]:
    return {"Cookie": f"{SESSION_COOKIE_NAME}={world.sessions[who]}"}


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _mint(client: TestClient, world: _World, who: str, **body: Any) -> dict[str, Any]:
    response = client.post(_TOKENS, json={"name": "agent", **body}, headers=_session(world, who))
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


def _mcp(client: TestClient, headers: dict[str, str], method: str, params: dict[str, Any] | None = None) -> Any:
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
        headers={**_MCP_HEADERS, **headers},
    )


def _call_tool(client: TestClient, token: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = _mcp(client, _bearer(token), "tools/call", {"name": name, "arguments": arguments})
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()["result"]
    return result


def test_a_token_is_shown_once_and_listed_without_it(client: TestClient, world: _World) -> None:
    created = _mint(client, world, "alpha_member", expiration="7-days")
    assert created["token"].startswith("otari-pat-")
    assert created["token"].endswith(created["token_suffix"])
    assert created["expires_at"] is not None

    listed = client.get(_TOKENS, headers=_session(world, "alpha_member")).json()["data"]
    assert [row["id"] for row in listed] == [created["id"]]
    assert "token" not in listed[0]
    assert client.get(_TOKENS, headers=_session(world, "alpha_colleague")).json()["data"] == []


def test_a_token_reads_as_its_owner(client: TestClient, world: _World) -> None:
    token = _mint(client, world, "alpha_member")["token"]

    context = client.get(f"{API_ROOT}/organizations/me", headers=_bearer(token))
    assert context.status_code == 200, context.text
    assert context.json()["organization"]["id"] == str(world.alpha)
    assert context.json()["role"] == "member"
    assert client.get(f"{API_ROOT}/organizations/me/usage/summary", headers=_bearer(token)).status_code == 200
    # Narrowing to a workspace of another organization answers as it does for the owner's session.
    beta = {"workspace_id": str(world.workspaces["beta_one"])}
    usage = f"{API_ROOT}/organizations/me/usage"
    assert client.get(usage, params=beta, headers=_session(world, "alpha_member")).status_code == 404
    assert client.get(usage, params=beta, headers=_bearer(token)).status_code == 404


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", _TOKENS, {"name": "successor"}),
        ("POST", f"{API_ROOT}/workspaces", {"name": "new"}),
        ("PATCH", f"{API_ROOT}/auth/profile", {"full_name": "Renamed"}),
    ],
)
def test_a_token_never_writes(
    client: TestClient, world: _World, method: str, path: str, body: dict[str, Any] | None
) -> None:
    token = _mint(client, world, "alpha_owner")["token"]
    response = client.request(method, path, json=body, headers=_bearer(token))
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Access tokens are read-only"


def test_a_token_is_not_an_inference_credential(client: TestClient, world: _World) -> None:
    token = _mint(client, world, "alpha_member")["token"]
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hello"}]},
        headers=_bearer(token),
    )
    assert response.status_code == 401


def test_only_the_owner_revokes_a_token_and_it_stops_at_once(client: TestClient, world: _World) -> None:
    created = _mint(client, world, "alpha_member")
    headers = _bearer(created["token"])
    assert client.delete(f"{_TOKENS}/{created['id']}", headers=_session(world, "alpha_colleague")).status_code == 404
    assert client.get(f"{API_ROOT}/organizations/me", headers=headers).status_code == 200

    assert client.delete(f"{_TOKENS}/{created['id']}", headers=_session(world, "alpha_member")).status_code == 204
    assert client.get(f"{API_ROOT}/organizations/me", headers=headers).status_code == 401


def test_an_expired_token_is_refused(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    created = _mint(client, world, "alpha_member", expiration="1-hour")
    with db_session_factory() as db:
        db.execute(
            update(PersonalAccessToken)
            .where(PersonalAccessToken.token_hash != "")
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        db.commit()
    response = client.get(f"{API_ROOT}/organizations/me", headers=_bearer(created["token"]))
    assert response.status_code == 401


def test_deactivating_the_owner_ends_their_tokens(client: TestClient, world: _World) -> None:
    token = _mint(client, world, "alpha_member")["token"]
    deactivated = client.patch(
        f"{API_ROOT}/admin/users/{world.users['alpha_member']}",
        json={"is_active": False},
        headers=_session(world, "alpha_operator"),
    )
    assert deactivated.status_code == 200, deactivated.text
    assert client.get(f"{API_ROOT}/organizations/me", headers=_bearer(token)).status_code == 401


def test_a_token_always_ends(client: TestClient, world: _World) -> None:
    refused = client.post(
        _TOKENS, json={"name": "forever", "expiration": "never"}, headers=_session(world, "alpha_member")
    )
    assert refused.status_code == 422


def test_a_token_reads_the_organization_it_was_made_in(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    created = _mint(client, world, "alpha_member")
    assert created["organization_id"] == str(world.alpha)
    with db_session_factory() as db:
        db.execute(
            update(User).where(col(User.id) == world.users["alpha_member"]).values(active_organization_id=world.beta)
        )
        db.commit()
    context = client.get(f"{API_ROOT}/organizations/me", headers=_bearer(created["token"]))
    assert context.status_code == 200, context.text
    assert context.json()["organization"]["id"] == str(world.alpha)
    with db_session_factory() as db:
        active = db.get(User, world.users["alpha_member"])
        assert active is not None and active.active_organization_id == world.beta


def test_a_token_stops_with_the_membership_it_was_made_for(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    created = _mint(client, world, "alpha_member")
    assert client.get(f"{API_ROOT}/organizations/me", headers=_bearer(created["token"])).status_code == 200
    with db_session_factory() as db:
        db.execute(
            delete(OrganizationMember).where(
                col(OrganizationMember.organization_id) == world.alpha,
                col(OrganizationMember.user_id) == world.users["alpha_member"],
            )
        )
        db.commit()
    # Every read resolves its organization from the pointer the token sets, and falls back to
    # another organization when the membership behind it is gone; the token is bound to this one.
    assert client.get(f"{API_ROOT}/organizations/me", headers=_bearer(created["token"])).status_code == 401


def test_a_token_needs_an_organization_to_read(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    with db_session_factory() as db:
        db.execute(update(User).where(col(User.id) == world.users["alpha_member"]).values(active_organization_id=None))
        db.commit()
    refused = client.post(_TOKENS, json={"name": "nowhere"}, headers=_session(world, "alpha_member"))
    assert refused.status_code == 400, refused.text
    assert "organization" in refused.json()["detail"]


def test_a_blocked_address_is_never_given_a_token(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    # Staged without the session cleanup a block does, which is the state a creation overlapping
    # a block reaches: the request authenticated before the entry committed.
    with db_session_factory() as db:
        db.add(BlockedEmail(pattern="member@alpha.test", reason=None, created_by_user_id=None))
        db.commit()
    refused = client.post(_TOKENS, json={"name": "blocked"}, headers=_session(world, "alpha_member"))
    assert refused.status_code == 403, refused.text
    assert "blocked" in refused.json()["detail"]


@pytest.mark.asyncio
async def test_a_session_that_ended_cannot_make_a_token(async_db: AsyncSession, world: _World) -> None:
    """The creation re-reads the session it authenticated with, so a cleanup cannot be outrun."""
    owner = await async_db.get(User, world.users["alpha_member"])
    assert owner is not None
    service = build_personal_access_token_service(UnitOfWork(async_db))
    with pytest.raises(PersonalAccessTokenSessionEndedError):
        await service.create(owner, name="late", lifetime=timedelta(days=1), session_token="not-a-live-session")


def test_an_ended_token_makes_room_for_a_new_one(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    ended = _mint(client, world, "alpha_member", expiration="1-hour")
    with db_session_factory() as db:
        db.execute(
            update(PersonalAccessToken)
            .where(PersonalAccessToken.token_hash != "")
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        db.commit()
    fresh = _mint(client, world, "alpha_member")
    listed = client.get(_TOKENS, headers=_session(world, "alpha_member")).json()["data"]
    assert [row["id"] for row in listed] == [fresh["id"]]
    assert ended["id"] != fresh["id"]


def test_a_token_cannot_make_a_provider_call_through_a_read(client: TestClient, world: _World) -> None:
    token = _mint(client, world, "alpha_owner")["token"]
    path = f"{API_ROOT}/organizations/me/provider-keys/{uuid.uuid4()}/available-models"
    response = client.get(path, headers=_bearer(token))
    assert response.status_code == 403
    assert response.json()["detail"] == "Access tokens cannot use this endpoint"


def test_a_password_change_ends_the_tokens_minted_before_it(client: TestClient, world: _World) -> None:
    token = _mint(client, world, "alpha_member")["token"]
    changed = client.put(
        f"{API_ROOT}/auth/password", json={"new_password": "a-new-password"}, headers=_session(world, "alpha_member")
    )
    assert changed.status_code == 200, changed.text
    assert client.get(f"{API_ROOT}/organizations/me", headers=_bearer(token)).status_code == 401
    assert client.get(_TOKENS, headers=_session(world, "alpha_member")).json()["data"] == []


def test_reactivating_an_account_does_not_bring_its_tokens_back(client: TestClient, world: _World) -> None:
    token = _mint(client, world, "alpha_member")["token"]
    user = f"{API_ROOT}/admin/users/{world.users['alpha_member']}"
    for is_active in (False, True):
        changed = client.patch(user, json={"is_active": is_active}, headers=_session(world, "alpha_operator"))
        assert changed.status_code == 200, changed.text
    assert client.get(f"{API_ROOT}/organizations/me", headers=_bearer(token)).status_code == 401


def test_the_mcp_server_requires_a_token(client: TestClient) -> None:
    assert _mcp(client, {}, "tools/list").status_code == 401
    assert _mcp(client, _bearer("otari-pat-unknown"), "tools/list").status_code == 401
    assert _mcp(client, _bearer("tk-not-a-personal-token"), "tools/list").status_code == 401


def test_the_mcp_server_lists_read_only_tools(client: TestClient, world: _World) -> None:
    token = _mint(client, world, "alpha_member")["token"]
    response = _mcp(client, _bearer(token), "tools/list")
    assert response.status_code == 200, response.text
    tools = {tool["name"]: tool for tool in response.json()["result"]["tools"]}
    assert {"get_context", "get_overview", "get_usage_summary", "list_requests", "get_request", "api_get"} <= set(tools)
    assert all(tool["annotations"]["readOnlyHint"] for tool in tools.values())


def test_an_mcp_tool_answers_what_the_route_answers_its_owner(client: TestClient, world: _World) -> None:
    token = _mint(client, world, "alpha_member")["token"]
    result = _call_tool(client, token, "get_context", {})
    assert result["isError"] is False
    context = json.loads(result["content"][0]["text"])
    assert context["organization"]["id"] == str(world.alpha)
    assert context["role"] == "member"

    beta = _call_tool(client, token, "list_requests", {"workspace_id": str(world.workspaces["beta_one"])})
    assert beta["isError"] is True
    assert "404" in beta["content"][0]["text"]


def test_api_get_stays_below_the_api_root(client: TestClient, world: _World) -> None:
    token = _mint(client, world, "alpha_member")["token"]
    members = _call_tool(client, token, "api_get", {"path": "/organizations/me/members"})
    assert members["isError"] is False
    escaped = _call_tool(client, token, "api_get", {"path": "/../../dashboard-build.json"})
    assert escaped["isError"] is True
